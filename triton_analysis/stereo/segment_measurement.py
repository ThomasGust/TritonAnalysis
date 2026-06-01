"""Stereo segment measurement helpers.

The applet measures a straight 3D segment from two manually matched endpoint
correspondences in rectified stereo images. Dense disparity can still help with
scene inspection, but the result comes from direct triangulation of the clicked
endpoints.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Sequence

import numpy as np

from triton_analysis.stereo.depth import (
    CorrespondenceSample,
    distance_between_samples,
    point_from_rectified_correspondence,
)


PointLike = Sequence[float] | np.ndarray


@dataclass(frozen=True)
class StereoSegmentPreset:
    """Labels and report wording for a two-endpoint stereo measurement."""

    key: str
    name: str
    start_label: str
    end_label: str
    result_label: str
    report_title: str


STEREO_SEGMENT_PRESETS = (
    StereoSegmentPreset(
        key="generic",
        name="Generic Segment",
        start_label="Start",
        end_label="End",
        result_label="Segment length",
        report_title="Stereo Segment Measurement",
    ),
    StereoSegmentPreset(
        key="iceberg",
        name="Iceberg Keel",
        start_label="Keel top",
        end_label="Keel bottom",
        result_label="Keel length",
        report_title="Stereo Iceberg Keel Measurement",
    ),
    StereoSegmentPreset(
        key="coral",
        name="Coral Rig Length",
        start_label="Rig start",
        end_label="Rig end",
        result_label="Coral rig length",
        report_title="Stereo Coral Rig Length Measurement",
    ),
)


def preset_by_key(key: str | None) -> StereoSegmentPreset:
    """Return a measurement preset by key, defaulting to generic."""

    normalized = str(key or "").strip().lower()
    for preset in STEREO_SEGMENT_PRESETS:
        if preset.key == normalized:
            return preset
    return STEREO_SEGMENT_PRESETS[0]


def _as_image_point(point: PointLike, name: str) -> np.ndarray:
    value = np.asarray(point, dtype=np.float64).reshape(-1)
    if value.shape != (2,):
        raise ValueError(f"{name} must be a 2D point")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must contain finite coordinates")
    return value


def right_endpoint_order_mismatch(
    *,
    left_start_pixel: PointLike,
    left_end_pixel: PointLike,
    right_start_pixel: PointLike,
    right_end_pixel: PointLike,
    min_axis_delta_px: float = 8.0,
) -> bool:
    """Return whether right endpoints appear clicked in the opposite order.

    Rectified stereo preserves the visible ordering of two endpoints along the
    segment's dominant image axis in ordinary measurement cases. Horizontal
    segments are especially easy to mis-click because swapped right endpoints
    can still have nearly perfect vertical epipolar error.
    """

    left_start = _as_image_point(left_start_pixel, "left_start_pixel")
    left_end = _as_image_point(left_end_pixel, "left_end_pixel")
    right_start = _as_image_point(right_start_pixel, "right_start_pixel")
    right_end = _as_image_point(right_end_pixel, "right_end_pixel")

    left_delta = left_end - left_start
    right_delta = right_end - right_start
    combined = np.abs(left_delta) + np.abs(right_delta)
    axis = int(np.argmax(combined))
    if combined[axis] < float(min_axis_delta_px) * 2.0:
        return False
    if abs(left_delta[axis]) < float(min_axis_delta_px) or abs(right_delta[axis]) < float(min_axis_delta_px):
        return False
    return bool(np.sign(left_delta[axis]) != np.sign(right_delta[axis]))


@dataclass(frozen=True)
class StereoSegmentMeasurementResult:
    """One stereo endpoint measurement of a straight segment."""

    length_units: float
    units: str
    start: CorrespondenceSample
    end: CorrespondenceSample
    length_cm: float | None
    length_m: float | None
    preset_key: str = "generic"

    @property
    def max_vertical_error_px(self) -> float:
        return max(abs(self.start.vertical_error_px), abs(self.end.vertical_error_px))

    @property
    def min_abs_disparity_px(self) -> float:
        return min(abs(self.start.disparity), abs(self.end.disparity))

    @property
    def top(self) -> CorrespondenceSample:
        """Backward-compatible alias for iceberg keel measurements."""

        return self.start

    @property
    def bottom(self) -> CorrespondenceSample:
        """Backward-compatible alias for iceberg keel measurements."""

        return self.end


@dataclass(frozen=True)
class StereoSegmentMeasurementSeries:
    """Summary of repeated stereo segment measurements."""

    count: int
    units: str
    median_length_units: float
    min_length_units: float
    max_length_units: float
    spread_units: float
    median_length_cm: float | None
    spread_cm: float | None

    @property
    def median_length_m(self) -> float | None:
        if self.median_length_cm is None:
            return None
        return self.median_length_cm / 100.0


def unit_scale_to_cm(units: str) -> float | None:
    """Return a scale factor from calibration units to centimeters."""

    normalized = str(units or "").strip().lower()
    if normalized in {"mm", "millimeter", "millimeters"}:
        return 0.1
    if normalized in {"cm", "centimeter", "centimeters"}:
        return 1.0
    if normalized in {"m", "meter", "meters"}:
        return 100.0
    return None


def measure_stereo_segment(
    *,
    q: np.ndarray,
    start_left_pixel: PointLike,
    start_right_pixel: PointLike,
    end_left_pixel: PointLike,
    end_right_pixel: PointLike,
    units: str = "",
    preset_key: str = "generic",
    min_abs_disparity: float = 1.0,
    max_vertical_error_px: float | None = 3.0,
) -> StereoSegmentMeasurementResult:
    """Triangulate segment endpoints and return their 3D distance."""

    start = point_from_rectified_correspondence(
        q,
        tuple(float(value) for value in start_left_pixel),
        tuple(float(value) for value in start_right_pixel),
        min_abs_disparity=min_abs_disparity,
        max_vertical_error_px=max_vertical_error_px,
    )
    end = point_from_rectified_correspondence(
        q,
        tuple(float(value) for value in end_left_pixel),
        tuple(float(value) for value in end_right_pixel),
        min_abs_disparity=min_abs_disparity,
        max_vertical_error_px=max_vertical_error_px,
    )
    length_units = distance_between_samples(start, end)
    scale = unit_scale_to_cm(units)
    length_cm = None if scale is None else length_units * scale
    return StereoSegmentMeasurementResult(
        length_units=length_units,
        units=str(units or ""),
        start=start,
        end=end,
        length_cm=length_cm,
        length_m=None if length_cm is None else length_cm / 100.0,
        preset_key=preset_by_key(preset_key).key,
    )


def summarize_segment_measurements(
    results: Sequence[StereoSegmentMeasurementResult],
) -> StereoSegmentMeasurementSeries:
    """Return median and spread for repeated segment measurements."""

    if not results:
        raise ValueError("At least one segment measurement is required")

    units = results[0].units
    values = [float(result.length_units) for result in results]
    min_length = min(values)
    max_length = max(values)
    scale = unit_scale_to_cm(units)
    median_length = float(median(values))
    spread = max_length - min_length
    return StereoSegmentMeasurementSeries(
        count=len(results),
        units=units,
        median_length_units=median_length,
        min_length_units=min_length,
        max_length_units=max_length,
        spread_units=spread,
        median_length_cm=None if scale is None else median_length * scale,
        spread_cm=None if scale is None else spread * scale,
    )
