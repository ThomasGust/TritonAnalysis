"""Backward-compatible wrappers for stereo iceberg keel measurement."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from triton_analysis.stereo.segment_measurement import (
    StereoSegmentMeasurementResult,
    StereoSegmentMeasurementSeries,
    measure_stereo_segment,
    summarize_segment_measurements,
    unit_scale_to_cm,
)


PointLike = Sequence[float] | np.ndarray
StereoKeelMeasurementResult = StereoSegmentMeasurementResult
StereoKeelMeasurementSeries = StereoSegmentMeasurementSeries


def measure_stereo_keel_depth(
    *,
    q: np.ndarray,
    top_left_pixel: PointLike,
    top_right_pixel: PointLike,
    bottom_left_pixel: PointLike,
    bottom_right_pixel: PointLike,
    units: str = "",
    min_abs_disparity: float = 1.0,
    max_vertical_error_px: float | None = 3.0,
) -> StereoKeelMeasurementResult:
    """Triangulate top/bottom keel endpoints and return their 3D distance."""

    return measure_stereo_segment(
        q=q,
        start_left_pixel=top_left_pixel,
        start_right_pixel=top_right_pixel,
        end_left_pixel=bottom_left_pixel,
        end_right_pixel=bottom_right_pixel,
        units=units,
        preset_key="iceberg",
        min_abs_disparity=min_abs_disparity,
        max_vertical_error_px=max_vertical_error_px,
    )


def summarize_keel_measurements(
    results: Sequence[StereoKeelMeasurementResult],
) -> StereoKeelMeasurementSeries:
    """Return median and spread for repeated keel measurements."""

    return summarize_segment_measurements(results)
