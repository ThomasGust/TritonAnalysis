import numpy as np
import pytest

from triton_analysis.stereo.segment_measurement import (
    measure_stereo_segment,
    preset_by_key,
    right_endpoint_order_mismatch,
    summarize_segment_measurements,
    unit_scale_to_cm,
)


def _q_matrix() -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0, 0.0, -4.0],
            [0.0, 1.0, 0.0, -3.0],
            [0.0, 0.0, 0.0, 50.0],
            [0.0, 0.0, 0.1, 0.0],
        ],
        dtype=np.float64,
    )


def test_stereo_segment_measurement_triangulates_generic_distance():
    result = measure_stereo_segment(
        q=_q_matrix(),
        start_left_pixel=(4, 3),
        start_right_pixel=(-6, 3),
        end_left_pixel=(14, 3),
        end_right_pixel=(4, 3),
        units="mm",
        preset_key="generic",
    )

    assert result.preset_key == "generic"
    assert result.length_units == pytest.approx(10.0)
    assert result.length_cm == pytest.approx(1.0)
    assert result.length_m == pytest.approx(0.01)
    assert result.max_vertical_error_px == pytest.approx(0.0)
    assert result.min_abs_disparity_px == pytest.approx(10.0)


def test_stereo_segment_presets_include_iceberg_and_coral_labels():
    assert preset_by_key("iceberg").start_label == "Keel top"
    assert preset_by_key("iceberg").end_label == "Keel bottom"
    assert preset_by_key("coral").result_label == "Coral rig length"
    assert preset_by_key("missing").key == "generic"


def test_stereo_segment_measurement_rejects_bad_rectified_match():
    with pytest.raises(ValueError, match="vertically"):
        measure_stereo_segment(
            q=_q_matrix(),
            start_left_pixel=(4, 3),
            start_right_pixel=(-6, 8),
            end_left_pixel=(14, 3),
            end_right_pixel=(4, 3),
            units="mm",
        )


def test_right_endpoint_order_mismatch_detects_reversed_horizontal_clicks():
    assert right_endpoint_order_mismatch(
        left_start_pixel=(100, 50),
        left_end_pixel=(260, 52),
        right_start_pixel=(230, 51),
        right_end_pixel=(80, 49),
    )
    assert not right_endpoint_order_mismatch(
        left_start_pixel=(100, 50),
        left_end_pixel=(260, 52),
        right_start_pixel=(80, 49),
        right_end_pixel=(230, 51),
    )


def test_right_endpoint_order_mismatch_detects_reversed_vertical_clicks():
    assert right_endpoint_order_mismatch(
        left_start_pixel=(120, 80),
        left_end_pixel=(118, 240),
        right_start_pixel=(110, 230),
        right_end_pixel=(112, 78),
    )
    assert not right_endpoint_order_mismatch(
        left_start_pixel=(120, 80),
        left_end_pixel=(118, 240),
        right_start_pixel=(112, 78),
        right_end_pixel=(110, 230),
    )


def test_stereo_segment_series_reports_median_and_spread():
    results = [
        measure_stereo_segment(
            q=_q_matrix(),
            start_left_pixel=(4, 3),
            start_right_pixel=(-6, 3),
            end_left_pixel=(x, 3),
            end_right_pixel=(x - 10, 3),
            units="mm",
            preset_key="coral",
        )
        for x in (13, 14, 16)
    ]

    summary = summarize_segment_measurements(results)

    assert summary.count == 3
    assert summary.median_length_units == pytest.approx(10.0)
    assert summary.spread_units == pytest.approx(3.0)
    assert summary.median_length_cm == pytest.approx(1.0)
    assert summary.spread_cm == pytest.approx(0.3)


def test_unit_scale_to_cm_handles_common_calibration_units():
    assert unit_scale_to_cm("mm") == pytest.approx(0.1)
    assert unit_scale_to_cm("cm") == pytest.approx(1.0)
    assert unit_scale_to_cm("m") == pytest.approx(100.0)
    assert unit_scale_to_cm("") is None
