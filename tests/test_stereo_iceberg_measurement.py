import numpy as np
import pytest

from triton_analysis.stereo.iceberg_measurement import (
    measure_stereo_keel_depth,
    summarize_keel_measurements,
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


def test_stereo_keel_measurement_triangulates_endpoint_distance_in_cm():
    result = measure_stereo_keel_depth(
        q=_q_matrix(),
        top_left_pixel=(4, 3),
        top_right_pixel=(-6, 3),
        bottom_left_pixel=(4, 23),
        bottom_right_pixel=(-6, 23),
        units="mm",
    )

    assert result.length_units == pytest.approx(20.0)
    assert result.length_cm == pytest.approx(2.0)
    assert result.length_m == pytest.approx(0.02)
    assert result.max_vertical_error_px == pytest.approx(0.0)
    assert result.min_abs_disparity_px == pytest.approx(10.0)


def test_stereo_keel_measurement_rejects_bad_rectified_match():
    with pytest.raises(ValueError, match="vertically"):
        measure_stereo_keel_depth(
            q=_q_matrix(),
            top_left_pixel=(4, 3),
            top_right_pixel=(-6, 8),
            bottom_left_pixel=(4, 23),
            bottom_right_pixel=(-6, 23),
            units="mm",
        )


def test_stereo_keel_series_reports_median_and_spread():
    results = [
        measure_stereo_keel_depth(
            q=_q_matrix(),
            top_left_pixel=(4, 3),
            top_right_pixel=(-6, 3),
            bottom_left_pixel=(4, y),
            bottom_right_pixel=(-6, y),
            units="mm",
        )
        for y in (22, 23, 25)
    ]

    summary = summarize_keel_measurements(results)

    assert summary.count == 3
    assert summary.median_length_units == pytest.approx(20.0)
    assert summary.spread_units == pytest.approx(3.0)
    assert summary.median_length_cm == pytest.approx(2.0)
    assert summary.spread_cm == pytest.approx(0.3)


def test_unit_scale_to_cm_handles_common_calibration_units():
    assert unit_scale_to_cm("mm") == pytest.approx(0.1)
    assert unit_scale_to_cm("cm") == pytest.approx(1.0)
    assert unit_scale_to_cm("m") == pytest.approx(100.0)
    assert unit_scale_to_cm("") is None
