import json
from pathlib import Path

import numpy as np
import pytest

from stereo_depth import (
    colorize_depth,
    colorize_disparity,
    distance_between_samples,
    load_depth_calibration,
    normalized_block_size,
    normalized_num_disparities,
    point_from_rectified_correspondence,
    rectification_maps_from_artifact,
    rectify_stereo_images,
    reproject_disparity,
    sample_depth_point,
)


def _artifact() -> dict:
    camera = [[50.0, 0.0, 4.0], [0.0, 50.0, 3.0], [0.0, 0.0, 1.0]]
    projection_left = [[50.0, 0.0, 4.0, 0.0], [0.0, 50.0, 3.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
    projection_right = [[50.0, 0.0, 4.0, -500.0], [0.0, 50.0, 3.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
    return {
        "schema": "tritonanalysis.stereo_calibration",
        "image_size": [8, 6],
        "board": {"units": "mm"},
        "left": {"camera_matrix": camera, "dist_coeffs": [0.0, 0.0, 0.0, 0.0, 0.0]},
        "right": {"camera_matrix": camera, "dist_coeffs": [0.0, 0.0, 0.0, 0.0, 0.0]},
        "rectification": {
            "r1": np.eye(3).tolist(),
            "r2": np.eye(3).tolist(),
            "p1": projection_left,
            "p2": projection_right,
            "q": [[1.0, 0.0, 0.0, -4.0], [0.0, 1.0, 0.0, -3.0], [0.0, 0.0, 0.0, 50.0], [0.0, 0.0, 0.1, 0.0]],
        },
    }


def test_load_depth_calibration_rejects_wrong_schema(tmp_path: Path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"schema": "nope"}), encoding="utf-8")

    with pytest.raises(ValueError, match="stereo calibration"):
        load_depth_calibration(path)


def test_rectification_maps_preserve_identity_pair():
    maps = rectification_maps_from_artifact(_artifact())
    left = np.zeros((6, 8, 3), dtype=np.uint8)
    left[:, :, 1] = np.arange(8, dtype=np.uint8)
    right = left.copy()

    left_rect, right_rect = rectify_stereo_images(left, right, maps)

    assert maps.units == "mm"
    assert left_rect.shape == left.shape
    assert right_rect.shape == right.shape
    assert np.mean(np.abs(left_rect.astype(int) - left.astype(int))) < 1.0


def test_disparity_reprojection_and_distance_sampling():
    maps = rectification_maps_from_artifact(_artifact())
    disparity = np.full((6, 8), 10.0, dtype=np.float32)
    valid = np.ones_like(disparity, dtype=bool)
    points, valid_3d = reproject_disparity(disparity, maps.q, valid)

    first = sample_depth_point(points, disparity, valid_3d, (4, 3), radius=1)
    second = sample_depth_point(points, disparity, valid_3d, (6, 3), radius=1)

    assert first.disparity == pytest.approx(10.0)
    assert first.point[2] == pytest.approx(50.0)
    assert distance_between_samples(first, second) == pytest.approx(2.0)


def test_reproject_disparity_rejects_near_zero_disparity_by_default():
    maps = rectification_maps_from_artifact(_artifact())
    disparity = np.array([[0.5, 2.0]], dtype=np.float32)
    valid = np.ones_like(disparity, dtype=bool)

    _points, valid_3d = reproject_disparity(disparity, maps.q, valid)
    _points_unfiltered, valid_unfiltered = reproject_disparity(
        disparity,
        maps.q,
        valid,
        min_abs_disparity=0.0,
    )

    assert valid_3d.tolist() == [[False, True]]
    assert valid_unfiltered.tolist() == [[True, True]]


def test_manual_rectified_correspondence_triangulates_without_dense_disparity():
    maps = rectification_maps_from_artifact(_artifact())

    first = point_from_rectified_correspondence(maps.q, (4, 3), (-6, 3))
    second = point_from_rectified_correspondence(maps.q, (6, 3), (-4, 3))

    assert first.disparity == pytest.approx(10.0)
    assert first.point[2] == pytest.approx(50.0)
    assert distance_between_samples(first, second) == pytest.approx(2.0)
    with pytest.raises(ValueError, match="vertically"):
        point_from_rectified_correspondence(maps.q, (4, 3), (-6, 8))


def test_disparity_parameter_normalization_and_colorizing():
    assert normalized_num_disparities(17) == 32
    assert normalized_num_disparities(0) == 16
    assert normalized_block_size(4) == 5
    disparity = np.array([[np.nan, 1.0], [2.0, 3.0]], dtype=np.float32)
    valid = np.isfinite(disparity)
    points = np.dstack([np.zeros_like(disparity), np.zeros_like(disparity), np.nan_to_num(disparity, nan=0.0)])

    assert colorize_disparity(disparity, valid).shape == (2, 2, 3)
    assert colorize_depth(points, valid).shape == (2, 2, 3)
