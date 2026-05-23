import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from stereo_calibration import (
    CheckerboardSpec,
    StereoObservations,
    calibrate_stereo_from_observations,
    checkerboard_object_points,
    load_manifest_collection,
    manifest_image_pairs,
    read_calibration_artifact,
    write_calibration_artifact,
)


def test_checkerboard_object_points_use_inner_corner_grid():
    board = CheckerboardSpec(columns=3, rows=2, square_size=2.5)

    points = checkerboard_object_points(board)

    assert points.shape == (6, 3)
    assert points[0].tolist() == [0.0, 0.0, 0.0]
    assert points[-1].tolist() == [5.0, 2.5, 0.0]


def test_manifest_image_pairs_resolves_relative_paths(tmp_path: Path):
    manifest = {
        "frames": [
            {"left_path": "left/pair_000001_left.png", "right_path": "right/pair_000001_right.png"}
        ]
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    pairs = manifest_image_pairs(manifest_path)

    assert pairs == [
        (
            tmp_path / "left" / "pair_000001_left.png",
            tmp_path / "right" / "pair_000001_right.png",
        )
    ]


def test_manifest_collection_combines_multiple_sessions(tmp_path: Path):
    pair = {"name": "Forward Stereo", "left": "Left", "right": "Right", "rig_id": "rig-a"}
    for idx in range(2):
        session = tmp_path / f"session-{idx + 1}"
        session.mkdir()
        (session / "manifest.json").write_text(
            json.dumps(
                {
                    "session_name": session.name,
                    "pair": pair,
                    "frames": [
                        {
                            "index": 1,
                            "stem": "pair_000001",
                            "left_path": "left/pair_000001_left.png",
                            "right_path": "right/pair_000001_right.png",
                            "pair_delta_ms": 10.0 + idx,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

    manifest, pairs = load_manifest_collection([tmp_path])

    assert manifest["source_count"] == 2
    assert manifest["pair"]["rig_id"] == "rig-a"
    assert len(manifest["frames"]) == 2
    assert manifest["frames"][1]["index"] == 2
    assert manifest["frames"][1]["source_session"] == "session-2"
    assert pairs[0][0] == tmp_path / "session-1" / "left" / "pair_000001_left.png"
    assert pairs[1][1] == tmp_path / "session-2" / "right" / "pair_000001_right.png"


def test_manifest_collection_rejects_mixed_stereo_pairs(tmp_path: Path):
    for idx, rig_id in enumerate(["rig-a", "rig-b"], start=1):
        session = tmp_path / f"session-{idx}"
        session.mkdir()
        (session / "manifest.json").write_text(
            json.dumps(
                {
                    "pair": {"name": "Forward Stereo", "left": "Left", "right": "Right", "rig_id": rig_id},
                    "frames": [],
                }
            ),
            encoding="utf-8",
        )

    with pytest.raises(ValueError, match="different stereo pair"):
        load_manifest_collection([tmp_path])


def test_stereo_calibration_recovers_fixed_intrinsic_baseline(tmp_path: Path):
    board = CheckerboardSpec(columns=6, rows=4, square_size=2.0)
    object_template = checkerboard_object_points(board)
    image_size = (640, 480)
    camera_matrix = np.array(
        [
            [820.0, 0.0, 320.0],
            [0.0, 820.0, 240.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    dist = np.zeros((5, 1), dtype=np.float64)
    rig_rotation = np.eye(3, dtype=np.float64)
    rig_translation = np.array([-8.0, 0.0, 0.0], dtype=np.float64)

    object_points = []
    left_points = []
    right_points = []
    for idx in range(8):
        rvec = np.array([0.02 * idx, -0.015 * idx, 0.01 * idx], dtype=np.float64)
        tvec = np.array([-4.0 + idx, -2.0 + 0.35 * idx, 70.0 + 4.0 * idx], dtype=np.float64)
        left, _ = cv2.projectPoints(object_template, rvec, tvec, camera_matrix, dist)

        board_rotation, _ = cv2.Rodrigues(rvec)
        right_rotation = rig_rotation @ board_rotation
        right_rvec, _ = cv2.Rodrigues(right_rotation)
        right_tvec = (rig_rotation @ tvec.reshape(3, 1) + rig_translation.reshape(3, 1)).reshape(3)
        right, _ = cv2.projectPoints(object_template, right_rvec, right_tvec, camera_matrix, dist)

        object_points.append(object_template.copy())
        left_points.append(left.reshape(-1, 2).astype(np.float32))
        right_points.append(right.reshape(-1, 2).astype(np.float32))

    observations = StereoObservations(
        object_points=object_points,
        left_image_points=left_points,
        right_image_points=right_points,
        image_size=image_size,
        rejected=[],
    )

    artifact = calibrate_stereo_from_observations(
        observations,
        rig_id="synthetic",
        pair_name="Synthetic Pair",
        board_spec=board,
        camera_matrix_left=camera_matrix,
        dist_coeffs_left=dist,
        camera_matrix_right=camera_matrix,
        dist_coeffs_right=dist,
    )

    assert artifact["stereo"]["baseline"] == pytest.approx(8.0, abs=1.0e-3)
    assert artifact["stereo"]["translation"][0] == pytest.approx(-8.0, abs=1.0e-3)
    assert artifact["rms"]["stereo"] < 1.0e-3
    assert artifact["quality"]["left_reprojection"]["rms_px"] < 1.0e-3
    assert artifact["quality"]["right_reprojection"]["rms_px"] < 1.0e-3
    assert artifact["quality"]["epipolar"]["rms_px"] < 1.0e-3
    assert artifact["quality"]["left_coverage"]["area_fraction"] > 0.0
    assert artifact["quality"]["warnings"]

    out_path = write_calibration_artifact(artifact, tmp_path / "calibration.json")
    loaded = read_calibration_artifact(out_path)
    assert loaded["rig_id"] == "synthetic"
    assert loaded["rectification"]["q"]
