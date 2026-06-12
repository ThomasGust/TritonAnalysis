import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from triton_analysis.stereo.calibration import (
    DEFAULT_CHARUCO_DICTIONARY,
    DEFAULT_CHARUCO_MARKER_SIZE,
    DEFAULT_CHARUCO_SQUARES_X,
    DEFAULT_CHARUCO_SQUARES_Y,
    DEFAULT_CHARUCO_SQUARE_SIZE,
    CharucoBoardSpec,
    CheckerboardSpec,
    StereoObservations,
    annotate_board_detection,
    calibrate_stereo_from_observations,
    checkerboard_object_points,
    collect_checkerboard_observations,
    collect_charuco_observations,
    detect_stereo_board,
    first_percent_stereo_pairs,
    load_manifest_collection,
    manifest_image_pairs,
    read_calibration_artifact,
    write_calibration_artifact,
)
from triton_analysis.stereo.depth import rectification_maps_from_artifact


def _invalid_rectification_fraction(map_x: np.ndarray, map_y: np.ndarray, image_size: tuple[int, int]) -> float:
    width, height = image_size
    valid = (map_x >= 0) & (map_x < width) & (map_y >= 0) & (map_y < height)
    return 1.0 - float(np.count_nonzero(valid)) / float(valid.size)


def test_checkerboard_object_points_use_inner_corner_grid():
    board = CheckerboardSpec(columns=3, rows=2, square_size=2.5)

    points = checkerboard_object_points(board)

    assert points.shape == (6, 3)
    assert points[0].tolist() == [0.0, 0.0, 0.0]
    assert points[-1].tolist() == [5.0, 2.5, 0.0]


def test_charuco_defaults_use_calibio_dictionary():
    board = CharucoBoardSpec(
        squares_x=DEFAULT_CHARUCO_SQUARES_X,
        squares_y=DEFAULT_CHARUCO_SQUARES_Y,
        square_size=DEFAULT_CHARUCO_SQUARE_SIZE,
        marker_size=DEFAULT_CHARUCO_MARKER_SIZE,
    )

    assert board.dictionary == DEFAULT_CHARUCO_DICTIONARY == "DICT_5X5_1000"
    assert (board.squares_x, board.squares_y) == (12, 9)
    assert (board.square_size, board.marker_size, board.units) == (6.0, 4.5, "cm")


def test_blank_stereo_detection_returns_preview_summary():
    board = CharucoBoardSpec(
        squares_x=DEFAULT_CHARUCO_SQUARES_X,
        squares_y=DEFAULT_CHARUCO_SQUARES_Y,
        square_size=DEFAULT_CHARUCO_SQUARE_SIZE,
        marker_size=DEFAULT_CHARUCO_MARKER_SIZE,
    )
    left = np.zeros((80, 120, 3), dtype=np.uint8)
    right = np.zeros((80, 120, 3), dtype=np.uint8)

    detection = detect_stereo_board(left, right, board, min_corners=8)
    annotated = annotate_board_detection(left, detection["left"], matched_ids=detection["matched_ids"])

    assert detection["kind"] == "charuco"
    assert detection["matched_count"] == 0
    assert detection["accepted"] is False
    assert annotated.shape == left.shape


def test_charuco_marker_only_failure_points_to_board_dimensions(tmp_path: Path):
    actual_board = cv2.aruco.CharucoBoard(
        (DEFAULT_CHARUCO_SQUARES_X, DEFAULT_CHARUCO_SQUARES_Y),
        DEFAULT_CHARUCO_SQUARE_SIZE,
        DEFAULT_CHARUCO_MARKER_SIZE,
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000),
    )
    board_image = actual_board.generateImage((1200, 900))
    left_path = tmp_path / "left.png"
    right_path = tmp_path / "right.png"
    assert cv2.imwrite(str(left_path), board_image)
    assert cv2.imwrite(str(right_path), board_image)
    wrong_board = CharucoBoardSpec(squares_x=24, squares_y=17, square_size=30.0, marker_size=22.0)

    with pytest.raises(ValueError, match="markers found but no ChArUco corners"):
        collect_charuco_observations([(left_path, right_path)], wrong_board, min_pairs=1)


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


def test_first_percent_stereo_pairs_preserves_manifest_order():
    pairs = [(f"left-{index}", f"right-{index}") for index in range(10)]

    assert first_percent_stereo_pairs(pairs, 100) == pairs
    assert first_percent_stereo_pairs(pairs, 25) == pairs[:3]
    assert first_percent_stereo_pairs(pairs, 1) == pairs[:1]
    assert first_percent_stereo_pairs(pairs, 0) == pairs[:1]


def test_checkerboard_collection_reports_detection_progress(tmp_path: Path):
    board = CheckerboardSpec(columns=4, rows=3, square_size=2.5)
    blank = np.zeros((80, 120, 3), dtype=np.uint8)
    image_pairs = []
    for index in range(2):
        left_path = tmp_path / f"left_{index}.png"
        right_path = tmp_path / f"right_{index}.png"
        assert cv2.imwrite(str(left_path), blank)
        assert cv2.imwrite(str(right_path), blank)
        image_pairs.append((left_path, right_path))
    events: list[dict] = []

    with pytest.raises(ValueError, match="Only 0 valid stereo pairs"):
        collect_checkerboard_observations(
            image_pairs,
            board,
            min_pairs=1,
            progress_callback=events.append,
        )

    assert events[0] == {
        "event": "detect_start",
        "detector": "checkerboard",
        "total": 2,
        "accepted": 0,
        "rejected": 0,
    }
    pair_events = [event for event in events if event["event"] == "detect_pair"]
    assert [event["index"] for event in pair_events] == [1, 2]
    assert pair_events[-1]["accepted"] == 0
    assert pair_events[-1]["rejected"] == 2
    assert events[-1] == {
        "event": "detect_complete",
        "detector": "checkerboard",
        "total": 2,
        "accepted": 0,
        "rejected": 2,
    }


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
    progress_events: list[dict] = []

    artifact = calibrate_stereo_from_observations(
        observations,
        rig_id="synthetic",
        pair_name="Synthetic Pair",
        board_spec=board,
        camera_matrix_left=camera_matrix,
        dist_coeffs_left=dist,
        camera_matrix_right=camera_matrix,
        dist_coeffs_right=dist,
        progress_callback=progress_events.append,
    )

    assert artifact["stereo"]["baseline"] == pytest.approx(8.0, abs=1.0e-3)
    assert artifact["stereo"]["translation"][0] == pytest.approx(-8.0, abs=1.0e-3)
    assert artifact["rms"]["stereo"] < 1.0e-3
    assert artifact["quality"]["left_reprojection"]["rms_px"] < 1.0e-3
    assert artifact["quality"]["right_reprojection"]["rms_px"] < 1.0e-3
    assert artifact["quality"]["epipolar"]["rms_px"] < 1.0e-3
    assert artifact["quality"]["epipolar"]["space"] == "rectified_pixels"
    assert artifact["quality"]["raw_distorted_epipolar"]["space"] == "distorted_pixels"
    assert artifact["quality"]["left_coverage"]["area_fraction"] > 0.0
    assert artifact["quality"]["warnings"]
    assert artifact["rectification"]["alpha"] == pytest.approx(0.0)
    assert [event["event"] for event in progress_events][0] == "solve_start"
    assert any(
        event["stage"] == "stereo_extrinsics" and event["busy"] is True
        for event in progress_events
    )
    assert progress_events[-1]["event"] == "solve_complete"

    out_path = write_calibration_artifact(artifact, tmp_path / "calibration.json")
    loaded = read_calibration_artifact(out_path)
    assert loaded["rig_id"] == "synthetic"
    assert loaded["rectification"]["q"]


def test_rectified_epipolar_quality_undistorts_points_before_scoring():
    board = CheckerboardSpec(columns=8, rows=6, square_size=35.0)
    object_template = checkerboard_object_points(board)
    image_size = (1920, 1080)
    camera_matrix = np.array(
        [
            [1150.0, 0.0, 960.0],
            [0.0, 1160.0, 540.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    dist = np.array([-0.35, 0.16, 0.0015, -0.002, -0.03], dtype=np.float64).reshape(-1, 1)
    rig_rotation = np.eye(3, dtype=np.float64)
    rig_translation = np.array([-200.0, 2.0, 6.0], dtype=np.float64)

    object_points = []
    left_points = []
    right_points = []
    for idx in range(12):
        rvec = np.array(
            [0.02 * ((idx % 4) - 1.5), -0.08 + 0.025 * idx, 0.04 * ((idx % 3) - 1)],
            dtype=np.float64,
        )
        tvec = np.array(
            [-260.0 + 55.0 * (idx % 4), -140.0 + 70.0 * (idx // 4), 900.0 + 45.0 * idx],
            dtype=np.float64,
        )
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
        rig_id="distorted",
        pair_name="Distorted Pair",
        board_spec=board,
        camera_matrix_left=camera_matrix,
        dist_coeffs_left=dist,
        camera_matrix_right=camera_matrix,
        dist_coeffs_right=dist,
    )

    assert artifact["rms"]["stereo"] < 1.0e-3
    assert artifact["quality"]["epipolar"]["rms_px"] < 1.0e-3
    assert artifact["quality"]["raw_distorted_epipolar"]["rms_px"] > 1.0
    assert artifact["rectification"]["alpha"] == pytest.approx(0.0)
    maps = rectification_maps_from_artifact(artifact)
    assert _invalid_rectification_fraction(maps.left_x, maps.left_y, image_size) < 0.01
    assert _invalid_rectification_fraction(maps.right_x, maps.right_y, image_size) < 0.01
