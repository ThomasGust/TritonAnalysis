from pathlib import Path

import cv2
import numpy as np
import pytest

from crab_detector_cv import (
    apply_channel_gains,
    classify_crab_crop,
    detect_crabs_in_video,
    detect_crabs,
    estimate_board_white_balance_gains,
    unwrap_board,
)
from stereo_crab_analysis import (
    attach_stereo_depth_to_detections,
    detect_stereo_reference_copy_crabs,
    draw_stereo_depth_overlay,
    format_stereo_distance,
    stereo_depth_summary_text,
)
import stereo_crab_analysis


pytestmark = pytest.mark.vision
REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_ROOT = REPO_ROOT


def test_crab_detector_finds_expected_counts_on_bundled_sample():
    sample_path = ANALYSIS_ROOT / "data" / "crab_samples" / "crabby.jpg"
    image = cv2.imread(str(sample_path))
    assert image is not None

    result = detect_crabs(image)

    assert result is not None
    assert result["count"] == 8
    assert result["green_count"] == 4
    assert result["other_count"] == 4


def test_crab_detector_accepts_manual_board_polygon():
    sample_path = ANALYSIS_ROOT / "data" / "crab_samples" / "crabby.jpg"
    image = cv2.imread(str(sample_path))
    assert image is not None

    auto_result = detect_crabs(image)
    assert auto_result is not None

    unordered_polygon = np.roll(auto_result["board_polygon"], 2, axis=0)
    manual_result = detect_crabs(image, board_polygon=unordered_polygon)

    assert manual_result is not None
    assert manual_result["board_polygon_source"] == "manual"
    assert manual_result["count"] == 8
    assert manual_result["green_count"] == 4
    assert manual_result["other_count"] == 4


def test_stereo_depth_annotations_attach_to_crab_detections():
    detection_result = {
        "board_polygon": None,
        "detections": [
            {
                "index": 1,
                "original_box": np.array([8, 10, 8, 6], dtype=np.int32),
                "original_quad": np.array([[8, 10], [16, 10], [16, 16], [8, 16]], dtype=np.int32),
                "classification": {"label": "european_green", "is_european_green": True},
            }
        ],
        "count": 1,
        "green_count": 1,
        "other_count": 0,
        "species_counts": {"european_green": 1},
    }
    points_3d = np.zeros((24, 24, 3), dtype=np.float32)
    points_3d[:, :, 2] = 300.0
    disparity = np.full((24, 24), 12.0, dtype=np.float32)
    valid_depth = np.ones((24, 24), dtype=bool)

    summary = attach_stereo_depth_to_detections(
        detection_result,
        points_3d,
        disparity,
        valid_depth,
        units="mm",
        sample_radius=1,
    )

    stereo_depth = detection_result["detections"][0]["stereo_depth"]
    assert summary["available_count"] == 1
    assert summary["median_depth_units"] == pytest.approx(300.0)
    assert stereo_depth["available"] is True
    assert stereo_depth["depth_label"] == "30.0 cm"
    assert format_stereo_distance(1250.0, "mm") == "1.25 m"
    assert "1/1" in stereo_depth_summary_text(detection_result)

    overlay = draw_stereo_depth_overlay(np.zeros((24, 24, 3), dtype=np.uint8), detection_result, units="mm")
    assert overlay.shape == (24, 24, 3)
    assert np.count_nonzero(overlay) > 0


def test_stereo_reference_detector_keeps_epipolar_candidate_matches(monkeypatch):
    left = np.zeros((160, 220, 3), dtype=np.uint8)
    right = np.zeros((160, 220, 3), dtype=np.uint8)
    left[0, 0, 0] = 1
    right[0, 0, 0] = 2

    def fake_candidates(image):
        if int(image[0, 0, 0]) == 1:
            return [
                {"box": np.array([110, 45, 32, 28], dtype=np.int32), "area": 700},
                {"box": np.array([155, 92, 34, 30], dtype=np.int32), "area": 720},
            ]
        return [
            {"box": np.array([70, 46, 33, 28], dtype=np.int32), "area": 690},
            {"box": np.array([112, 91, 35, 30], dtype=np.int32), "area": 730},
        ]

    def fake_classify(_image, box):
        label = "european_green" if int(box[1]) < 80 else "jonah"
        return {
            "label": label,
            "is_european_green": label == "european_green",
            "copy_color_label": label,
            "copy_feature_scores": {},
        }

    monkeypatch.setattr(stereo_crab_analysis, "_relaxed_reference_copy_candidate_boxes", fake_candidates)
    monkeypatch.setattr(stereo_crab_analysis, "classify_reference_copy_candidate", fake_classify)

    result = detect_stereo_reference_copy_crabs(left, right)

    assert result is not None
    assert result["detector"] == "stereo_reference_copy"
    assert result["count"] == 2
    assert result["green_count"] == 1
    assert all("stereo_match" in detection for detection in result["detections"])


def test_crab_classifier_keeps_native_rock_under_red_attenuation():
    sample_path = ANALYSIS_ROOT / "data" / "crab_samples" / "crabby.jpg"
    image = cv2.imread(str(sample_path))
    assert image is not None

    baseline = detect_crabs(image)
    assert baseline is not None

    underwater = image.astype(np.float32)
    underwater[:, :, 2] *= 0.55
    underwater[:, :, 1] = underwater[:, :, 1] * 1.02 + 10.0
    underwater[:, :, 0] = underwater[:, :, 0] * 1.05 + 16.0
    underwater = np.clip(underwater, 0, 255).astype(np.uint8)

    unwrapped, _, _ = unwrap_board(
        underwater,
        polygon=baseline["board_polygon"],
        output_size=baseline["unwrapped_image"].shape[1::-1],
    )
    assert unwrapped is not None

    gains = estimate_board_white_balance_gains(unwrapped, baseline["unwrapped_mask"])
    corrected = apply_channel_gains(unwrapped, gains)
    green_count = 0

    for detection in baseline["detections"]:
        x, y, width, height = detection["unwrapped_box"]
        crop = corrected[y : y + height, x : x + width]
        classification = classify_crab_crop(crop)
        green_count += int(classification["is_european_green"])

    assert green_count == 4


@pytest.mark.groundtruth
@pytest.mark.slow
def test_reference_copy_detector_counts_underwater_aux_video_frame():
    video_path = (
        REPO_ROOT
        / "recordings"
        / "20260506-184600"
        / "Aux Camera.mp4"
    )
    if not video_path.exists():
        import pytest

        pytest.skip("underwater auxiliary camera recording is not available")

    capture = cv2.VideoCapture(str(video_path))
    capture.set(cv2.CAP_PROP_POS_MSEC, 5000)
    ok, frame = capture.read()
    capture.release()
    assert ok

    result = detect_crabs(frame)

    assert result is not None
    assert result["detector"] == "reference_copy"
    assert result["count"] == 8
    assert result["green_count"] == 4
    assert result["species_counts"]["jonah"] == 2
    assert result["species_counts"]["native_rock"] == 2


@pytest.mark.groundtruth
@pytest.mark.slow
def test_reference_copy_detector_ignores_tiny_artifact_candidate():
    video_path = (
        REPO_ROOT
        / "recordings"
        / "20260506-184445"
        / "Aux Camera.mp4"
    )
    if not video_path.exists():
        import pytest

        pytest.skip("underwater auxiliary camera recording is not available")

    capture = cv2.VideoCapture(str(video_path))
    capture.set(cv2.CAP_PROP_POS_MSEC, 3500)
    ok, frame = capture.read()
    capture.release()
    assert ok

    result = detect_crabs(frame)

    assert result is not None
    assert result["detector"] == "reference_copy"
    assert result["count"] == 8


@pytest.mark.groundtruth
@pytest.mark.slow
def test_video_detector_selects_underwater_frame_with_expected_counts():
    video_path = (
        REPO_ROOT
        / "recordings"
        / "20260506-184600"
        / "Aux Camera.mp4"
    )
    if not video_path.exists():
        import pytest

        pytest.skip("underwater auxiliary camera recording is not available")

    result = detect_crabs_in_video(
        video_path,
        start_seconds=4.5,
        end_seconds=5.1,
        sample_interval_seconds=0.5,
    )

    assert result is not None
    detection_result = result["detection_result"]
    assert detection_result["count"] == 8
    assert detection_result["green_count"] == 4
    assert result["temporal_vote"] is not None
    assert result["temporal_vote"]["signature"][:3] == (4, 2, 2)
    assert result["quality"]["confidence"] > 0.0


@pytest.mark.groundtruth
@pytest.mark.slow
def test_video_detector_rejects_low_evidence_gripper_scan():
    video_path = (
        REPO_ROOT
        / "recordings"
        / "20260507-204434"
        / "Aux Camera.mp4"
    )
    if not video_path.exists():
        import pytest

        pytest.skip("low-evidence gripper recording is not available")

    result = detect_crabs_in_video(
        video_path,
        start_seconds=0.0,
        end_seconds=5.0,
        sample_interval_seconds=0.5,
    )

    assert result is None


@pytest.mark.groundtruth
@pytest.mark.slow
def test_hard_pool_video_rejects_compression_artifact_frame():
    video_path = (
        REPO_ROOT
        / "recordings"
        / "20260507-154235"
        / "Aux Camera.mp4"
    )
    if not video_path.exists():
        import pytest

        pytest.skip("hard pool auxiliary camera recording is not available")

    capture = cv2.VideoCapture(str(video_path))
    capture.set(cv2.CAP_PROP_POS_MSEC, 2500)
    ok, frame = capture.read()
    capture.release()
    assert ok

    result = detect_crabs(frame)

    assert result is None or result["count"] <= 12


@pytest.mark.groundtruth
@pytest.mark.slow
def test_video_detector_selects_plausible_frame_in_hard_pool_video():
    video_path = (
        REPO_ROOT
        / "recordings"
        / "20260507-154235"
        / "Aux Camera.mp4"
    )
    if not video_path.exists():
        import pytest

        pytest.skip("hard pool auxiliary camera recording is not available")

    result = detect_crabs_in_video(
        video_path,
        start_seconds=0.0,
        end_seconds=5.0,
        sample_interval_seconds=0.5,
    )

    assert result is not None
    detection_result = result["detection_result"]
    assert detection_result["count"] == 8
    assert detection_result["green_count"] == 4
    assert detection_result["species_counts"]["jonah"] == 2
    assert detection_result["species_counts"]["native_rock"] == 2
    assert result["temporal_vote"] is not None
    assert result["temporal_vote"]["signature"][:3] == (4, 2, 2)


@pytest.mark.groundtruth
@pytest.mark.slow
def test_hard_pool_video_keeps_edge_touching_green_crab():
    video_path = (
        REPO_ROOT
        / "recordings"
        / "20260507-154235"
        / "Aux Camera.mp4"
    )
    if not video_path.exists():
        import pytest

        pytest.skip("hard pool auxiliary camera recording is not available")

    capture = cv2.VideoCapture(str(video_path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, 47)
    ok, frame = capture.read()
    capture.release()
    assert ok

    result = detect_crabs(frame)

    assert result is not None
    assert result["count"] == 8
    assert result["green_count"] == 4
    assert result["species_counts"]["jonah"] == 2
    assert result["species_counts"]["native_rock"] == 2
