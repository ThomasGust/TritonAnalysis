import os
from pathlib import Path

import cv2
import pytest

from triton_analysis.crab.detector import (
    default_reference_image_path,
    detect_european_green_crabs,
    detection_summary_text,
    draw_european_green_crab_detections,
)


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytestmark = pytest.mark.vision


def _reference_image() -> Path:
    path = default_reference_image_path()
    if path is None:
        pytest.skip("default crab reference image is not available on this machine")
    return path


def test_crab_detector_counts_reference_board_green_crabs():
    image_path = _reference_image()
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    result = detect_european_green_crabs(image)

    assert result is not None
    assert result.count == 4
    assert result.inlier_count > 50
    assert "European green crabs: 4" in detection_summary_text(result)
    assert all(detection.bbox[2] > 0 and detection.bbox[3] > 0 for detection in result.detections)


def test_crab_detector_handles_rotated_scaled_snapshot():
    image_path = _reference_image().parent / "20260530-081228_Arm_Camera_snapshot.png"
    if not image_path.exists():
        pytest.skip("rotated crab snapshot is not available on this machine")
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    result = detect_european_green_crabs(image)

    assert result is not None
    assert result.count == 4
    assert result.inlier_count >= 20


def test_crab_detector_draws_annotated_output():
    image_path = _reference_image()
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    result = detect_european_green_crabs(image)

    annotated = draw_european_green_crab_detections(image, result)

    assert annotated.shape == image.shape
    assert annotated.dtype == image.dtype
    assert (annotated != image).any()


def test_crab_detection_window_runs_on_loaded_image():
    pytest.importorskip("PyQt6")
    from PyQt6.QtWidgets import QApplication
    from triton_analysis.gui.crab_detection_window import CrabDetectionWindow

    app = QApplication.instance() or QApplication([])
    image_path = _reference_image()
    window = CrabDetectionWindow(image_paths=[image_path], detector_mode="board")
    try:
        window.show()
        app.processEvents()
        assert window._current_result is not None
        assert window._current_result.count == 4
        assert "European green crabs: 4" in window.status_label.text()
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()
