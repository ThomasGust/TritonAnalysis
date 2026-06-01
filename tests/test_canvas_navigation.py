import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt6")

import numpy as np
from PyQt6.QtCore import QPoint, QRectF, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from triton_analysis.gui.canvas_navigation import clamp_pan_to_edge_margin


def test_zoomed_canvas_pan_allows_image_edges_to_move_inward():
    contents = QRectF(0.0, 0.0, 400.0, 300.0)
    target = QRectF(-200.0, -150.0, 800.0, 600.0)

    pan = [1_000_000.0, 1_000_000.0]
    clamp_pan_to_edge_margin(pan, contents, target)
    panned = QRectF(target.x() + pan[0], target.y() + pan[1], target.width(), target.height())
    assert panned.left() > contents.left() + 100.0
    assert panned.top() > contents.top() + 75.0

    pan = [-1_000_000.0, -1_000_000.0]
    clamp_pan_to_edge_margin(pan, contents, target)
    panned = QRectF(target.x() + pan[0], target.y() + pan[1], target.width(), target.height())
    assert panned.right() < contents.right() - 100.0
    assert panned.bottom() < contents.bottom() - 75.0


def test_left_drag_on_empty_zoomed_measurement_canvas_pans_instead_of_adding_point():
    app = QApplication.instance() or QApplication([])
    from triton_analysis.gui.iceberg_measurement_window import AFFINE_MODE, ClickSpec, MeasurementCanvas

    canvas = MeasurementCanvas()
    canvas.set_click_specs(AFFINE_MODE, [ClickSpec("a", "Point A")], clear=True)
    canvas.set_frame(np.zeros((240, 320, 3), dtype=np.uint8))
    canvas.resize(420, 320)
    canvas.show()
    app.processEvents()

    canvas._set_zoom(2.0, (210.0, 160.0))
    original_pan = tuple(canvas._pan)
    QTest.mousePress(canvas, Qt.MouseButton.LeftButton, pos=QPoint(210, 160))
    QTest.mouseMove(canvas, QPoint(250, 160))
    QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=QPoint(250, 160))

    assert canvas.point_count() == 0
    assert tuple(canvas._pan) != original_pan

    QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=QPoint(210, 160))
    assert canvas.point_count() == 1
