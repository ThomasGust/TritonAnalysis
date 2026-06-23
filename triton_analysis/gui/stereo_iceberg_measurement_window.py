"""Stereo applet for measuring two-point segments from calibrated stereo."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QCursor, QPainter, QPen
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from triton_analysis.gui.file_dialogs import ThumbnailFileDialog as QFileDialog

from triton_analysis.workspace import latest_pilot_stereo_sessions_dir, workspace_paths
from triton_analysis.gui.canvas_navigation import clamp_pan_to_edge_margin, moved_past_pan_threshold
from triton_analysis.gui.image_preview import frame_to_pixmap
from triton_analysis.gui.responsive import resize_to_available_screen, vertical_scroll_area
from triton_analysis.stereo.calibration import load_manifest_collection
from triton_analysis.stereo.depth import (
    CorrespondenceSample,
    load_depth_calibration,
    rectification_maps_from_artifact,
    rectify_stereo_images,
)
from triton_analysis.stereo.segment_measurement import (
    STEREO_SEGMENT_PRESETS,
    StereoSegmentReferenceCheck,
    StereoSegmentMeasurementResult,
    evaluate_reference_scale_check,
    measure_stereo_segment,
    preset_by_key,
    right_endpoint_order_mismatch,
    summarize_segment_measurements,
)


class _SectionCard(QFrame):
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setObjectName("stereoCard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        title_label = QLabel(title)
        title_label.setStyleSheet("font-weight: 700; font-size: 14px;")
        layout.addWidget(title_label)
        self.body = QVBoxLayout()
        self.body.setContentsMargins(0, 0, 0, 0)
        self.body.setSpacing(8)
        layout.addLayout(self.body)


class _StereoKeelCanvas(QWidget):
    """Zoomable rectified-image canvas with two endpoint clicks."""

    pointsChanged = pyqtSignal()

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.title = str(title)
        self._pixmap = frame_to_pixmap(None)
        self._image_width = 0
        self._image_height = 0
        self._points: list[tuple[int, int]] = []
        self._reference_points: list[tuple[int, int]] = []
        self._active_line = "target"
        self._point_labels = ["Start", "End"]
        self._reference_point_labels = ["Ref start", "Ref end"]
        self._badge = ""
        self._reference_badge = ""
        self._placeholder = "No rectified image"
        self._zoom = 1.0
        self._pan = np.array([0.0, 0.0], dtype=np.float64)
        self._drag_index: int | None = None
        self._panning = False
        self._last_pan_pos: tuple[float, float] | None = None
        self._pending_point_press: tuple[float, float] | None = None
        self.setMinimumSize(240, 220)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def _line_key(self, line: str | None = None) -> str:
        key = str(line or self._active_line or "target").strip().lower()
        return "reference" if key == "reference" else "target"

    def _line_points(self, line: str | None = None) -> list[tuple[int, int]]:
        return self._reference_points if self._line_key(line) == "reference" else self._points

    def set_active_line(self, line: str) -> None:
        self._active_line = self._line_key(line)
        self._drag_index = None
        self._pending_point_press = None
        self._refresh_cursor()
        self.update()

    def points(self, line: str | None = None) -> list[tuple[int, int]]:
        return list(self._line_points(line))

    def set_points(self, points: list[tuple[int, int]], *, emit: bool = True, line: str | None = None) -> None:
        line_points = self._line_points(line)
        line_points[:] = [
            (
                int(round(np.clip(float(point[0]), 0, max(0, self._image_width - 1)))),
                int(round(np.clip(float(point[1]), 0, max(0, self._image_height - 1)))),
            )
            for point in points[:2]
        ]
        if self._line_key(line) == "reference":
            self._reference_badge = ""
        else:
            self._badge = ""
        self._drag_index = None
        self._refresh_cursor()
        self.update()
        if emit:
            self.pointsChanged.emit()

    def swap_points(self, *, emit: bool = True, line: str | None = None) -> None:
        line_points = self._line_points(line)
        if len(line_points) != 2:
            return
        self.set_points([line_points[1], line_points[0]], emit=emit, line=line)

    def set_point_labels(self, start_label: str, end_label: str) -> None:
        self._point_labels = [str(start_label or "Start"), str(end_label or "End")]
        self.update()

    def set_reference_point_labels(self, start_label: str, end_label: str) -> None:
        self._reference_point_labels = [str(start_label or "Ref start"), str(end_label or "Ref end")]
        self.update()

    def set_frame(self, frame_bgr: np.ndarray | None, *, placeholder: str = "No rectified image") -> None:
        self._placeholder = placeholder
        self._pixmap = frame_to_pixmap(frame_bgr)
        if frame_bgr is None:
            self._image_width = 0
            self._image_height = 0
        else:
            self._image_height, self._image_width = frame_bgr.shape[:2]
        self._zoom = 1.0
        self._pan[:] = 0.0
        self._pending_point_press = None
        self._points.clear()
        self._reference_points.clear()
        self._badge = ""
        self._reference_badge = ""
        self.pointsChanged.emit()
        self._refresh_cursor()
        self.update()

    def clear_points(self, line: str | None = None) -> None:
        key = self._line_key(line)
        line_points = self._line_points(key)
        badge = self._reference_badge if key == "reference" else self._badge
        if not line_points and not badge:
            return
        line_points.clear()
        if key == "reference":
            self._reference_badge = ""
        else:
            self._badge = ""
        self._drag_index = None
        self._pending_point_press = None
        self.pointsChanged.emit()
        self._refresh_cursor()
        self.update()

    def clear_all_points(self) -> None:
        changed = bool(self._points or self._reference_points or self._badge or self._reference_badge)
        self._points.clear()
        self._reference_points.clear()
        self._badge = ""
        self._reference_badge = ""
        self._drag_index = None
        self._pending_point_press = None
        self._refresh_cursor()
        self.update()
        if changed:
            self.pointsChanged.emit()

    def set_badge(self, badge: str, line: str | None = None) -> None:
        if self._line_key(line) == "reference":
            self._reference_badge = str(badge or "")
        else:
            self._badge = str(badge or "")
        self.update()

    def _centered_target_rect(self) -> QRectF:
        contents = self.contentsRect()
        if self._pixmap.isNull() or self._image_width <= 0 or self._image_height <= 0:
            return QRectF(contents)
        scale = min(
            contents.width() / float(self._image_width),
            contents.height() / float(self._image_height),
        ) * self._zoom
        draw_width = self._image_width * scale
        draw_height = self._image_height * scale
        return QRectF(
            contents.x() + (contents.width() - draw_width) / 2.0,
            contents.y() + (contents.height() - draw_height) / 2.0,
            draw_width,
            draw_height,
        )

    def _target_rect(self) -> QRectF:
        target = self._centered_target_rect()
        if self._pixmap.isNull() or self._image_width <= 0 or self._image_height <= 0:
            return target
        return QRectF(
            target.x() + float(self._pan[0]),
            target.y() + float(self._pan[1]),
            target.width(),
            target.height(),
        )

    def _clamp_pan(self) -> None:
        if self._pixmap.isNull() or self._image_width <= 0 or self._image_height <= 0:
            self._pan[:] = 0.0
            return
        contents = self.contentsRect()
        target = self._centered_target_rect()
        clamp_pan_to_edge_margin(self._pan, contents, target)

    def _image_to_widget(self, point: tuple[int, int]) -> tuple[float, float] | None:
        target = self._target_rect()
        if target.isNull() or self._image_width <= 0 or self._image_height <= 0:
            return None
        return (
            target.x() + point[0] * target.width() / max(1.0, float(self._image_width - 1)),
            target.y() + point[1] * target.height() / max(1.0, float(self._image_height - 1)),
        )

    def _widget_to_image(self, x: float, y: float) -> tuple[int, int] | None:
        target = self._target_rect()
        if target.isNull() or not target.contains(x, y):
            return None
        image_x = (x - target.x()) * max(1.0, float(self._image_width - 1)) / max(1.0, target.width())
        image_y = (y - target.y()) * max(1.0, float(self._image_height - 1)) / max(1.0, target.height())
        return (
            int(round(np.clip(image_x, 0, max(0, self._image_width - 1)))),
            int(round(np.clip(image_y, 0, max(0, self._image_height - 1)))),
        )

    def _nearest_point_index(self, x: float, y: float, *, max_distance: float = 12.0) -> int | None:
        nearest = None
        nearest_distance = float(max_distance)
        for index, point in enumerate(self._line_points()):
            widget_point = self._image_to_widget(point)
            if widget_point is None:
                continue
            distance = float(np.hypot(widget_point[0] - x, widget_point[1] - y))
            if distance <= nearest_distance:
                nearest = index
                nearest_distance = distance
        return nearest

    def _refresh_cursor(self) -> None:
        if self._panning:
            self.setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))
        elif self._drag_index is not None:
            self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        elif not self._pixmap.isNull() and len(self._line_points()) < 2:
            self.setCursor(QCursor(Qt.CursorShape.CrossCursor))
        else:
            self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))

    def _start_pan(self, x: float, y: float) -> bool:
        if self._pixmap.isNull() or self._zoom <= 1.0:
            return False
        self._pending_point_press = None
        self._panning = True
        self._last_pan_pos = (x, y)
        self._refresh_cursor()
        return True

    def mousePressEvent(self, event) -> None:
        x = float(event.position().x())
        y = float(event.position().y())
        if event.button() == Qt.MouseButton.MiddleButton:
            self._start_pan(x, y)
            return
        if event.button() == Qt.MouseButton.RightButton:
            line_points = self._line_points()
            nearest = self._nearest_point_index(x, y)
            if nearest is not None:
                del line_points[nearest]
            elif line_points:
                line_points.pop()
            self.set_badge("", line=self._active_line)
            self.pointsChanged.emit()
            self._refresh_cursor()
            self.update()
            return
        if event.button() != Qt.MouseButton.LeftButton or self._pixmap.isNull():
            return
        nearest = self._nearest_point_index(x, y)
        if nearest is not None:
            self._drag_index = nearest
            self._refresh_cursor()
            return
        if len(self._line_points()) >= 2:
            self._start_pan(x, y)
            return
        point = self._widget_to_image(x, y)
        if point is None:
            return
        self._pending_point_press = (x, y)

    def mouseMoveEvent(self, event) -> None:
        x = float(event.position().x())
        y = float(event.position().y())
        if self._panning and self._last_pan_pos is not None:
            last_x, last_y = self._last_pan_pos
            self._pan += np.array([x - last_x, y - last_y], dtype=np.float64)
            self._last_pan_pos = (x, y)
            self._clamp_pan()
            self.update()
            return
        if self._pending_point_press is not None:
            if self._zoom > 1.0 and moved_past_pan_threshold(self._pending_point_press, x, y):
                start_x, start_y = self._pending_point_press
                self._start_pan(start_x, start_y)
                self._pan += np.array([x - start_x, y - start_y], dtype=np.float64)
                self._last_pan_pos = (x, y)
                self._clamp_pan()
                self.update()
            return
        if self._drag_index is not None:
            point = self._widget_to_image(x, y)
            if point is not None:
                self._line_points()[self._drag_index] = point
                self.set_badge("", line=self._active_line)
                self.pointsChanged.emit()
                self.update()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() in (Qt.MouseButton.MiddleButton, Qt.MouseButton.LeftButton) and self._panning:
            self._panning = False
            self._last_pan_pos = None
            self._refresh_cursor()
            return
        if event.button() == Qt.MouseButton.LeftButton and self._pending_point_press is not None:
            self._pending_point_press = None
            line_points = self._line_points()
            if len(line_points) >= 2:
                return
            point = self._widget_to_image(float(event.position().x()), float(event.position().y()))
            if point is None:
                return
            line_points.append(point)
            self.set_badge("", line=self._active_line)
            self.pointsChanged.emit()
            self._refresh_cursor()
            self.update()
            return
        if event.button() == Qt.MouseButton.LeftButton and self._drag_index is not None:
            self._drag_index = None
            self._refresh_cursor()

    def wheelEvent(self, event) -> None:
        if self._pixmap.isNull():
            return
        steps = float(event.angleDelta().y()) / 120.0
        if abs(steps) < 1.0e-6:
            return
        anchor = self._widget_to_image(float(event.position().x()), float(event.position().y()))
        old_zoom = self._zoom
        self._zoom = float(np.clip(self._zoom * (1.2 ** steps), 1.0, 8.0))
        if abs(self._zoom - old_zoom) < 1.0e-6:
            return
        if anchor is not None:
            centered = self._centered_target_rect()
            self._pan[0] = (
                float(event.position().x())
                - anchor[0] * centered.width() / max(1.0, float(self._image_width - 1))
                - centered.x()
            )
            self._pan[1] = (
                float(event.position().y())
                - anchor[1] * centered.height() / max(1.0, float(self._image_height - 1))
                - centered.y()
            )
        self._clamp_pan()
        self._refresh_cursor()
        self.update()
        event.accept()

    def resizeEvent(self, event) -> None:
        self._clamp_pan()
        super().resizeEvent(event)

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#101116"))
        target = self._target_rect()
        if self._pixmap.isNull():
            painter.setPen(QColor("#aab0c0"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._placeholder)
            return
        painter.drawPixmap(target.toRect(), self._pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self._draw_points(painter)

    def _draw_points(self, painter: QPainter) -> None:
        self._draw_measurement_line(
            painter,
            points=self._points,
            labels=self._point_labels,
            point_colors=(QColor("#55d6ff"), QColor("#ffe66a")),
            line_color=QColor("#ffe66a"),
            badge=self._badge,
            active=self._active_line == "target",
            badge_offset=-32,
        )
        self._draw_measurement_line(
            painter,
            points=self._reference_points,
            labels=self._reference_point_labels,
            point_colors=(QColor("#84f29a"), QColor("#ff79c6")),
            line_color=QColor("#84f29a"),
            badge=self._reference_badge,
            active=self._active_line == "reference",
            badge_offset=10,
        )

    def _draw_measurement_line(
        self,
        painter: QPainter,
        *,
        points: list[tuple[int, int]],
        labels: list[str],
        point_colors: tuple[QColor, QColor],
        line_color: QColor,
        badge: str,
        active: bool,
        badge_offset: int,
    ) -> None:
        widget_points = [self._image_to_widget(point) for point in points]
        widget_points = [point for point in widget_points if point is not None]
        target = self._target_rect()
        for index, point in enumerate(widget_points):
            color = point_colors[min(index, len(point_colors) - 1)]
            x, y = point
            painter.setPen(QPen(QColor(color.red(), color.green(), color.blue(), 145 if active else 85), 1))
            painter.drawLine(int(target.left()), int(round(y)), int(target.right()), int(round(y)))
            painter.setPen(QPen(QColor("#111218"), 5))
            painter.drawEllipse(int(round(x)) - 8, int(round(y)) - 8, 16, 16)
            painter.setPen(QPen(color, 4 if active else 2))
            painter.drawEllipse(int(round(x)) - 8, int(round(y)) - 8, 16, 16)
            painter.setPen(QColor("#f7f8ff"))
            painter.drawText(int(round(x)) + 12, int(round(y)) - 10, labels[index])
        if len(widget_points) == 2:
            painter.setPen(QPen(line_color, 4 if active else 2))
            painter.drawLine(
                int(round(widget_points[0][0])),
                int(round(widget_points[0][1])),
                int(round(widget_points[1][0])),
                int(round(widget_points[1][1])),
            )
        if badge and len(widget_points) == 2:
            mid_x = int(round((widget_points[0][0] + widget_points[1][0]) * 0.5))
            mid_y = int(round((widget_points[0][1] + widget_points[1][1]) * 0.5))
            metrics = painter.fontMetrics()
            width = max(170, metrics.horizontalAdvance(badge) + 18)
            badge_rect = QRectF(mid_x - width / 2.0, mid_y + badge_offset, width, 24)
            painter.fillRect(badge_rect, QColor(20, 22, 28, 220))
            painter.setPen(line_color)
            painter.drawText(badge_rect, Qt.AlignmentFlag.AlignCenter, badge)


class StereoSegmentMeasurementWindow(QMainWindow):
    """Operator window for measuring straight segments from stereo pairs."""

    def __init__(
        self,
        manifest_path: str | None = None,
        calibration_path: str | None = None,
        *,
        preset_key: str = "generic",
        parent=None,
    ):
        super().__init__(parent)
        self.active_preset = preset_by_key(preset_key)
        self.setWindowTitle(self.active_preset.report_title)
        self.manifest_paths: list[Path] = []
        self.manifest: dict = {}
        self.image_pairs: list[tuple[Path, Path]] = []
        self.calibration_path: Path | None = None
        self.calibration_artifact: dict | None = None
        self.rectification_maps = None
        self.current_result: StereoSegmentMeasurementResult | None = None
        self.reference_result: StereoSegmentMeasurementResult | None = None
        self.reference_check: StereoSegmentReferenceCheck | None = None
        self.saved_results: list[dict] = []
        self._auto_corrected_right_order = False
        self._reference_auto_corrected_right_order = False
        self._active_line = "target"
        self._target_error = ""
        self._reference_error = ""

        self._build_ui()
        self._apply_preset(clear_results=False)
        if manifest_path:
            self.load_manifest(Path(manifest_path))
        if calibration_path:
            self.load_calibration(Path(calibration_path))
        resize_to_available_screen(self, 1450, 880, min_width=960, min_height=640)

    def _build_ui(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)
        self.setCentralWidget(central)

        top = QGridLayout()
        self.manifest_edit = QLineEdit()
        self.manifest_edit.setReadOnly(True)
        self.manifest_edit.setPlaceholderText("TritonPilot stereo session folder or manifest")
        self.open_manifest_btn = QPushButton("Open Session")
        self.open_manifest_btn.clicked.connect(self._choose_manifest)
        self.calibration_edit = QLineEdit()
        self.calibration_edit.setReadOnly(True)
        self.calibration_edit.setPlaceholderText("stereo_calibration.json")
        self.open_calibration_btn = QPushButton("Open Calibration")
        self.open_calibration_btn.clicked.connect(self._choose_calibration)
        top.addWidget(QLabel("Session"), 0, 0)
        top.addWidget(self.manifest_edit, 0, 1)
        top.addWidget(self.open_manifest_btn, 0, 2)
        top.addWidget(QLabel("Calibration"), 1, 0)
        top.addWidget(self.calibration_edit, 1, 1)
        top.addWidget(self.open_calibration_btn, 1, 2)
        root.addLayout(top)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter, 1)

        workspace = QWidget()
        workspace_layout = QVBoxLayout(workspace)
        workspace_layout.setContentsMargins(0, 0, 0, 0)
        workspace_layout.setSpacing(8)
        splitter.addWidget(workspace)

        image_splitter = QSplitter(Qt.Orientation.Horizontal)
        left_wrap = QWidget()
        left_layout = QVBoxLayout(left_wrap)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(4)
        self.left_title = QLabel("Rectified Left")
        self.left_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.left_canvas = _StereoKeelCanvas("Left")
        self.left_canvas.pointsChanged.connect(self._points_changed)
        left_layout.addWidget(self.left_title)
        left_layout.addWidget(self.left_canvas, 1)

        right_wrap = QWidget()
        right_layout = QVBoxLayout(right_wrap)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(4)
        self.right_title = QLabel("Rectified Right")
        self.right_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.right_canvas = _StereoKeelCanvas("Right")
        self.right_canvas.pointsChanged.connect(self._points_changed)
        right_layout.addWidget(self.right_title)
        right_layout.addWidget(self.right_canvas, 1)

        image_splitter.addWidget(left_wrap)
        image_splitter.addWidget(right_wrap)
        workspace_layout.addWidget(image_splitter, 4)

        self.summary_lbl = QLabel("Open a stereo session and calibration.")
        self.summary_lbl.setObjectName("summaryCard")
        self.summary_lbl.setWordWrap(True)
        self.summary_lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        workspace_layout.addWidget(self.summary_lbl, 0)

        self.pairs_table = QTableWidget(0, 5)
        self.pairs_table.setHorizontalHeaderLabels(["#", "Delta", "Left Seq", "Right Seq", "Stem"])
        self.pairs_table.verticalHeader().hide()
        self.pairs_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.pairs_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.pairs_table.currentCellChanged.connect(lambda row, *_args: self._show_pair(row))
        workspace_layout.addWidget(self.pairs_table, 1)

        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(8)
        splitter.addWidget(vertical_scroll_area(controls))

        loaded_card = _SectionCard("Loaded")
        loaded_form = QFormLayout()
        self.pair_lbl = self._value_label()
        self.image_size_lbl = self._value_label()
        self.calib_size_lbl = self._value_label()
        self.baseline_lbl = self._value_label()
        self.pair_delta_lbl = self._value_label()
        loaded_form.addRow("Pair", self.pair_lbl)
        loaded_form.addRow("Images", self.image_size_lbl)
        loaded_form.addRow("Calibration", self.calib_size_lbl)
        loaded_form.addRow("Baseline", self.baseline_lbl)
        loaded_form.addRow("Frame Delta", self.pair_delta_lbl)
        loaded_card.body.addLayout(loaded_form)
        controls_layout.addWidget(loaded_card)

        measure_card = _SectionCard("Measurement")
        measure_form = QFormLayout()
        self.preset_combo = QComboBox()
        for preset in STEREO_SEGMENT_PRESETS:
            self.preset_combo.addItem(preset.name, preset.key)
        preset_index = self.preset_combo.findData(self.active_preset.key)
        if preset_index >= 0:
            self.preset_combo.setCurrentIndex(preset_index)
        self.preset_combo.currentIndexChanged.connect(self._preset_changed)
        self.active_line_combo = QComboBox()
        self.active_line_combo.addItem("Variable Length", "target")
        self.active_line_combo.addItem("Known Reference", "reference")
        self.active_line_combo.currentIndexChanged.connect(self._active_line_changed)
        self.max_y_error_spin = QDoubleSpinBox()
        self.max_y_error_spin.setRange(0.25, 25.0)
        self.max_y_error_spin.setDecimals(2)
        self.max_y_error_spin.setSingleStep(0.25)
        self.max_y_error_spin.setValue(3.0)
        self.max_y_error_spin.setSuffix(" px")
        self.max_y_error_spin.valueChanged.connect(lambda _value: self._recalculate_measurement())
        self.min_disparity_spin = QDoubleSpinBox()
        self.min_disparity_spin.setRange(0.25, 2000.0)
        self.min_disparity_spin.setDecimals(2)
        self.min_disparity_spin.setSingleStep(0.5)
        self.min_disparity_spin.setValue(1.0)
        self.min_disparity_spin.setSuffix(" px")
        self.start_row_label = QLabel(self.active_preset.start_label)
        self.end_row_label = QLabel(self.active_preset.end_label)
        self.length_row_label = QLabel(self.active_preset.result_label)
        self.top_lbl = self._value_label()
        self.bottom_lbl = self._value_label()
        self.length_lbl = self._value_label()
        self.sensitivity_lbl = self._value_label()
        measure_form.addRow("Preset", self.preset_combo)
        measure_form.addRow("Active Line", self.active_line_combo)
        measure_form.addRow("Max Y Error", self.max_y_error_spin)
        measure_form.addRow("Min Disparity", self.min_disparity_spin)
        measure_form.addRow(self.start_row_label, self.top_lbl)
        measure_form.addRow(self.end_row_label, self.bottom_lbl)
        measure_form.addRow(self.length_row_label, self.length_lbl)
        measure_form.addRow("1 px Sensitivity", self.sensitivity_lbl)
        measure_card.body.addLayout(measure_form)
        measure_buttons = QHBoxLayout()
        self.clear_points_btn = QPushButton("Clear Line")
        self.clear_points_btn.clicked.connect(self._clear_points)
        self.clear_all_points_btn = QPushButton("Clear All")
        self.clear_all_points_btn.clicked.connect(self._clear_all_points)
        self.swap_right_btn = QPushButton("Swap Right")
        self.swap_right_btn.clicked.connect(self._swap_right_points)
        self.add_result_btn = QPushButton("Add Result")
        self.add_result_btn.clicked.connect(self._add_current_result)
        self.copy_report_btn = QPushButton("Copy Report")
        self.copy_report_btn.clicked.connect(self._copy_report)
        measure_buttons.addWidget(self.clear_points_btn)
        measure_buttons.addWidget(self.clear_all_points_btn)
        measure_buttons.addWidget(self.swap_right_btn)
        measure_buttons.addWidget(self.add_result_btn)
        measure_buttons.addWidget(self.copy_report_btn)
        measure_card.body.addLayout(measure_buttons)
        controls_layout.addWidget(measure_card)

        reference_card = _SectionCard("Reference Check")
        reference_form = QFormLayout()
        self.reference_length_spin = QDoubleSpinBox()
        self.reference_length_spin.setRange(1.0, 500.0)
        self.reference_length_spin.setDecimals(1)
        self.reference_length_spin.setSingleStep(1.0)
        self.reference_length_spin.setValue(100.0)
        self.reference_length_spin.setSuffix(" cm")
        self.reference_length_spin.valueChanged.connect(lambda _value: self._recalculate_measurement())
        self.reference_measured_lbl = self._value_label()
        self.reference_scale_lbl = self._value_label()
        self.reference_adjusted_lbl = self._value_label()
        reference_form.addRow("Known Length", self.reference_length_spin)
        reference_form.addRow("Measured", self.reference_measured_lbl)
        reference_form.addRow("Frame Scale", self.reference_scale_lbl)
        reference_form.addRow("Adjusted Length", self.reference_adjusted_lbl)
        reference_card.body.addLayout(reference_form)
        controls_layout.addWidget(reference_card)

        repeats_card = _SectionCard("Ensemble")
        self.repeat_summary_lbl = self._value_label()
        repeats_card.body.addWidget(self.repeat_summary_lbl)
        self.repeat_average_lbl = self._value_label()
        repeats_card.body.addWidget(self.repeat_average_lbl)
        self.results_table = QTableWidget(0, 7)
        self.results_table.setHorizontalHeaderLabels(["#", "Pair", "Length", "Line", "1px Sens", "Y Err", "Disp"])
        self.results_table.verticalHeader().hide()
        self.results_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.results_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.results_table.setWordWrap(False)
        repeats_card.body.addWidget(self.results_table)
        ensemble_buttons = QHBoxLayout()
        self.average_results_btn = QPushButton("Average")
        self.average_results_btn.clicked.connect(self._copy_average_results)
        self.clear_results_btn = QPushButton("Clear Results")
        self.clear_results_btn.clicked.connect(self._clear_results)
        ensemble_buttons.addWidget(self.average_results_btn)
        ensemble_buttons.addWidget(self.clear_results_btn)
        repeats_card.body.addLayout(ensemble_buttons)
        controls_layout.addWidget(repeats_card)
        controls_layout.addStretch(1)

        splitter.setSizes([1020, 400])
        self._refresh_controls()

    def _value_label(self) -> QLabel:
        label = QLabel("-")
        label.setWordWrap(True)
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        return label

    def _preset_changed(self, _index: int) -> None:
        self.active_preset = preset_by_key(self.preset_combo.currentData())
        self._apply_preset(clear_results=True)
        self._recalculate_measurement()

    def _active_line_changed(self, _index: int) -> None:
        self._active_line = str(self.active_line_combo.currentData() or "target")
        self.left_canvas.set_active_line(self._active_line)
        self.right_canvas.set_active_line(self._active_line)
        self._refresh_measurement_labels()
        self._refresh_controls()

    def _apply_preset(self, *, clear_results: bool) -> None:
        self.setWindowTitle(self.active_preset.report_title)
        if hasattr(self, "left_canvas"):
            self.left_canvas.set_point_labels(self.active_preset.start_label, self.active_preset.end_label)
            self.right_canvas.set_point_labels(self.active_preset.start_label, self.active_preset.end_label)
            self.left_canvas.set_reference_point_labels("Ref start", "Ref end")
            self.right_canvas.set_reference_point_labels("Ref start", "Ref end")
        if hasattr(self, "start_row_label"):
            self.start_row_label.setText(self.active_preset.start_label)
            self.end_row_label.setText(self.active_preset.end_label)
            self.length_row_label.setText(self.active_preset.result_label)
        if clear_results:
            self.saved_results.clear()
            self._populate_results_table()

    def _choose_manifest(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            "Open TritonPilot stereo session folder",
            str(self._stereo_session_start()),
        )
        if path:
            self.load_manifest(Path(path))

    def _choose_calibration(self) -> None:
        start = self._calibration_start()
        path, _filter = QFileDialog.getOpenFileName(
            self,
            "Open stereo calibration artifact",
            str(start),
            "Calibration JSON (*.json);;All files (*)",
        )
        if path:
            self.load_calibration(Path(path))

    def load_manifest(self, path: Path) -> None:
        try:
            self.manifest, self.image_pairs = load_manifest_collection([path])
        except Exception as exc:
            self._set_summary(f"Could not load stereo session: {exc}", tone="warn")
            self.statusBar().showMessage(str(exc), 7000)
            return
        self.manifest_paths = [Path(source["path"]) for source in self.manifest.get("sources", [])]
        if self.manifest_paths:
            display = self.manifest_paths[0] if len(self.manifest_paths) == 1 else self.manifest_paths[0].parent
            self.manifest_edit.setText(str(display))
        self.saved_results.clear()
        self._populate_results_table()
        self._populate_loaded_labels()
        self._populate_pairs_table()
        default_calibration = self._default_calibration_path()
        if default_calibration and default_calibration.exists() and self.calibration_artifact is None:
            self.load_calibration(default_calibration)
        if self.image_pairs:
            self.pairs_table.selectRow(0)
            self._show_pair(0)
        self._refresh_controls()
        self.statusBar().showMessage(f"Loaded {len(self.image_pairs)} stereo pair(s)", 4000)

    def load_calibration(self, path: Path) -> None:
        try:
            artifact = load_depth_calibration(path)
            self.rectification_maps = rectification_maps_from_artifact(artifact)
        except Exception as exc:
            self._set_summary(f"Could not load calibration: {exc}", tone="warn")
            self.statusBar().showMessage(str(exc), 7000)
            return
        self.calibration_artifact = artifact
        self.calibration_path = Path(path)
        self.calibration_edit.setText(str(path))
        self._populate_loaded_labels()
        if self.image_pairs:
            row = self.pairs_table.currentRow()
            self._show_pair(row if row >= 0 else 0)
        self._refresh_controls()
        self.statusBar().showMessage(f"Loaded calibration: {path}", 5000)

    def _default_calibration_path(self) -> Path | None:
        if not self.manifest_paths:
            return None
        first = self.manifest_paths[0]
        session_dir = first.parent if first.name.lower() == "manifest.json" else first
        if not session_dir.is_dir():
            session_dir = first.parent
        session_name = session_dir.name
        calibration_dir = workspace_paths(create=True).calibrations
        candidates = [
            session_dir / "stereo_calibration.json",
            calibration_dir / f"{session_name}_stereo_calibration.json",
            calibration_dir / "stereo_calibration.json",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[1]

    @staticmethod
    def _stereo_session_start() -> Path:
        return latest_pilot_stereo_sessions_dir(create=True)

    def _calibration_start(self) -> Path:
        default = self._default_calibration_path()
        if default is not None and default.exists():
            return default.parent
        return workspace_paths(create=True).calibrations

    def _populate_loaded_labels(self) -> None:
        pair = self.manifest.get("pair") or {}
        self.pair_lbl.setText(str(pair.get("name") or "-"))
        frames = self.manifest.get("frames") or []
        first = frames[0] if frames else {}
        shape = (first.get("left") or {}).get("shape")
        if isinstance(shape, list) and len(shape) >= 2:
            self.image_size_lbl.setText(f"{shape[1]}x{shape[0]}")
        else:
            self.image_size_lbl.setText("-")
        if self.calibration_artifact:
            size = self.calibration_artifact.get("image_size") or []
            self.calib_size_lbl.setText(f"{size[0]}x{size[1]}" if len(size) == 2 else "-")
            baseline = (self.calibration_artifact.get("stereo") or {}).get("baseline")
            self.baseline_lbl.setText(self._format_distance(float(baseline)) if baseline is not None else "-")
        else:
            self.calib_size_lbl.setText("-")
            self.baseline_lbl.setText("-")

    def _populate_pairs_table(self) -> None:
        self.pairs_table.setRowCount(0)
        for row, frame in enumerate(self.manifest.get("frames") or []):
            self.pairs_table.insertRow(row)
            values = [
                str(frame.get("index", row + 1)),
                f"{float(frame.get('pair_delta_ms', 0.0)):.1f} ms",
                str((frame.get("left") or {}).get("seq", "-")),
                str((frame.get("right") or {}).get("seq", "-")),
                str(frame.get("stem", "")),
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.pairs_table.setItem(row, col, item)
        self.pairs_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)

    def _show_pair(self, row: int) -> None:
        if row < 0 or row >= len(self.image_pairs):
            return
        self.current_result = None
        self.reference_result = None
        self.reference_check = None
        left_path, right_path = self.image_pairs[row]
        left_image = cv2.imread(str(left_path), cv2.IMREAD_COLOR)
        right_image = cv2.imread(str(right_path), cv2.IMREAD_COLOR)
        if left_image is None or right_image is None:
            self.left_canvas.set_frame(None, placeholder="Left image unreadable")
            self.right_canvas.set_frame(None, placeholder="Right image unreadable")
            self._set_summary("One or both stereo images could not be read.", tone="warn")
            self._refresh_controls()
            return
        frame = (self.manifest.get("frames") or [])[row]
        self.pair_delta_lbl.setText(f"{float(frame.get('pair_delta_ms', 0.0)):.1f} ms")
        if self.rectification_maps is None:
            self.left_canvas.set_frame(left_image, placeholder="Load calibration")
            self.right_canvas.set_frame(right_image, placeholder="Load calibration")
            self._set_summary("Calibration not loaded.", tone="warn")
            self._refresh_controls()
            return
        try:
            left_rect, right_rect = rectify_stereo_images(left_image, right_image, self.rectification_maps)
        except Exception as exc:
            self.left_canvas.set_frame(None, placeholder="Rectification failed")
            self.right_canvas.set_frame(None, placeholder="Rectification failed")
            self._set_summary(f"Rectification failed: {exc}", tone="warn")
            self.statusBar().showMessage(str(exc), 7000)
            self._refresh_controls()
            return
        self.left_canvas.set_frame(left_rect)
        self.right_canvas.set_frame(right_rect)
        self._set_summary(f"Pair {row + 1}: ready for endpoint clicks.")
        self._refresh_measurement_labels()
        self._refresh_controls()

    def _points_changed(self) -> None:
        self._auto_corrected_right_order = False
        self._reference_auto_corrected_right_order = False
        self._recalculate_measurement()

    def _recalculate_measurement(self) -> None:
        self.current_result = None
        self.reference_result = None
        self.reference_check = None
        self._target_error = ""
        self._reference_error = ""
        self.left_canvas.set_badge("", line="target")
        self.right_canvas.set_badge("", line="target")
        self.left_canvas.set_badge("", line="reference")
        self.right_canvas.set_badge("", line="reference")
        self._auto_corrected_right_order = False
        self._reference_auto_corrected_right_order = False
        if self.rectification_maps is None:
            if self.left_canvas.points("target") or self.right_canvas.points("target"):
                self._target_error = "Load calibration"
            if self.left_canvas.points("reference") or self.right_canvas.points("reference"):
                self._reference_error = "Load calibration"
            self._refresh_controls()
            self._refresh_measurement_labels()
            return

        self.current_result, self._target_error = self._measure_canvas_line(
            "target",
            preset_key=self.active_preset.key,
        )
        self.reference_result, self._reference_error = self._measure_canvas_line(
            "reference",
            preset_key="generic",
        )
        if self.current_result is not None:
            badge = self._format_result_length(self.current_result)
            self.left_canvas.set_badge(badge, line="target")
            self.right_canvas.set_badge(badge, line="target")
        if self.reference_result is not None:
            self.left_canvas.set_badge(self._format_reference_badge(self.reference_result), line="reference")
            self.right_canvas.set_badge(self._format_reference_badge(self.reference_result), line="reference")
            try:
                self.reference_check = evaluate_reference_scale_check(
                    reference_result=self.reference_result,
                    known_length_cm=float(self.reference_length_spin.value()),
                    target_result=self.current_result,
                )
            except Exception as exc:
                self._reference_error = str(exc)

        self._set_measurement_summary()
        self._refresh_measurement_labels()
        self._refresh_controls()

    def _measure_canvas_line(
        self,
        line: str,
        *,
        preset_key: str,
    ) -> tuple[StereoSegmentMeasurementResult | None, str]:
        left = self.left_canvas.points(line)
        right = self.right_canvas.points(line)
        if len(left) < 2 or len(right) < 2:
            return None, ""
        if right_endpoint_order_mismatch(
            left_start_pixel=left[0],
            left_end_pixel=left[1],
            right_start_pixel=right[0],
            right_end_pixel=right[1],
        ):
            self.right_canvas.swap_points(emit=False, line=line)
            right = self.right_canvas.points(line)
            if line == "reference":
                self._reference_auto_corrected_right_order = True
                message = "Swapped reference right endpoints to match the left endpoint order."
            else:
                self._auto_corrected_right_order = True
                message = "Swapped right endpoints to match the left endpoint order."
            self.statusBar().showMessage(message, 5000)
        try:
            return (
                measure_stereo_segment(
                    q=self.rectification_maps.q,
                    start_left_pixel=left[0],
                    start_right_pixel=right[0],
                    end_left_pixel=left[1],
                    end_right_pixel=right[1],
                    units=self._units(),
                    preset_key=preset_key,
                    min_abs_disparity=float(self.min_disparity_spin.value()),
                    max_vertical_error_px=float(self.max_y_error_spin.value()),
                ),
                "",
            )
        except Exception as exc:
            return None, str(exc)

    def _set_measurement_summary(self) -> None:
        pieces = []
        tone = None
        if self.current_result is not None:
            pieces.append(
                f"{self.active_preset.result_label}: {self._format_result_length(self.current_result)} | "
                f"stability {self._format_stability(self.current_result)} | "
                f"y error max {self.current_result.max_vertical_error_px:.2f} px | "
                f"min disparity {self.current_result.min_abs_disparity_px:.1f} px"
            )
            if self._auto_corrected_right_order:
                pieces.append("right order auto-corrected")
        elif self._target_error:
            pieces.append(self._target_error)
            tone = "warn"

        if self.reference_check is not None:
            pieces.append(self._format_reference_check(self.reference_check))
            if self.reference_check.abs_error_cm > 5.0:
                tone = "warn"
        elif self._reference_error:
            pieces.append(f"Reference: {self._reference_error}")
            tone = "warn"
        elif self.reference_result is not None:
            pieces.append(f"Reference: {self._format_result_length(self.reference_result)}")

        if not pieces:
            active = "reference" if self._active_line == "reference" else "variable"
            pieces.append(f"Pair {self.pairs_table.currentRow() + 1}: ready for {active} endpoint clicks.")
        self._set_summary(" | ".join(pieces), tone=tone)

    def _refresh_measurement_labels(self) -> None:
        left = self.left_canvas.points("target")
        right = self.right_canvas.points("target")
        if self.current_result is not None:
            self.top_lbl.setText(self._format_sample(self.current_result.start))
            self.bottom_lbl.setText(self._format_sample(self.current_result.end))
            self.length_lbl.setText(self._format_result_length(self.current_result))
            self.sensitivity_lbl.setText(self._format_sensitivity(self.current_result))
        else:
            self.top_lbl.setText(self._format_pending_endpoint(0, left, right))
            self.bottom_lbl.setText(self._format_pending_endpoint(1, left, right))
            self.length_lbl.setText(self._target_error or "-")
            self.sensitivity_lbl.setText("-")
        self._refresh_reference_labels()

    def _refresh_reference_labels(self) -> None:
        ref_left = self.left_canvas.points("reference")
        ref_right = self.right_canvas.points("reference")
        if self.reference_result is not None:
            self.reference_measured_lbl.setText(self._format_reference_result(self.reference_result))
        elif self._reference_error:
            self.reference_measured_lbl.setText(self._reference_error)
        else:
            start = self._format_pending_endpoint(0, ref_left, ref_right)
            end = self._format_pending_endpoint(1, ref_left, ref_right)
            self.reference_measured_lbl.setText("-" if start == "-" and end == "-" else f"{start} | {end}")

        if self.reference_check is not None:
            self.reference_scale_lbl.setText(self._format_reference_check(self.reference_check))
            self.reference_adjusted_lbl.setText(self._format_adjusted_length(self.reference_check))
        else:
            self.reference_scale_lbl.setText("-")
            self.reference_adjusted_lbl.setText("-")

    def _format_pending_endpoint(
        self,
        index: int,
        left: list[tuple[int, int]],
        right: list[tuple[int, int]],
    ) -> str:
        parts = []
        if index < len(left):
            parts.append(f"L ({left[index][0]}, {left[index][1]})")
        if index < len(right):
            parts.append(f"R ({right[index][0]}, {right[index][1]})")
        return " | ".join(parts) if parts else "-"

    def _format_sample(self, sample: CorrespondenceSample) -> str:
        point = sample.point
        return (
            f"L ({sample.left_pixel[0]}, {sample.left_pixel[1]}) | "
            f"R ({sample.right_pixel[0]}, {sample.right_pixel[1]}) | "
            f"yerr {abs(sample.vertical_error_px):.2f} px | d {sample.disparity:.1f} px | "
            f"X {point[0]:.1f}, Y {point[1]:.1f}, Z {point[2]:.1f} {self._units()}"
        ).strip()

    @staticmethod
    def _format_pixel(point: tuple[int, int]) -> str:
        return f"({int(point[0])}, {int(point[1])})"

    def _format_result_line(self, record: dict) -> str:
        result = record["result"]
        start_label = str(record.get("start_label") or preset_by_key(result.preset_key).start_label)
        end_label = str(record.get("end_label") or preset_by_key(result.preset_key).end_label)
        return (
            f"{start_label} L{self._format_pixel(result.start.left_pixel)} "
            f"R{self._format_pixel(result.start.right_pixel)} -> "
            f"{end_label} L{self._format_pixel(result.end.left_pixel)} "
            f"R{self._format_pixel(result.end.right_pixel)}"
        )

    def _format_summary_length(self, value_units: float | None, value_cm: float | None) -> str:
        if value_cm is not None:
            return f"{value_cm:.1f} cm ({value_cm / 100.0:.3f} m)"
        return self._format_distance(value_units)

    def _average_results_text(self) -> str:
        if not self.saved_results:
            return "-"
        summary = summarize_segment_measurements([record["result"] for record in self.saved_results])
        average_text = self._format_summary_length(summary.mean_length_units, summary.mean_length_cm)
        return f"Average: {average_text} from {summary.count} result(s)"

    def _clear_points(self) -> None:
        self.current_result = None
        self.reference_result = None
        self.reference_check = None
        self._auto_corrected_right_order = False
        self._reference_auto_corrected_right_order = False
        self.left_canvas.clear_points(line=self._active_line)
        self.right_canvas.clear_points(line=self._active_line)
        self._refresh_measurement_labels()
        self._refresh_controls()

    def _clear_all_points(self) -> None:
        self.current_result = None
        self.reference_result = None
        self.reference_check = None
        self._auto_corrected_right_order = False
        self._reference_auto_corrected_right_order = False
        self.left_canvas.clear_all_points()
        self.right_canvas.clear_all_points()
        self._refresh_measurement_labels()
        self._refresh_controls()

    def _swap_right_points(self) -> None:
        self.right_canvas.swap_points(emit=False, line=self._active_line)
        self.statusBar().showMessage("Right endpoint order swapped.", 4000)
        self._recalculate_measurement()

    def _add_current_result(self) -> None:
        if self.current_result is None:
            return
        row = self.pairs_table.currentRow()
        frame = (self.manifest.get("frames") or [])[row] if 0 <= row < len(self.manifest.get("frames") or []) else {}
        record = {
            "pair_index": row + 1,
            "stem": str(frame.get("stem") or ""),
            "delta_ms": float(frame.get("pair_delta_ms", 0.0)),
            "start_label": self.active_preset.start_label,
            "end_label": self.active_preset.end_label,
            "result": self.current_result,
        }
        self.saved_results.append(record)
        self._populate_results_table()
        self.statusBar().showMessage("Stereo segment measurement added.", 4000)
        self._refresh_controls()

    def _populate_results_table(self) -> None:
        self.results_table.setRowCount(0)
        for row, record in enumerate(self.saved_results):
            result = record["result"]
            self.results_table.insertRow(row)
            values = [
                str(row + 1),
                str(record["pair_index"]),
                self._format_result_length(result),
                self._format_result_line(record),
                self._format_sensitivity(result),
                f"{result.max_vertical_error_px:.2f} px",
                f"{result.min_abs_disparity_px:.1f} px",
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setTextAlignment(
                    Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
                    if col == 3
                    else Qt.AlignmentFlag.AlignCenter
                )
                if col == 3:
                    item.setToolTip(value)
                self.results_table.setItem(row, col, item)
        header = self.results_table.horizontalHeader()
        for col in range(self.results_table.columnCount()):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        if self.results_table.columnCount() > 3:
            header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self._update_repeat_summary()

    def _update_repeat_summary(self) -> None:
        results = [record["result"] for record in self.saved_results]
        if not results:
            self.repeat_summary_lbl.setText("Add measurements from this stereo session to build an ensemble.")
            self.repeat_average_lbl.setText("Average: -")
            return
        summary = summarize_segment_measurements(results)
        average_text = self._format_summary_length(summary.mean_length_units, summary.mean_length_cm)
        median_text = self._format_summary_length(summary.median_length_units, summary.median_length_cm)
        spread_text = self._format_summary_length(summary.spread_units, summary.spread_cm)
        self.repeat_summary_lbl.setText(
            f"{summary.count} result(s) | average {average_text} | median {median_text} | spread {spread_text}"
        )
        self.repeat_average_lbl.setText(f"Average: {average_text}")

    def _clear_results(self) -> None:
        self.saved_results.clear()
        self._populate_results_table()
        self._refresh_controls()

    def _copy_average_results(self) -> None:
        if not self.saved_results:
            return
        text = self._average_results_text()
        QApplication.clipboard().setText(text)
        self.repeat_average_lbl.setText(text)
        self.statusBar().showMessage("Ensemble average copied.", 4000)

    def _copy_report(self) -> None:
        QApplication.clipboard().setText(self._report_text())
        self.statusBar().showMessage("Stereo segment report copied.", 4000)

    def _report_text(self) -> str:
        lines = [
            self.active_preset.report_title,
            f"Preset: {self.active_preset.name}",
            f"Session: {self.manifest_edit.text() or '-'}",
            f"Calibration: {self.calibration_edit.text() or '-'}",
            f"Pair: {self.pair_lbl.text()}",
            f"Baseline: {self.baseline_lbl.text()}",
        ]
        if self.current_result is not None:
            lines.extend(
                [
                    "",
                    f"Current length: {self._format_result_length(self.current_result)}",
                    f"{self.active_preset.start_label}: {self._format_sample(self.current_result.start)}",
                    f"{self.active_preset.end_label}: {self._format_sample(self.current_result.end)}",
                    f"1 px sensitivity: {self._format_sensitivity(self.current_result)}",
                    f"Max vertical error: {self.current_result.max_vertical_error_px:.2f} px",
                    f"Min disparity: {self.current_result.min_abs_disparity_px:.1f} px",
                ]
            )
        if self.reference_result is not None:
            lines.extend(
                [
                    "",
                    f"Reference known length: {float(self.reference_length_spin.value()):.1f} cm",
                    f"Reference measured: {self._format_reference_result(self.reference_result)}",
                ]
            )
            if self.reference_check is not None:
                lines.append(f"Reference scale: {self._format_reference_check(self.reference_check)}")
                lines.append(f"Reference-adjusted length: {self._format_adjusted_length(self.reference_check)}")
        if self.saved_results:
            summary = summarize_segment_measurements([record["result"] for record in self.saved_results])
            average_text = self._format_summary_length(summary.mean_length_units, summary.mean_length_cm)
            median_text = self._format_summary_length(summary.median_length_units, summary.median_length_cm)
            spread_text = self._format_summary_length(summary.spread_units, summary.spread_cm)
            lines.extend(
                [
                    "",
                    f"Ensemble results: {summary.count}",
                    f"Average: {average_text}",
                    f"Median: {median_text}",
                    f"Spread: {spread_text}",
                ]
            )
            if summary.mean_length_cm is not None:
                lines.append(f"Average cm/m: {summary.mean_length_cm:.1f} cm / {summary.mean_length_m:.3f} m")
            if summary.median_length_cm is not None and summary.spread_cm is not None:
                lines.append(f"Median cm/m: {summary.median_length_cm:.1f} cm / {summary.median_length_m:.3f} m")
                lines.append(f"Spread cm: {summary.spread_cm:.1f} cm")
            for index, record in enumerate(self.saved_results, start=1):
                result = record["result"]
                lines.append(
                    f"{index}. pair {record['pair_index']} {record['stem']} "
                    f"{self._format_result_length(result)} "
                    f"line={self._format_result_line(record)} "
                    f"sens={self._format_sensitivity(result)} "
                    f"yerr={result.max_vertical_error_px:.2f}px disp={result.min_abs_disparity_px:.1f}px"
                )
        return "\n".join(lines)

    def _set_summary(self, text: str, *, tone: str | None = None) -> None:
        self.summary_lbl.setText(text)
        self.summary_lbl.setProperty("tone", tone or "")
        self.summary_lbl.style().unpolish(self.summary_lbl)
        self.summary_lbl.style().polish(self.summary_lbl)
        self.summary_lbl.update()

    def _units(self) -> str:
        if self.calibration_artifact is None:
            return ""
        return str((self.calibration_artifact.get("board") or {}).get("units") or "")

    def _format_result_length(self, result: StereoSegmentMeasurementResult) -> str:
        if result.length_cm is not None and result.length_m is not None:
            return f"{result.length_cm:.1f} cm ({result.length_m:.3f} m)"
        return self._format_distance(result.length_units)

    def _format_sensitivity(self, result: StereoSegmentMeasurementResult) -> str:
        if result.click_sensitivity_units is None:
            return "-"
        pixel_text = f"{result.click_sensitivity_px:g} px"
        if result.click_sensitivity_cm is not None:
            return f"+/-{result.click_sensitivity_cm:.1f} cm per {pixel_text}"
        return f"+/-{self._format_distance(result.click_sensitivity_units)} per {pixel_text}"

    def _format_reference_badge(self, result: StereoSegmentMeasurementResult) -> str:
        if result.length_cm is not None:
            return f"ref {result.length_cm:.1f} cm"
        return f"ref {self._format_distance(result.length_units)}"

    def _format_reference_result(self, result: StereoSegmentMeasurementResult) -> str:
        return (
            f"{self._format_result_length(result)} | stability {self._format_stability(result)} | "
            f"y error max {result.max_vertical_error_px:.2f} px | "
            f"min disparity {result.min_abs_disparity_px:.1f} px"
            + (" | right order auto-corrected" if self._reference_auto_corrected_right_order else "")
        )

    def _format_reference_check(self, check: StereoSegmentReferenceCheck) -> str:
        if check.abs_error_cm <= 2.0:
            label = "scale ok"
        elif check.abs_error_cm <= 5.0:
            label = "scale check"
        else:
            label = "scale warning"
        return (
            f"{label}: measured {check.measured_length_cm:.1f}/{check.known_length_cm:.1f} cm "
            f"({check.error_cm:+.1f} cm, {check.percent_error:+.1f}%), factor {check.scale_factor:.4f}"
        )

    def _format_adjusted_length(self, check: StereoSegmentReferenceCheck) -> str:
        if check.target_corrected_length_cm is None:
            return "-"
        return f"{check.target_corrected_length_cm:.1f} cm ({check.target_corrected_length_m:.3f} m)"

    def _format_stability(self, result: StereoSegmentMeasurementResult) -> str:
        sensitivity = self._format_sensitivity(result)
        if result.click_sensitivity_cm is None:
            return f"unknown ({sensitivity})"
        if result.click_sensitivity_cm <= 2.0 and result.max_vertical_error_px <= 1.5:
            label = "high"
        elif result.click_sensitivity_cm <= 5.0 and result.max_vertical_error_px <= 3.0:
            label = "check"
        else:
            label = "remeasure"
        return f"{label} ({sensitivity})"

    def _format_distance(self, value: float | None) -> str:
        if value is None:
            return "-"
        units = self._units().strip()
        if units.lower() == "mm":
            return f"{value:.1f} mm ({value / 10.0:.2f} cm)"
        if units.lower() == "cm":
            return f"{value:.2f} cm"
        if units.lower() in {"m", "meter", "meters"}:
            return f"{value:.3f} m"
        return f"{value:.2f} {units}".strip()

    def _refresh_controls(self) -> None:
        active_left = self.left_canvas.points(self._active_line)
        active_right = self.right_canvas.points(self._active_line)
        has_active_points = bool(active_left or active_right)
        has_any_points = bool(
            self.left_canvas.points("target")
            or self.right_canvas.points("target")
            or self.left_canvas.points("reference")
            or self.right_canvas.points("reference")
        )
        self.clear_points_btn.setEnabled(has_active_points)
        self.clear_all_points_btn.setEnabled(has_any_points)
        self.swap_right_btn.setEnabled(len(active_right) == 2)
        self.add_result_btn.setEnabled(self.current_result is not None)
        self.copy_report_btn.setEnabled(
            self.current_result is not None
            or self.reference_result is not None
            or bool(self.saved_results)
        )
        self.average_results_btn.setEnabled(bool(self.saved_results))
        self.clear_results_btn.setEnabled(bool(self.saved_results))


class StereoIcebergMeasurementWindow(StereoSegmentMeasurementWindow):
    """Compatibility wrapper that opens the segment app in iceberg mode."""

    def __init__(self, manifest_path: str | None = None, calibration_path: str | None = None, parent=None):
        super().__init__(
            manifest_path=manifest_path,
            calibration_path=calibration_path,
            preset_key="iceberg",
            parent=parent,
        )
