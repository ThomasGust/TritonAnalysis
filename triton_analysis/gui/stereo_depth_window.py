"""Stereo depth and 3D length measurement applet."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import QRect, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from triton_analysis.gui.file_dialogs import ThumbnailFileDialog as QFileDialog

from triton_analysis.workspace import workspace_paths
from triton_analysis.gui.image_preview import ImagePreviewPanel, frame_to_pixmap
from triton_analysis.gui.responsive import resize_to_available_screen, vertical_scroll_area
from triton_analysis.stereo.calibration import load_manifest_collection
from triton_analysis.stereo.depth import (
    CorrespondenceSample,
    DepthSample,
    analyze_charuco_stereo_geometry,
    colorize_depth,
    colorize_disparity,
    compute_disparity,
    distance_between_samples,
    load_depth_calibration,
    point_from_rectified_correspondence,
    rectification_maps_from_artifact,
    rectify_stereo_images,
    reproject_disparity,
    sample_depth_point,
    normalized_num_disparities,
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


class _MeasurementCanvas(QWidget):
    clickedImagePoint = pyqtSignal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap = frame_to_pixmap(None)
        self._image_size: tuple[int, int] | None = None
        self._points: list[tuple[int, int]] = []
        self._badge = ""
        self._placeholder = "No rectified image"
        self.setMinimumSize(260, 220)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)

    def set_frame(self, frame_bgr: np.ndarray | None, *, placeholder: str = "No rectified image") -> None:
        self._placeholder = placeholder
        self._pixmap = frame_to_pixmap(frame_bgr)
        if frame_bgr is None:
            self._image_size = None
        else:
            self._image_size = (int(frame_bgr.shape[1]), int(frame_bgr.shape[0]))
        self.update()

    def clear(self, placeholder: str = "No rectified image") -> None:
        self.set_frame(None, placeholder=placeholder)

    def set_measurement(self, points: list[tuple[int, int]], badge: str = "") -> None:
        self._points = list(points)
        self._badge = str(badge or "")
        self.update()

    def _target_rect(self) -> QRect:
        if self._pixmap.isNull() or not self._image_size:
            return QRect()
        image_w, image_h = self._image_size
        if image_w <= 0 or image_h <= 0:
            return QRect()
        margin = 8
        available_w = max(1, self.width() - margin * 2)
        available_h = max(1, self.height() - margin * 2)
        scale = min(available_w / float(image_w), available_h / float(image_h))
        draw_w = max(1, int(round(image_w * scale)))
        draw_h = max(1, int(round(image_h * scale)))
        return QRect((self.width() - draw_w) // 2, (self.height() - draw_h) // 2, draw_w, draw_h)

    def _image_to_widget(self, point: tuple[int, int]) -> tuple[float, float] | None:
        rect = self._target_rect()
        if rect.isNull() or not self._image_size:
            return None
        image_w, image_h = self._image_size
        return (
            rect.left() + (float(point[0]) / max(1.0, image_w - 1.0)) * rect.width(),
            rect.top() + (float(point[1]) / max(1.0, image_h - 1.0)) * rect.height(),
        )

    def _widget_to_image(self, x: float, y: float) -> tuple[float, float] | None:
        rect = self._target_rect()
        if rect.isNull() or not self._image_size or not rect.contains(int(x), int(y)):
            return None
        image_w, image_h = self._image_size
        image_x = (float(x) - rect.left()) / max(1.0, rect.width()) * max(1.0, image_w - 1.0)
        image_y = (float(y) - rect.top()) / max(1.0, rect.height()) * max(1.0, image_h - 1.0)
        return image_x, image_y

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        point = self._widget_to_image(float(event.position().x()), float(event.position().y()))
        if point is not None:
            self.clickedImagePoint.emit(float(point[0]), float(point[1]))

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(14, 14, 18))
        rect = self._target_rect()
        if rect.isNull():
            painter.setPen(QColor(180, 184, 194))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._placeholder)
            return

        painter.drawPixmap(rect, self._pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        widget_points = [self._image_to_widget(point) for point in self._points]
        widget_points = [point for point in widget_points if point is not None]
        if len(widget_points) == 2:
            painter.setPen(QPen(QColor(255, 230, 120), 3))
            painter.drawLine(
                int(round(widget_points[0][0])),
                int(round(widget_points[0][1])),
                int(round(widget_points[1][0])),
                int(round(widget_points[1][1])),
            )
        for index, point in enumerate(widget_points, start=1):
            x, y = point
            painter.setPen(QPen(QColor(10, 10, 10), 5))
            painter.drawEllipse(int(round(x)) - 7, int(round(y)) - 7, 14, 14)
            painter.setPen(QPen(QColor(70, 255, 130), 3))
            painter.drawEllipse(int(round(x)) - 7, int(round(y)) - 7, 14, 14)
            painter.setPen(QColor(245, 248, 255))
            painter.drawText(int(round(x)) + 10, int(round(y)) - 10, f"P{index}")
        if self._badge and len(widget_points) == 2:
            mid_x = int(round((widget_points[0][0] + widget_points[1][0]) * 0.5))
            mid_y = int(round((widget_points[0][1] + widget_points[1][1]) * 0.5))
            metrics = painter.fontMetrics()
            badge_rect = QRect(mid_x - 90, mid_y - 28, max(180, metrics.horizontalAdvance(self._badge) + 18), 24)
            painter.fillRect(badge_rect, QColor(20, 22, 28, 220))
            painter.setPen(QColor(255, 230, 120))
            painter.drawText(badge_rect, Qt.AlignmentFlag.AlignCenter, self._badge)


class StereoDepthWindow(QMainWindow):
    """Inspect rectification, disparity, depth, and 3D measurements."""

    def __init__(self, manifest_path: str | None = None, calibration_path: str | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Stereo Depth")
        self.manifest_paths: list[Path] = []
        self.manifest: dict = {}
        self.image_pairs: list[tuple[Path, Path]] = []
        self.calibration_path: Path | None = None
        self.calibration_artifact: dict | None = None
        self.rectification_maps = None
        self.disparity: np.ndarray | None = None
        self.points_3d: np.ndarray | None = None
        self.valid_depth: np.ndarray | None = None
        self.samples: list[DepthSample | CorrespondenceSample] = []
        self.left_measure_points: list[tuple[int, int]] = []
        self.right_measure_points: list[tuple[int, int]] = []
        self.geometry_check: dict | None = None

        self._build_ui()
        if manifest_path:
            self.load_manifest(Path(manifest_path))
        if calibration_path:
            self.load_calibration(Path(calibration_path))
        resize_to_available_screen(self, 1500, 900, min_width=980, min_height=650)

    def _build_ui(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)
        self.setCentralWidget(central)

        top = QGridLayout()
        self.manifest_edit = QLineEdit()
        self.manifest_edit.setReadOnly(True)
        self.manifest_edit.setPlaceholderText("TritonPilot stereo manifest or session folder")
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

        rectified_splitter = QSplitter(Qt.Orientation.Horizontal)
        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(4)
        left_title = QLabel("Rectified Left")
        left_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.left_canvas = _MeasurementCanvas()
        self.left_canvas.clickedImagePoint.connect(self._on_left_measurement_click)
        left_layout.addWidget(left_title)
        left_layout.addWidget(self.left_canvas, 1)
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(4)
        right_title = QLabel("Rectified Right")
        right_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.right_canvas = _MeasurementCanvas()
        self.right_canvas.clickedImagePoint.connect(self._on_right_measurement_click)
        right_layout.addWidget(right_title)
        right_layout.addWidget(self.right_canvas, 1)
        rectified_splitter.addWidget(left_container)
        rectified_splitter.addWidget(right_container)
        workspace_layout.addWidget(rectified_splitter, 3)

        depth_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.disparity_preview = ImagePreviewPanel("Disparity")
        self.depth_preview = ImagePreviewPanel("Depth")
        depth_splitter.addWidget(self.disparity_preview)
        depth_splitter.addWidget(self.depth_preview)
        workspace_layout.addWidget(depth_splitter, 2)

        self.summary_lbl = QLabel("Open a stereo session and calibration.")
        self.summary_lbl.setObjectName("summaryCard")
        self.summary_lbl.setWordWrap(True)
        self.summary_lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        workspace_layout.addWidget(self.summary_lbl, 0)

        self.pairs_table = QTableWidget(0, 6)
        self.pairs_table.setHorizontalHeaderLabels(["#", "Delta", "Left Seq", "Right Seq", "Stem", "Source"])
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

        session_card = _SectionCard("Loaded")
        session_form = QFormLayout()
        self.pair_lbl = self._value_label()
        self.image_size_lbl = self._value_label()
        self.calib_size_lbl = self._value_label()
        self.baseline_lbl = self._value_label()
        self.board_check_lbl = self._value_label()
        session_form.addRow("Pair", self.pair_lbl)
        session_form.addRow("Images", self.image_size_lbl)
        session_form.addRow("Calibration", self.calib_size_lbl)
        session_form.addRow("Baseline", self.baseline_lbl)
        session_form.addRow("Board Check", self.board_check_lbl)
        session_card.body.addLayout(session_form)
        controls_layout.addWidget(session_card)

        disparity_card = _SectionCard("Disparity")
        disparity_form = QFormLayout()
        self.min_disparity_spin = self._int_spin(-128, 128, 0, step=1)
        self.num_disparities_spin = self._int_spin(16, 1024, 320, step=16)
        self.block_size_spin = self._int_spin(3, 31, 7, step=2)
        self.uniqueness_spin = self._int_spin(0, 30, 8, step=1)
        self.speckle_window_spin = self._int_spin(0, 300, 80, step=10)
        self.speckle_range_spin = self._int_spin(0, 32, 2, step=1)
        self.clahe_check = QCheckBox()
        self.clahe_check.setChecked(True)
        self.lr_check = QCheckBox()
        self.lr_check.setChecked(True)
        self.max_depth_spin = QDoubleSpinBox()
        self.max_depth_spin.setRange(0.0, 1000000.0)
        self.max_depth_spin.setDecimals(1)
        self.max_depth_spin.setSingleStep(100.0)
        self.max_depth_spin.setValue(0.0)
        for label, field in (
            ("Min Disparity", self.min_disparity_spin),
            ("Num Disparities", self.num_disparities_spin),
            ("Block Size", self.block_size_spin),
            ("Uniqueness", self.uniqueness_spin),
            ("Speckle Window", self.speckle_window_spin),
            ("Speckle Range", self.speckle_range_spin),
            ("CLAHE", self.clahe_check),
            ("L/R Check", self.lr_check),
            ("Max |Depth|", self.max_depth_spin),
        ):
            disparity_form.addRow(label, field)
        disparity_card.body.addLayout(disparity_form)
        self.compute_btn = QPushButton("Compute")
        self.compute_btn.clicked.connect(lambda: self._show_pair(self.pairs_table.currentRow(), force=True))
        disparity_card.body.addWidget(self.compute_btn)
        controls_layout.addWidget(disparity_card)

        measure_card = _SectionCard("Measurement")
        measure_form = QFormLayout()
        self.sample_radius_spin = self._int_spin(0, 25, 4, step=1)
        self.point1_lbl = self._value_label()
        self.point2_lbl = self._value_label()
        self.distance_lbl = self._value_label()
        measure_form.addRow("Sample Radius", self.sample_radius_spin)
        measure_form.addRow("P1", self.point1_lbl)
        measure_form.addRow("P2", self.point2_lbl)
        measure_form.addRow("Distance", self.distance_lbl)
        measure_card.body.addLayout(measure_form)
        self.clear_points_btn = QPushButton("Clear Points")
        self.clear_points_btn.clicked.connect(self._clear_measurement)
        measure_card.body.addWidget(self.clear_points_btn)
        controls_layout.addWidget(measure_card)
        controls_layout.addStretch(1)

        splitter.setSizes([1040, 420])

    def _value_label(self) -> QLabel:
        label = QLabel("-")
        label.setWordWrap(True)
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        return label

    def _int_spin(self, minimum: int, maximum: int, value: int, *, step: int) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        spin.setSingleStep(step)
        return spin

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
        self.manifest, self.image_pairs = load_manifest_collection([path])
        self.manifest_paths = [Path(source["path"]) for source in self.manifest.get("sources", [])]
        if self.manifest_paths:
            self.manifest_edit.setText(str(self.manifest_paths[0] if len(self.manifest_paths) == 1 else self.manifest_paths[0].parent))
        self._populate_loaded_labels()
        self._populate_pairs_table()
        default_calibration = self._default_calibration_path()
        if default_calibration and default_calibration.exists() and self.calibration_artifact is None:
            self.load_calibration(default_calibration)
        if self.image_pairs:
            self.pairs_table.selectRow(0)
            self._show_pair(0)
        self.statusBar().showMessage(f"Loaded {len(self.image_pairs)} stereo pair(s)", 4000)

    def load_calibration(self, path: Path) -> None:
        artifact = load_depth_calibration(path)
        self.calibration_artifact = artifact
        self.rectification_maps = rectification_maps_from_artifact(artifact)
        self.calibration_path = Path(path)
        self.calibration_edit.setText(str(path))
        if float(self.max_depth_spin.value()) <= 0.0:
            default_max_depth = self._default_max_depth_for_units(self._units())
            if default_max_depth > 0.0:
                self.max_depth_spin.setValue(default_max_depth)
        self._populate_loaded_labels()
        if self.image_pairs:
            row = self.pairs_table.currentRow()
            self._show_pair(row if row >= 0 else 0, force=True)
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
        workspace = workspace_paths(create=True)
        stereo_sessions = workspace.pilot_incoming / "stereo_sessions"
        return stereo_sessions if stereo_sessions.exists() else workspace.pilot_incoming

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
            baseline = ((self.calibration_artifact.get("stereo") or {}).get("baseline"))
            units = ((self.calibration_artifact.get("board") or {}).get("units") or "")
            self.baseline_lbl.setText(self._format_distance(float(baseline), units) if baseline is not None else "-")
        else:
            self.calib_size_lbl.setText("-")
            self.baseline_lbl.setText("-")
        self._set_board_check_label(None)

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
                str(frame.get("source_session", "")),
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.pairs_table.setItem(row, col, item)
        self.pairs_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)

    def _show_pair(self, row: int, *, force: bool = False) -> None:
        if row < 0 or row >= len(self.image_pairs):
            return
        self._clear_measurement()
        left_path, right_path = self.image_pairs[row]
        left_image = cv2.imread(str(left_path), cv2.IMREAD_COLOR)
        right_image = cv2.imread(str(right_path), cv2.IMREAD_COLOR)
        if left_image is None or right_image is None:
            self.summary_lbl.setText("One or both stereo images could not be read.")
            return

        if self.rectification_maps is None:
            self.left_canvas.set_frame(left_image, placeholder="Left image missing")
            self.right_canvas.set_frame(right_image, placeholder="Right image missing")
            self.disparity_preview.clear("Load calibration")
            self.depth_preview.clear("Load calibration")
            self.summary_lbl.setText("Calibration not loaded.")
            return

        try:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            board_check = None
            if self.calibration_artifact is not None:
                board_check = analyze_charuco_stereo_geometry(left_image, right_image, self.calibration_artifact)
                self.geometry_check = board_check
                self._set_board_check_label(board_check)
                self._fit_disparity_range_to_board_check(board_check)
            left_rect, right_rect = rectify_stereo_images(left_image, right_image, self.rectification_maps)
            self.left_canvas.set_frame(left_rect)
            self.right_canvas.set_frame(right_rect)
            disparity, disparity_valid = compute_disparity(
                left_rect,
                right_rect,
                min_disparity=self.min_disparity_spin.value(),
                num_disparities=self.num_disparities_spin.value(),
                block_size=self.block_size_spin.value(),
                uniqueness_ratio=self.uniqueness_spin.value(),
                speckle_window_size=self.speckle_window_spin.value(),
                speckle_range=self.speckle_range_spin.value(),
                preprocess="clahe" if self.clahe_check.isChecked() else "none",
                left_right_check=self.lr_check.isChecked(),
            )
            max_depth = float(self.max_depth_spin.value()) or None
            points_3d, valid_depth = reproject_disparity(
                disparity,
                self.rectification_maps.q,
                disparity_valid,
                max_abs_depth=max_depth,
            )
            self.disparity = disparity
            self.points_3d = points_3d
            self.valid_depth = valid_depth
            self.disparity_preview.set_frame(colorize_disparity(disparity, disparity_valid))
            self.depth_preview.set_frame(colorize_depth(points_3d, valid_depth))
            self._set_depth_summary(row, valid_depth, points_3d, board_check)
        except Exception as exc:
            self.disparity = None
            self.points_3d = None
            self.valid_depth = None
            self.disparity_preview.clear("Disparity failed")
            self.depth_preview.clear("Depth failed")
            self.summary_lbl.setText(f"Depth failed: {exc}")
            self.statusBar().showMessage(f"Depth failed: {exc}", 7000)
        finally:
            QApplication.restoreOverrideCursor()

    def _fit_disparity_range_to_board_check(self, board_check: dict | None) -> None:
        if not board_check or not board_check.get("available"):
            return
        disparity_min = float(board_check.get("disparity_min") or 0.0)
        disparity_max = float(board_check.get("disparity_max") or 0.0)
        margin = 48.0
        target_min = int(np.floor(disparity_min - margin))
        target_max = int(np.ceil(disparity_max + margin))
        current_min = self.min_disparity_spin.value()
        current_max = current_min + self.num_disparities_spin.value() - 1
        if target_min < current_min:
            self.min_disparity_spin.setValue(max(self.min_disparity_spin.minimum(), target_min))
            current_min = self.min_disparity_spin.value()
        if target_max > current_max:
            needed = normalized_num_disparities(target_max - current_min + 1)
            self.num_disparities_spin.setValue(min(self.num_disparities_spin.maximum(), needed))

    def _set_depth_summary(
        self,
        row: int,
        valid_depth: np.ndarray,
        points_3d: np.ndarray,
        board_check: dict | None,
    ) -> None:
        valid_count = int(np.count_nonzero(valid_depth))
        total = int(valid_depth.size)
        coverage = valid_count / max(1, total)
        units = self._units()
        depth_values = np.abs(points_3d[:, :, 2][valid_depth])
        median_depth = float(np.median(depth_values)) if depth_values.size else None
        depth_text = self._format_distance(median_depth, units) if median_depth is not None else "-"
        board_text = ""
        if board_check and board_check.get("available"):
            board_text = " | board disp {disp:.1f}px, edge {edge}".format(
                disp=float(board_check.get("disparity_median") or 0.0),
                edge=self._format_distance(board_check.get("edge_median"), units),
            )
        self.summary_lbl.setText(
            "Pair {pair}: valid depth {coverage:.0%} | median depth {depth}{board}".format(
                pair=row + 1,
                coverage=coverage,
                depth=depth_text,
                board=board_text,
            )
        )

    def _set_board_check_label(self, board_check: dict | None) -> None:
        if not board_check:
            self.board_check_lbl.setText("-")
            return
        if not board_check.get("available"):
            self.board_check_lbl.setText(str(board_check.get("reason") or "-"))
            return
        units = self._units() or str(board_check.get("units") or "")
        edge = self._format_distance(board_check.get("edge_median"), units)
        self.board_check_lbl.setText(
            "{count} corners | disp {dmin:.1f}-{dmax:.1f}px | y RMS {yrms:.2f}px | edge {edge}".format(
                count=int(board_check.get("matched_count") or 0),
                dmin=float(board_check.get("disparity_min") or 0.0),
                dmax=float(board_check.get("disparity_max") or 0.0),
                yrms=float(board_check.get("vertical_rms_px") or 0.0),
                edge=edge,
            )
        )

    def _on_left_measurement_click(self, x: float, y: float) -> None:
        self._append_measurement_point(self.left_measure_points, x, y)
        self._rebuild_measurement_samples()

    def _on_right_measurement_click(self, x: float, y: float) -> None:
        self._append_measurement_point(self.right_measure_points, x, y)
        self._rebuild_measurement_samples()

    def _append_measurement_point(self, points: list[tuple[int, int]], x: float, y: float) -> None:
        if len(points) >= 2 or (len(self.left_measure_points) >= 2 and len(self.right_measure_points) >= 2):
            self._clear_measurement()
        points.append((int(round(float(x))), int(round(float(y)))))

    def _rebuild_measurement_samples(self) -> None:
        self.samples.clear()
        error_text = ""
        if self.right_measure_points:
            if self.rectification_maps is None:
                error_text = "Load calibration before measuring."
            else:
                count = min(len(self.left_measure_points), len(self.right_measure_points), 2)
                for idx in range(count):
                    try:
                        self.samples.append(
                            point_from_rectified_correspondence(
                                self.rectification_maps.q,
                                self.left_measure_points[idx],
                                self.right_measure_points[idx],
                            )
                        )
                    except Exception as exc:
                        error_text = str(exc)
                        break
        elif self.left_measure_points:
            if self.points_3d is None or self.disparity is None or self.valid_depth is None:
                error_text = "Compute depth before measuring."
            else:
                for point in self.left_measure_points[:2]:
                    try:
                        self.samples.append(
                            sample_depth_point(
                                self.points_3d,
                                self.disparity,
                                self.valid_depth,
                                point,
                                radius=self.sample_radius_spin.value(),
                            )
                        )
                    except Exception as exc:
                        error_text = str(exc)
                        break
        if error_text:
            self.statusBar().showMessage(error_text, 5000)
        self._update_measurement_labels(error_text)

    def _clear_measurement(self) -> None:
        self.samples.clear()
        self.left_measure_points.clear()
        self.right_measure_points.clear()
        self.left_canvas.set_measurement([])
        self.right_canvas.set_measurement([])
        self.point1_lbl.setText("-")
        self.point2_lbl.setText("-")
        self.distance_lbl.setText("-")

    def _update_measurement_labels(self, error_text: str = "") -> None:
        units = self._units()
        labels = [self.point1_lbl, self.point2_lbl]
        for idx, label in enumerate(labels):
            if idx < len(self.samples):
                label.setText(self._format_sample(self.samples[idx], units))
            elif idx < len(self.left_measure_points) or idx < len(self.right_measure_points):
                label.setText(self._format_pending_point(idx))
            else:
                label.setText("-")
        badge = ""
        if len(self.samples) == 2:
            distance = distance_between_samples(self.samples[0], self.samples[1])
            badge = self._format_distance(distance, units)
            self.distance_lbl.setText(badge)
            self.statusBar().showMessage(f"Stereo measurement: {badge}", 5000)
        else:
            self.distance_lbl.setText(error_text or "-")
        self.left_canvas.set_measurement(self.left_measure_points, badge)
        self.right_canvas.set_measurement(self.right_measure_points, badge)

    def _format_pending_point(self, index: int) -> str:
        parts = []
        if index < len(self.left_measure_points):
            left = self.left_measure_points[index]
            parts.append(f"L ({left[0]}, {left[1]})")
        if index < len(self.right_measure_points):
            right = self.right_measure_points[index]
            parts.append(f"R ({right[0]}, {right[1]})")
        return " | ".join(parts) if parts else "-"

    def _units(self) -> str:
        if self.calibration_artifact is None:
            return ""
        return str((self.calibration_artifact.get("board") or {}).get("units") or "")

    def _default_max_depth_for_units(self, units: str) -> float:
        unit = str(units or "").strip().lower()
        if unit == "mm":
            return 20000.0
        if unit == "cm":
            return 2000.0
        if unit in {"m", "meter", "meters"}:
            return 20.0
        return 0.0

    def _format_sample(self, sample: DepthSample | CorrespondenceSample, units: str) -> str:
        point = sample.point
        if isinstance(sample, CorrespondenceSample):
            return (
                f"L ({sample.left_pixel[0]}, {sample.left_pixel[1]}) | "
                f"R ({sample.right_pixel[0]}, {sample.right_pixel[1]}) | "
                f"d {sample.disparity:.1f}px | yerr {abs(sample.vertical_error_px):.1f}px | "
                f"X {point[0]:.1f}, Y {point[1]:.1f}, Z {point[2]:.1f} {units}"
            ).strip()
        return (
            f"({sample.pixel[0]}, {sample.pixel[1]}) | "
            f"X {point[0]:.1f}, Y {point[1]:.1f}, Z {point[2]:.1f} {units}".strip()
        )

    def _format_distance(self, value: float | None, units: str) -> str:
        if value is None:
            return "-"
        units = str(units or "").strip()
        if units.lower() == "mm":
            return f"{value:.1f} mm ({value / 10.0:.2f} cm)"
        if units.lower() == "cm":
            return f"{value:.2f} cm"
        return f"{value:.2f} {units}".strip()
