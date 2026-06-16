"""GUI for annotating crab-board planes and generating YOLO data."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QObject, QPointF, QRectF, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from triton_analysis.crab.plane_dataset import (
    BoardPlaneAnnotation,
    PlaneProjectedDatasetConfig,
    discover_board_images,
    discover_default_crab_template_paths,
    generate_plane_projected_dataset,
    load_board_plane_annotations,
    save_board_plane_annotations,
)
from triton_analysis.crab.synthetic import CRAB_CLASS_NAMES
from triton_analysis.gui.file_dialogs import ThumbnailFileDialog as QFileDialog
from triton_analysis.gui.responsive import resize_to_available_screen
from triton_analysis.workspace import fresh_output_subdir, workspace_paths


class BoardPlaneCanvas(QWidget):
    """Interactive image canvas for a four-corner board plane."""

    pointsChanged = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap = QPixmap()
        self._image_path: Path | None = None
        self._points: list[tuple[float, float]] = []
        self._drag_index: int | None = None
        self.setMinimumSize(300, 280)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)

    @property
    def image_size(self) -> tuple[int, int] | None:
        if self._pixmap.isNull():
            return None
        return self._pixmap.width(), self._pixmap.height()

    def set_image(self, path: str | Path | None) -> None:
        self._image_path = Path(path).expanduser() if path else None
        self._points = []
        self._drag_index = None
        self._pixmap = QPixmap(str(self._image_path)) if self._image_path else QPixmap()
        self.update()
        self.pointsChanged.emit(self.points())

    def set_points(self, points: list[tuple[float, float]] | tuple[tuple[float, float], ...]) -> None:
        self._points = [(float(x), float(y)) for x, y in list(points)[:4]]
        self._drag_index = None
        self.update()
        self.pointsChanged.emit(self.points())

    def points(self) -> list[tuple[float, float]]:
        return list(self._points)

    def clear_points(self) -> None:
        self.set_points([])

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(15, 15, 19))
        if self._pixmap.isNull():
            painter.setPen(QColor(180, 184, 196))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No image selected")
            return

        image_rect = self._image_rect()
        painter.drawPixmap(image_rect, self._pixmap, QRectF(self._pixmap.rect()))
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        widget_points = [self._image_to_widget(point) for point in self._points]
        if len(widget_points) >= 2:
            pen = QPen(QColor(80, 190, 255), 2.0)
            painter.setPen(pen)
            for index in range(len(widget_points) - 1):
                painter.drawLine(widget_points[index], widget_points[index + 1])
            if len(widget_points) == 4:
                painter.drawLine(widget_points[-1], widget_points[0])
                painter.setBrush(QColor(80, 190, 255, 34))
                painter.drawPolygon(*widget_points)

        for index, point in enumerate(widget_points, start=1):
            painter.setPen(QPen(QColor(8, 20, 28), 2.0))
            painter.setBrush(QColor(255, 205, 80))
            painter.drawEllipse(point, 7.5, 7.5)
            painter.setPen(QColor(10, 14, 18))
            painter.drawText(QRectF(point.x() - 8, point.y() - 23, 16, 16), Qt.AlignmentFlag.AlignCenter, str(index))

    def mousePressEvent(self, event) -> None:
        if self._pixmap.isNull() or event.button() not in (
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.RightButton,
        ):
            return
        image_point = self._widget_to_image(QPointF(event.position()))
        if image_point is None:
            return

        nearest_index = self._nearest_point_index(QPointF(event.position()), max_distance=18.0)
        if event.button() == Qt.MouseButton.RightButton:
            if nearest_index is not None:
                self._points.pop(nearest_index)
            elif self._points:
                self._points.pop()
            self.update()
            self.pointsChanged.emit(self.points())
            return

        if nearest_index is not None:
            self._drag_index = nearest_index
        elif len(self._points) < 4:
            self._points.append(image_point)
            self._drag_index = len(self._points) - 1
        self.update()
        self.pointsChanged.emit(self.points())

    def mouseMoveEvent(self, event) -> None:
        if self._drag_index is None:
            return
        image_point = self._widget_to_image(QPointF(event.position()), clamp=True)
        if image_point is None:
            return
        self._points[self._drag_index] = image_point
        self.update()
        self.pointsChanged.emit(self.points())

    def mouseReleaseEvent(self, event) -> None:
        self._drag_index = None
        super().mouseReleaseEvent(event)

    def _image_rect(self) -> QRectF:
        if self._pixmap.isNull():
            return QRectF()
        margin = 10.0
        available = QRectF(self.rect()).adjusted(margin, margin, -margin, -margin)
        scale = min(
            available.width() / max(1, self._pixmap.width()),
            available.height() / max(1, self._pixmap.height()),
        )
        width = self._pixmap.width() * scale
        height = self._pixmap.height() * scale
        return QRectF(
            available.left() + (available.width() - width) * 0.5,
            available.top() + (available.height() - height) * 0.5,
            width,
            height,
        )

    def _widget_to_image(self, point: QPointF, *, clamp: bool = False) -> tuple[float, float] | None:
        image_rect = self._image_rect()
        if image_rect.isNull() or self._pixmap.isNull():
            return None
        if not clamp and not image_rect.contains(point):
            return None
        x = (point.x() - image_rect.left()) / max(1.0, image_rect.width()) * self._pixmap.width()
        y = (point.y() - image_rect.top()) / max(1.0, image_rect.height()) * self._pixmap.height()
        if clamp:
            x = max(0.0, min(float(self._pixmap.width() - 1), x))
            y = max(0.0, min(float(self._pixmap.height() - 1), y))
        return float(x), float(y)

    def _image_to_widget(self, point: tuple[float, float]) -> QPointF:
        image_rect = self._image_rect()
        x = image_rect.left() + float(point[0]) / max(1, self._pixmap.width()) * image_rect.width()
        y = image_rect.top() + float(point[1]) / max(1, self._pixmap.height()) * image_rect.height()
        return QPointF(x, y)

    def _nearest_point_index(self, point: QPointF, *, max_distance: float) -> int | None:
        best_index: int | None = None
        best_distance = float(max_distance)
        for index, image_point in enumerate(self._points):
            widget_point = self._image_to_widget(image_point)
            dx = widget_point.x() - point.x()
            dy = widget_point.y() - point.y()
            distance = (dx * dx + dy * dy) ** 0.5
            if distance <= best_distance:
                best_index = index
                best_distance = distance
        return best_index


class CrabDatasetGenerationWorker(QObject):
    """Background worker for plane-projected dataset generation."""

    progress = pyqtSignal(object)
    finished = pyqtSignal(object)

    def __init__(self, config: PlaneProjectedDatasetConfig):
        super().__init__()
        self._config = config

    def run(self) -> None:
        try:
            result = generate_plane_projected_dataset(self._config, progress_callback=self.progress.emit)
        except Exception as exc:  # pragma: no cover - surfaced in GUI and covered by integration tests
            self.finished.emit({"ok": False, "error": str(exc)})
            return
        self.finished.emit(
            {
                "ok": True,
                "output_dir": str(result.output_dir),
                "data_yaml": str(result.data_yaml),
                "preview_image": str(result.preview_image) if result.preview_image else "",
                "train_images": result.train_images,
                "val_images": result.val_images,
                "class_counts": result.class_counts,
            }
        )


class CrabDatasetGeneratorWindow(QMainWindow):
    """Annotate empty boards and generate crab detector datasets."""

    def __init__(self, *, workspace_root: str | Path | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Crab Dataset Generator")
        self._workspace = workspace_paths(workspace_root, create=True)
        self._image_dir = self._workspace.root / "data" / "base images"
        self._annotation_path = self._workspace.root / "data" / "board_plane_annotations.json"
        self._images: list[Path] = []
        self._annotations: dict[Path, BoardPlaneAnnotation] = {}
        self._template_paths = discover_default_crab_template_paths(self._workspace.root)
        self._generation_thread: QThread | None = None
        self._generation_worker: CrabDatasetGenerationWorker | None = None

        self.canvas = BoardPlaneCanvas()
        self.canvas.pointsChanged.connect(self._handle_canvas_points_changed)
        self.image_list = QListWidget()
        self.image_list.currentItemChanged.connect(self._handle_image_selection)
        self.image_dir_edit = QLineEdit(str(self._image_dir))
        self.annotation_path_edit = QLineEdit(str(self._annotation_path))
        self.template_label = QLabel()
        self.template_label.setWordWrap(True)
        self.plane_status_label = QLabel()
        self.plane_status_label.setWordWrap(True)
        self.generate_status_label = QLabel()
        self.generate_status_label.setWordWrap(True)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)

        self.count_spin = self._spin(1, 200000, 2000, step=100)
        self.seed_spin = self._spin(0, 2147483647, 60929, step=1)
        self.board_size_spin = self._spin(256, 2400, 768, step=64)
        self.min_crabs_spin = self._spin(0, 40, 4, step=1)
        self.max_crabs_spin = self._spin(0, 60, 11, step=1)
        self.min_box_spin = self._spin(0, 200, 32, step=4)
        self.val_fraction_spin = self._double_spin(0.0, 0.8, 0.2, step=0.05)
        self.empty_fraction_spin = self._double_spin(0.0, 0.5, 0.0, step=0.02)
        self.green_fraction_spin = self._double_spin(0.0, 1.0, 0.7, step=0.05)
        self.crab_color_jitter_spin = self._double_spin(0.0, 1.0, 0.35, step=0.05)
        self.paper_alpha_spin = self._double_spin(0.0, 1.0, 0.0, step=0.05)

        self.generate_btn = QPushButton("Generate Dataset")
        self.generate_btn.clicked.connect(self._start_generation)

        central = QWidget()
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(8, 8, 8, 8)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self.canvas)
        splitter.addWidget(self._build_right_panel())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([320, 900, 360])
        root_layout.addWidget(splitter, 1)
        self.setCentralWidget(central)
        self.statusBar().showMessage("Crab dataset generator ready.")

        self._load_annotations()
        self._reload_images()
        self._update_template_label()
        resize_to_available_screen(self, 1640, 940, min_width=1120, min_height=700)

    def shutdown(self) -> None:
        thread = self._generation_thread
        if thread is not None and thread.isRunning():
            thread.quit()
            thread.wait(1500)

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        panel.setMinimumWidth(210)
        panel.setMaximumWidth(360)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        dir_row = QHBoxLayout()
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self._choose_image_dir)
        reload_btn = QPushButton("Reload")
        reload_btn.clicked.connect(self._reload_images)
        dir_row.addWidget(browse_btn)
        dir_row.addWidget(reload_btn)

        layout.addWidget(QLabel("Board Images"))
        layout.addWidget(self.image_dir_edit)
        layout.addLayout(dir_row)
        layout.addWidget(self.image_list, 1)
        layout.addWidget(self.plane_status_label)
        return panel

    def _build_right_panel(self) -> QScrollArea:
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(8, 0, 0, 0)
        layout.setSpacing(10)

        save_btn = QPushButton("Save Plane")
        save_btn.clicked.connect(self._save_current_plane)
        clear_btn = QPushButton("Clear Plane")
        clear_btn.clicked.connect(self._clear_current_plane)
        prev_btn = QPushButton("Previous")
        prev_btn.clicked.connect(lambda: self._step_image(-1))
        next_btn = QPushButton("Next")
        next_btn.clicked.connect(lambda: self._step_image(1))
        nav_row = QHBoxLayout()
        nav_row.addWidget(prev_btn)
        nav_row.addWidget(next_btn)
        plane_row = QHBoxLayout()
        plane_row.addWidget(save_btn)
        plane_row.addWidget(clear_btn)

        layout.addWidget(QLabel("Annotation File"))
        layout.addWidget(self.annotation_path_edit)
        layout.addLayout(nav_row)
        layout.addLayout(plane_row)
        layout.addWidget(self.template_label)

        refresh_templates_btn = QPushButton("Refresh Crab Sources")
        refresh_templates_btn.clicked.connect(self._refresh_templates)
        layout.addWidget(refresh_templates_btn)

        form = QFormLayout()
        form.addRow("Images", self.count_spin)
        form.addRow("Validation", self.val_fraction_spin)
        form.addRow("Seed", self.seed_spin)
        form.addRow("Plane size", self.board_size_spin)
        form.addRow("Min crabs", self.min_crabs_spin)
        form.addRow("Max crabs", self.max_crabs_spin)
        form.addRow("Min box", self.min_box_spin)
        form.addRow("Empty boards", self.empty_fraction_spin)
        form.addRow("Green-positive", self.green_fraction_spin)
        form.addRow("Crab color jitter", self.crab_color_jitter_spin)
        form.addRow("Paper edge", self.paper_alpha_spin)
        layout.addLayout(form)
        layout.addWidget(self.generate_btn)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.generate_status_label)
        layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidget(content)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(230)
        scroll.setMaximumWidth(390)
        return scroll

    @staticmethod
    def _spin(minimum: int, maximum: int, value: int, *, step: int) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setSingleStep(step)
        spin.setValue(value)
        return spin

    @staticmethod
    def _double_spin(minimum: float, maximum: float, value: float, *, step: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setSingleStep(step)
        spin.setDecimals(2)
        spin.setValue(value)
        return spin

    def _annotation_key(self, path: str | Path) -> Path:
        candidate = Path(path).expanduser()
        try:
            return candidate.resolve()
        except OSError:
            return candidate

    def _load_annotations(self) -> None:
        self._annotation_path = Path(self.annotation_path_edit.text()).expanduser()
        rows = load_board_plane_annotations(self._annotation_path, image_root=self._workspace.root / "data")
        self._annotations = {self._annotation_key(row.image_path): row for row in rows}

    def _save_annotations(self) -> None:
        self._annotation_path = Path(self.annotation_path_edit.text()).expanduser()
        rows = sorted(self._annotations.values(), key=lambda row: str(row.image_path).lower())
        save_board_plane_annotations(self._annotation_path, rows, image_root=self._workspace.root / "data")
        self._refresh_image_marks()

    def _reload_images(self) -> None:
        self._image_dir = Path(self.image_dir_edit.text()).expanduser()
        self._load_annotations()
        selected = self._current_image_path()
        self._images = discover_board_images(self._image_dir)
        self.image_list.blockSignals(True)
        self.image_list.clear()
        for path in self._images:
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, str(path))
            self.image_list.addItem(item)
        self.image_list.blockSignals(False)
        self._refresh_image_marks()
        if not self._images:
            self.canvas.set_image(None)
            self.plane_status_label.setText("No board images found.")
            return
        target_index = 0
        if selected is not None:
            selected_key = self._annotation_key(selected)
            for index, path in enumerate(self._images):
                if self._annotation_key(path) == selected_key:
                    target_index = index
                    break
        self.image_list.setCurrentRow(target_index)

    def _refresh_image_marks(self) -> None:
        current = self.image_list.currentRow()
        for index in range(self.image_list.count()):
            item = self.image_list.item(index)
            path = Path(str(item.data(Qt.ItemDataRole.UserRole)))
            mark = "[x]" if self._annotation_key(path) in self._annotations else "[ ]"
            item.setText(f"{mark} {path.name}")
            item.setToolTip(str(path))
        if current >= 0:
            self.image_list.setCurrentRow(current)
        self._update_plane_status()

    def _handle_image_selection(self, current: QListWidgetItem | None, previous: QListWidgetItem | None) -> None:
        if previous is not None:
            self._store_canvas_annotation(Path(str(previous.data(Qt.ItemDataRole.UserRole))))
        if current is None:
            self.canvas.set_image(None)
            return
        path = Path(str(current.data(Qt.ItemDataRole.UserRole)))
        self.canvas.set_image(path)
        annotation = self._annotations.get(self._annotation_key(path))
        if annotation is not None:
            self.canvas.set_points(annotation.quad_xy)
        self._update_plane_status()

    def _current_image_path(self) -> Path | None:
        item = self.image_list.currentItem()
        if item is None:
            return None
        return Path(str(item.data(Qt.ItemDataRole.UserRole)))

    def _handle_canvas_points_changed(self, _points: object) -> None:
        path = self._current_image_path()
        if path is not None:
            self._store_canvas_annotation(path)
        self._update_plane_status()

    def _store_canvas_annotation(self, path: Path) -> None:
        points = self.canvas.points()
        if len(points) != 4:
            return
        image_size = self.canvas.image_size
        self._annotations[self._annotation_key(path)] = BoardPlaneAnnotation(
            image_path=path,
            quad_xy=tuple(points),  # type: ignore[arg-type]
            image_size=image_size,
            label=path.name,
        )
        self._refresh_image_marks()

    def _save_current_plane(self) -> None:
        path = self._current_image_path()
        if path is None:
            self.statusBar().showMessage("Select a board image first.", 4000)
            return
        self._store_canvas_annotation(path)
        if len(self.canvas.points()) != 4:
            self.statusBar().showMessage("Mark four board corners before saving.", 5000)
            return
        self._save_annotations()
        self.statusBar().showMessage(f"Saved board plane: {path.name}", 5000)

    def _clear_current_plane(self) -> None:
        path = self._current_image_path()
        if path is not None:
            self._annotations.pop(self._annotation_key(path), None)
        self.canvas.clear_points()
        self._save_annotations()
        self.statusBar().showMessage("Cleared board plane.", 4000)

    def _choose_image_dir(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, "Choose empty-board image folder", str(self._image_dir))
        if not selected:
            return
        self.image_dir_edit.setText(selected)
        self._reload_images()

    def _step_image(self, delta: int) -> None:
        if self.image_list.count() <= 0:
            return
        row = self.image_list.currentRow()
        self.image_list.setCurrentRow(max(0, min(self.image_list.count() - 1, row + delta)))

    def _refresh_templates(self) -> None:
        self._template_paths = discover_default_crab_template_paths(self._workspace.root)
        self._update_template_label()

    def _update_template_label(self) -> None:
        pieces = []
        for class_name in CRAB_CLASS_NAMES:
            count = len(self._template_paths.get(class_name, []))
            pieces.append(f"{class_name.replace('_', ' ')}: {count}")
        self.template_label.setText("Crab sources: " + " | ".join(pieces))

    def _update_plane_status(self) -> None:
        current = self._current_image_path()
        saved = len(self._annotations)
        total = len(self._images)
        points = len(self.canvas.points())
        if current is None:
            self.plane_status_label.setText(f"Saved planes: {saved}/{total}")
        else:
            self.plane_status_label.setText(f"Saved planes: {saved}/{total} | Current corners: {points}/4")

    def _start_generation(self) -> None:
        if self._generation_thread is not None:
            return
        current = self._current_image_path()
        if current is not None:
            self._store_canvas_annotation(current)
        self._save_annotations()
        annotations = sorted(self._annotations.values(), key=lambda row: str(row.image_path).lower())
        if not annotations:
            self.statusBar().showMessage("Save at least one board plane before generating.", 6000)
            return
        self._refresh_templates()
        missing = [name for name in CRAB_CLASS_NAMES if not self._template_paths.get(name)]
        if missing:
            self.statusBar().showMessage("Missing crab sources: " + ", ".join(missing), 8000)
            return

        output_dir = fresh_output_subdir(self._workspace.root / "datasets", "crab_plane_synth", create=True)
        config = PlaneProjectedDatasetConfig(
            output_dir=output_dir,
            annotations=annotations,
            template_paths=self._template_paths,
            image_count=int(self.count_spin.value()),
            val_fraction=float(self.val_fraction_spin.value()),
            seed=int(self.seed_spin.value()),
            board_size=int(self.board_size_spin.value()),
            min_crabs=int(self.min_crabs_spin.value()),
            max_crabs=max(int(self.min_crabs_spin.value()), int(self.max_crabs_spin.value())),
            min_crab_box_long_edge_px=int(self.min_box_spin.value()),
            empty_fraction=float(self.empty_fraction_spin.value()),
            green_positive_fraction=float(self.green_fraction_spin.value()),
            crab_color_jitter_strength=float(self.crab_color_jitter_spin.value()),
            paper_alpha_scale=float(self.paper_alpha_spin.value()),
        )
        self.progress_bar.setValue(0)
        self.generate_status_label.setText(f"Writing {output_dir}")
        self.generate_btn.setEnabled(False)

        worker = CrabDatasetGenerationWorker(config)
        thread = QThread(self)
        self._generation_worker = worker
        self._generation_thread = thread
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._handle_generation_progress)
        worker.finished.connect(self._finish_generation)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(lambda: self._clear_generation_thread(thread))
        thread.start()

    def _handle_generation_progress(self, payload: object) -> None:
        data = payload if isinstance(payload, dict) else {}
        total = max(1, int(data.get("total") or 1))
        index = max(0, int(data.get("index") or 0))
        self.progress_bar.setValue(int(index / total * 100))
        source = Path(str(data.get("source") or "")).name
        self.generate_status_label.setText(f"Generated {index}/{total} from {source}")

    def _finish_generation(self, payload: object) -> None:
        self.generate_btn.setEnabled(True)
        data = payload if isinstance(payload, dict) else {}
        if not data.get("ok"):
            message = str(data.get("error") or "dataset generation failed")
            self.generate_status_label.setText(message)
            self.statusBar().showMessage(message, 8000)
            return
        self.progress_bar.setValue(100)
        output_dir = str(data.get("output_dir") or "")
        train_images = int(data.get("train_images") or 0)
        val_images = int(data.get("val_images") or 0)
        self.generate_status_label.setText(f"Dataset ready: {output_dir}")
        self.statusBar().showMessage(f"Generated crab dataset: train={train_images} val={val_images}", 8000)

    def _clear_generation_thread(self, thread: QThread) -> None:
        if self._generation_thread is thread:
            self._generation_thread = None
            self._generation_worker = None
