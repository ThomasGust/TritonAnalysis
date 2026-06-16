"""GUI for OpenAI-assisted European green crab counting."""

from __future__ import annotations

import os
from pathlib import Path

from PyQt6.QtCore import QObject, QRectF, Qt, QThread, QUrl, pyqtSignal
from PyQt6.QtGui import QColor, QDesktopServices, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from triton_analysis.crab.counter import (
    DEFAULT_MODEL,
    REFERENCE_CLASS_LABELS,
    CrabCountResult,
    CrabCounterConfig,
    CrabCounterOutputs,
    analyze_crab_image,
    default_output_dir,
    discover_counter_reference_paths,
    missing_reference_classes,
)
from triton_analysis.crab.synthetic import CRAB_CLASS_NAMES, IMAGE_EXTENSIONS
from triton_analysis.gui.file_dialogs import ThumbnailFileDialog as QFileDialog
from triton_analysis.gui.responsive import resize_to_available_screen
from triton_analysis.workspace import fresh_output_subdir, latest_pilot_run_dir, workspace_paths


class CrabCounterPreview(QWidget):
    """Scaled image preview with result boxes painted on top."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap = QPixmap()
        self._image_path: Path | None = None
        self._result: CrabCountResult | None = None
        self.setMinimumSize(360, 280)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_image(self, path: str | Path | None) -> None:
        self._image_path = Path(path).expanduser() if path else None
        self._pixmap = QPixmap(str(self._image_path)) if self._image_path else QPixmap()
        self._result = None
        self.update()

    def set_result(self, result: CrabCountResult | None, annotated_image: str | Path | None = None) -> None:
        self._result = result
        if annotated_image:
            self._image_path = Path(annotated_image).expanduser()
            self._pixmap = QPixmap(str(self._image_path))
        self.update()

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(15, 15, 19))
        if self._pixmap.isNull():
            painter.setPen(QColor(180, 184, 196))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No target image selected")
            return

        image_rect = self._image_rect()
        painter.drawPixmap(image_rect, self._pixmap, QRectF(self._pixmap.rect()))
        if self._result is None:
            return
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        scale_x = image_rect.width() / max(1, self._result.image_size[0])
        scale_y = image_rect.height() / max(1, self._result.image_size[1])
        pen = QPen(QColor(0, 255, 90), 2.5)
        painter.setPen(pen)
        for detection in self._result.detections:
            x0, y0, x1, y1 = detection.bbox
            rect = QRectF(
                image_rect.left() + x0 * scale_x,
                image_rect.top() + y0 * scale_y,
                max(1.0, (x1 - x0) * scale_x),
                max(1.0, (y1 - y0) * scale_y),
            )
            painter.drawRect(rect)

    def _image_rect(self) -> QRectF:
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


class CrabCounterWorker(QObject):
    """Run one crab-counter request off the UI thread."""

    finished = pyqtSignal(object)

    def __init__(self, config: CrabCounterConfig):
        super().__init__()
        self._config = config

    def run(self) -> None:
        try:
            outputs = analyze_crab_image(self._config)
        except Exception as exc:  # pragma: no cover - surfaced in GUI
            self.finished.emit({"ok": False, "error": str(exc)})
            return
        self.finished.emit({"ok": True, "outputs": outputs})


class CrabCounterWindow(QMainWindow):
    """Count European green crabs on a saved MATE board image."""

    def __init__(
        self,
        *,
        image_path: str | Path | None = None,
        workspace_root: str | Path | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Crab Counter")
        self._workspace = workspace_paths(workspace_root, create=True)
        self._analysis_thread: QThread | None = None
        self._analysis_worker: CrabCounterWorker | None = None
        self._last_outputs: CrabCounterOutputs | None = None

        self.preview = CrabCounterPreview()
        self.target_edit = QLineEdit(str(Path(image_path).expanduser()) if image_path else "")
        self.target_edit.editingFinished.connect(self._load_target_preview)
        self.model_edit = QLineEdit(DEFAULT_MODEL)
        self.output_root_edit = QLineEdit(str(self._workspace.results / "crab_counter"))
        self.status_label = QLabel("Set OPENAI_API_KEY, choose a target image, then analyze.")
        self.status_label.setWordWrap(True)
        self.count_label = QLabel("Count: -")
        self.count_label.setObjectName("crabCounterCount")
        self.count_label.setStyleSheet("QLabel#crabCounterCount { font-size: 22px; font-weight: 700; }")
        self.reference_edits: dict[str, QLineEdit] = {}
        self.detection_list = QListWidget()
        self.analyze_btn = QPushButton("Analyze")
        self.analyze_btn.clicked.connect(self._start_analysis)
        self.open_output_btn = QPushButton("Open Output")
        self.open_output_btn.clicked.connect(self._open_output_folder)
        self.open_output_btn.setEnabled(False)
        self.open_annotated_btn = QPushButton("Open Annotated")
        self.open_annotated_btn.clicked.connect(self._open_annotated_image)
        self.open_annotated_btn.setEnabled(False)

        central = QWidget()
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(8, 8, 8, 8)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_controls())
        splitter.addWidget(self.preview)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([420, 1040])
        root_layout.addWidget(splitter, 1)
        self.setCentralWidget(central)

        self._load_default_references()
        self._load_target_preview()
        self.statusBar().showMessage("Crab counter ready.")
        resize_to_available_screen(self, 1300, 840, min_width=720, min_height=620)

    def shutdown(self) -> None:
        thread = self._analysis_thread
        if thread is not None and thread.isRunning():
            thread.quit()
            thread.wait(1500)

    def _build_controls(self) -> QScrollArea:
        content = QWidget()
        content.setMinimumWidth(250)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 8, 0)
        layout.setSpacing(10)

        target_row = QHBoxLayout()
        browse_target_btn = QPushButton("Browse")
        browse_target_btn.clicked.connect(self._choose_target_image)
        latest_btn = QPushButton("Latest Pilot")
        latest_btn.clicked.connect(self._use_latest_pilot_image)
        target_row.addWidget(browse_target_btn)
        target_row.addWidget(latest_btn)

        layout.addWidget(QLabel("Target Image"))
        layout.addWidget(self.target_edit)
        layout.addLayout(target_row)

        reference_form = QFormLayout()
        for class_name in CRAB_CLASS_NAMES:
            edit = QLineEdit()
            self.reference_edits[class_name] = edit
            browse_btn = QPushButton("Browse")
            browse_btn.clicked.connect(lambda _checked=False, name=class_name: self._choose_reference_image(name))
            row = QHBoxLayout()
            row.addWidget(edit, 1)
            row.addWidget(browse_btn)
            reference_form.addRow(REFERENCE_CLASS_LABELS.get(class_name, class_name), row)
        layout.addWidget(QLabel("Reference Images"))
        layout.addLayout(reference_form)

        reference_row = QHBoxLayout()
        autofill_btn = QPushButton("Auto-fill")
        autofill_btn.clicked.connect(self._load_default_references)
        reference_row.addWidget(autofill_btn)
        reference_row.addStretch(1)
        layout.addLayout(reference_row)

        settings_form = QFormLayout()
        settings_form.addRow("Model", self.model_edit)
        settings_form.addRow("Output Root", self.output_root_edit)
        layout.addLayout(settings_form)

        action_row = QHBoxLayout()
        action_row.addWidget(self.analyze_btn)
        action_row.addWidget(self.open_output_btn)
        action_row.addWidget(self.open_annotated_btn)
        layout.addLayout(action_row)

        layout.addWidget(self.count_label)
        layout.addWidget(self.status_label)
        layout.addWidget(QLabel("Detections"))
        layout.addWidget(self.detection_list, 1)

        scroll = QScrollArea()
        scroll.setWidget(content)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(260)
        scroll.setMaximumWidth(420)
        return scroll

    def _load_default_references(self) -> None:
        references = discover_counter_reference_paths(self._workspace.root)
        for class_name, path in references.items():
            edit = self.reference_edits.get(class_name)
            if edit is not None and path is not None:
                edit.setText(str(path))
        missing = missing_reference_classes(references)
        if missing:
            self.status_label.setText("Missing reference images: " + ", ".join(REFERENCE_CLASS_LABELS[name] for name in missing))
        else:
            self.status_label.setText("Reference images loaded.")

    def _choose_target_image(self) -> None:
        start = self._target_start_dir()
        path, _filter = QFileDialog.getOpenFileName(
            self,
            "Choose crab-board image",
            str(start),
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)",
        )
        if not path:
            return
        self.target_edit.setText(path)
        self._load_target_preview()

    def _choose_reference_image(self, class_name: str) -> None:
        edit = self.reference_edits[class_name]
        start = Path(edit.text()).expanduser().parent if edit.text().strip() else self._workspace.root
        path, _filter = QFileDialog.getOpenFileName(
            self,
            f"Choose {REFERENCE_CLASS_LABELS.get(class_name, class_name)} reference",
            str(start),
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)",
        )
        if path:
            edit.setText(path)

    def _target_start_dir(self) -> Path:
        text = self.target_edit.text().strip()
        if text:
            path = Path(text).expanduser()
            return path.parent if path.is_file() else path
        return latest_pilot_run_dir(self._workspace.root, create=True)

    def _use_latest_pilot_image(self) -> None:
        root = latest_pilot_run_dir(self._workspace.root, create=True)
        image = self._latest_image_under(root)
        if image is None:
            self.status_label.setText(f"No images found under {root}.")
            return
        self.target_edit.setText(str(image))
        self._load_target_preview()

    def _latest_image_under(self, root: str | Path) -> Path | None:
        root_path = Path(root).expanduser()
        if not root_path.exists():
            return None
        candidates = [
            path
            for path in root_path.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda path: path.stat().st_mtime_ns)

    def _load_target_preview(self) -> None:
        text = self.target_edit.text().strip()
        path = Path(text).expanduser() if text else None
        if path and path.is_file():
            self.preview.set_image(path)
            self.statusBar().showMessage(f"Loaded target image: {path.name}", 4000)
        else:
            self.preview.set_image(None)

    def _start_analysis(self) -> None:
        if self._analysis_thread is not None:
            return
        if not os.environ.get("OPENAI_API_KEY"):
            self.status_label.setText("OPENAI_API_KEY is not set in this environment.")
            self.statusBar().showMessage("Set OPENAI_API_KEY before running the crab counter.", 6000)
            return
        image_path = Path(self.target_edit.text().strip()).expanduser()
        if not image_path.is_file():
            self.status_label.setText("Choose a target image first.")
            return
        references = {
            class_name: Path(edit.text().strip()).expanduser() if edit.text().strip() else None
            for class_name, edit in self.reference_edits.items()
        }
        missing = missing_reference_classes(references)
        if missing:
            self.status_label.setText("Missing reference images: " + ", ".join(REFERENCE_CLASS_LABELS[name] for name in missing))
            return

        output_root = Path(self.output_root_edit.text().strip()).expanduser()
        output_dir = fresh_output_subdir(output_root, "crab_counter", create=True)
        config = CrabCounterConfig(
            image_path=image_path,
            reference_paths={name: path for name, path in references.items() if path is not None},
            output_dir=output_dir,
            model=self.model_edit.text().strip() or DEFAULT_MODEL,
        )
        self._last_outputs = None
        self.open_output_btn.setEnabled(False)
        self.open_annotated_btn.setEnabled(False)
        self.count_label.setText("Count: ...")
        self.detection_list.clear()
        self.status_label.setText(f"Analyzing {image_path.name} with {config.model}...")
        self.analyze_btn.setEnabled(False)

        worker = CrabCounterWorker(config)
        thread = QThread(self)
        self._analysis_worker = worker
        self._analysis_thread = thread
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._finish_analysis)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(lambda: self._clear_analysis_thread(thread))
        thread.start()

    def _finish_analysis(self, payload: object) -> None:
        self.analyze_btn.setEnabled(True)
        data = payload if isinstance(payload, dict) else {}
        if not data.get("ok"):
            message = str(data.get("error") or "analysis failed")
            self.status_label.setText(message)
            self.count_label.setText("Count: -")
            self.statusBar().showMessage(message, 8000)
            return
        outputs = data.get("outputs")
        if not isinstance(outputs, CrabCounterOutputs):
            self.status_label.setText("Crab counter returned an unexpected result.")
            return
        self._last_outputs = outputs
        result = outputs.result
        self.count_label.setText(f"Count: {result.count}")
        self.status_label.setText(
            f"Wrote {outputs.annotated_image.name} and {outputs.result_json.name} under {outputs.output_dir}."
        )
        self.detection_list.clear()
        for index, detection in enumerate(result.detections, start=1):
            x0, y0, x1, y1 = detection.bbox
            self.detection_list.addItem(
                f"{index}: [{x0:.0f}, {y0:.0f}, {x1:.0f}, {y1:.0f}] confidence {detection.confidence:.2f}"
            )
        self.preview.set_result(result)
        self.open_output_btn.setEnabled(True)
        self.open_annotated_btn.setEnabled(True)
        self.statusBar().showMessage(f"European green crabs: {result.count}", 8000)

    def _clear_analysis_thread(self, thread: QThread) -> None:
        if self._analysis_thread is thread:
            self._analysis_thread = None
            self._analysis_worker = None

    def _open_output_folder(self) -> None:
        if self._last_outputs is None:
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._last_outputs.output_dir)))

    def _open_annotated_image(self) -> None:
        if self._last_outputs is None:
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._last_outputs.annotated_image)))
