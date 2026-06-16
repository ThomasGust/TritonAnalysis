"""GUI for OpenAI-assisted European green crab counting."""

from __future__ import annotations

import os
import time
from pathlib import Path

from PyQt6.QtCore import QObject, QRectF, Qt, QThread, QTimer, QUrl, pyqtSignal
from PyQt6.QtGui import QColor, QDesktopServices, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTabBar,
    QVBoxLayout,
    QWidget,
)

from triton_analysis.crab.counter import (
    DEFAULT_MODEL,
    DEFAULT_REASONING_EFFORT,
    DEFAULT_TARGET_MARGIN_THRESHOLD,
    DEFAULT_TARGET_MATCH_THRESHOLD,
    REFERENCE_CLASS_LABELS,
    REASONING_EFFORTS,
    TARGET_CLASS,
    UNCERTAIN_CLASS,
    CrabBenchmarkOutputs,
    CrabCountResult,
    CrabCounterConfig,
    CrabCounterOutputs,
    analyze_crab_image,
    benchmark_crab_image,
    default_output_dir,
    discover_counter_reference_atlas_paths,
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
        self._display_mode = "accepted"
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

    def set_display_mode(self, mode: str) -> None:
        self._display_mode = "all" if mode == "all" else "accepted"
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
        detections = self._result.candidates if self._display_mode == "all" else self._result.detections
        for detection in detections:
            x0, y0, x1, y1 = detection.bbox
            rect = QRectF(
                image_rect.left() + x0 * scale_x,
                image_rect.top() + y0 * scale_y,
                max(1.0, (x1 - x0) * scale_x),
                max(1.0, (y1 - y0) * scale_y),
            )
            color = self._candidate_color(detection)
            painter.setPen(QPen(QColor(0, 0, 0), 5.0))
            painter.drawRect(rect)
            painter.setPen(QPen(color, 2.5))
            painter.drawRect(rect)
            label = self._candidate_label(detection)
            text_rect = QRectF(rect.left(), max(image_rect.top(), rect.top() - 18), 130, 16)
            painter.fillRect(text_rect.adjusted(-2, -1, 2, 1), QColor(0, 0, 0, 170))
            painter.setPen(color)
            painter.drawText(text_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, label)

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

    def _candidate_color(self, detection) -> QColor:
        if detection.label == TARGET_CLASS:
            if not detection.accepted_as_target:
                return QColor(245, 210, 60)
            return QColor(0, 255, 90)
        if detection.label == "native_rock_crab":
            return QColor(255, 150, 40)
        if detection.label == "jonah_crab":
            return QColor(235, 85, 85)
        return QColor(90, 175, 255)

    def _candidate_label(self, detection) -> str:
        short = {
            TARGET_CLASS: "EGC",
            "native_rock_crab": "Rock",
            "jonah_crab": "Jonah",
            UNCERTAIN_CLASS: "?",
        }.get(detection.label, detection.label)
        return f"{short} {detection.target_match_confidence:.2f}"


class CrabCounterWorker(QObject):
    """Run one crab-counter request off the UI thread."""

    progress = pyqtSignal(object)
    finished = pyqtSignal(object)

    def __init__(self, config: CrabCounterConfig, *, benchmark: bool = False):
        super().__init__()
        self._config = config
        self._benchmark = benchmark

    def run(self) -> None:
        try:
            if self._benchmark:
                outputs = benchmark_crab_image(self._config, progress_callback=self.progress.emit)
            else:
                self.progress.emit({"event": "request_started", "effort": self._config.reasoning_effort})
                outputs = analyze_crab_image(self._config)
        except Exception as exc:  # pragma: no cover - surfaced in GUI
            self.finished.emit({"ok": False, "error": str(exc)})
            return
        self.finished.emit({"ok": True, "outputs": outputs, "benchmark": self._benchmark})


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
        self._last_benchmark_outputs: CrabBenchmarkOutputs | None = None
        self._last_output_dir: Path | None = None
        self._run_started_at = 0.0
        self._run_base_status = ""
        self._run_timer = QTimer(self)
        self._run_timer.setInterval(1000)
        self._run_timer.timeout.connect(self._update_running_status)

        self.preview = CrabCounterPreview()
        self.preview_tabs = QTabBar()
        self.preview_tabs.addTab("European Green")
        self.preview_tabs.addTab("All Candidates")
        self.preview_tabs.currentChanged.connect(self._preview_tab_changed)
        self.target_edit = QLineEdit(str(Path(image_path).expanduser()) if image_path else "")
        self.target_edit.editingFinished.connect(self._load_target_preview)
        self.model_edit = QLineEdit(DEFAULT_MODEL)
        self.reasoning_effort_combo = QComboBox()
        self.reasoning_effort_combo.addItems(REASONING_EFFORTS)
        default_effort_index = self.reasoning_effort_combo.findText(DEFAULT_REASONING_EFFORT)
        self.reasoning_effort_combo.setCurrentIndex(max(0, default_effort_index))
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.5, 0.99)
        self.threshold_spin.setDecimals(2)
        self.threshold_spin.setSingleStep(0.05)
        self.threshold_spin.setValue(DEFAULT_TARGET_MATCH_THRESHOLD)
        self.threshold_spin.setToolTip("Higher values reduce false positives but can miss ambiguous European green crabs.")
        self.margin_spin = QDoubleSpinBox()
        self.margin_spin.setRange(0.0, 0.5)
        self.margin_spin.setDecimals(2)
        self.margin_spin.setSingleStep(0.05)
        self.margin_spin.setValue(DEFAULT_TARGET_MARGIN_THRESHOLD)
        self.margin_spin.setToolTip("Higher values require European green crab to beat the closest non-target by more.")
        self.output_root_edit = QLineEdit(str(self._workspace.results / "crab_counter"))
        self.status_label = QLabel("Set OPENAI_API_KEY, choose a target image, then analyze.")
        self.status_label.setWordWrap(True)
        self.count_label = QLabel("Count: -")
        self.count_label.setObjectName("crabCounterCount")
        self.count_label.setStyleSheet("QLabel#crabCounterCount { font-size: 22px; font-weight: 700; }")
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(True)
        self.reference_edits: dict[str, QLineEdit] = {}
        self.detection_list = QListWidget()
        self.analyze_btn = QPushButton("Analyze")
        self.analyze_btn.clicked.connect(self._start_analysis)
        self.benchmark_btn = QPushButton("Benchmark Effort")
        self.benchmark_btn.clicked.connect(self._start_benchmark)
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
        splitter.addWidget(self._build_preview_panel())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([420, 1040])
        root_layout.addWidget(splitter, 1)
        self.setCentralWidget(central)

        self._load_default_references()
        self._load_target_preview()
        self.statusBar().showMessage("Crab counter ready.")
        resize_to_available_screen(self, 1300, 840, min_width=720, min_height=620)

    def _build_preview_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self.preview_tabs, 0)
        layout.addWidget(self.preview, 1)
        return panel

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
        settings_form.addRow("Reasoning Effort", self.reasoning_effort_combo)
        settings_form.addRow("EGC Threshold", self.threshold_spin)
        settings_form.addRow("EGC Margin", self.margin_spin)
        settings_form.addRow("Output Root", self.output_root_edit)
        layout.addLayout(settings_form)

        action_row = QHBoxLayout()
        action_row.addWidget(self.analyze_btn)
        action_row.addWidget(self.benchmark_btn)
        action_row.addWidget(self.open_output_btn)
        action_row.addWidget(self.open_annotated_btn)
        layout.addLayout(action_row)

        layout.addWidget(self.count_label)
        layout.addWidget(self.progress_bar)
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
        self._start_run(benchmark=False)

    def _start_benchmark(self) -> None:
        self._start_run(benchmark=True)

    def _start_run(self, *, benchmark: bool) -> None:
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
        output_prefix = "crab_benchmark" if benchmark else "crab_counter"
        output_dir = fresh_output_subdir(output_root, output_prefix, create=True)
        config = CrabCounterConfig(
            image_path=image_path,
            reference_paths={name: path for name, path in references.items() if path is not None},
            output_dir=output_dir,
            model=self.model_edit.text().strip() or DEFAULT_MODEL,
            reasoning_effort=self.reasoning_effort_combo.currentText(),
            target_confidence_threshold=self.threshold_spin.value(),
            target_margin_threshold=self.margin_spin.value(),
            reference_atlas_paths=discover_counter_reference_atlas_paths(self._workspace.root),
        )
        self._last_outputs = None
        self._last_benchmark_outputs = None
        self._last_output_dir = None
        self.open_output_btn.setEnabled(False)
        self.open_annotated_btn.setEnabled(False)
        self.count_label.setText("Count: ...")
        self.detection_list.clear()
        if benchmark:
            efforts = ", ".join(REASONING_EFFORTS)
            self._set_running_status(f"Benchmarking {image_path.name} with {config.model}: {efforts}...")
            self.progress_bar.setRange(0, len(REASONING_EFFORTS))
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat("0/%m efforts")
        else:
            self._set_running_status(
                f"Analyzing {image_path.name} with {config.model} ({config.reasoning_effort} effort)..."
            )
            self.progress_bar.setRange(0, 0)
            self.progress_bar.setFormat("Working")
        self.progress_bar.setVisible(True)
        self.analyze_btn.setEnabled(False)
        self.benchmark_btn.setEnabled(False)
        self._start_run_timer()

        worker = CrabCounterWorker(config, benchmark=benchmark)
        thread = QThread(self)
        self._analysis_worker = worker
        self._analysis_thread = thread
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._handle_worker_progress)
        worker.finished.connect(self._finish_analysis)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(lambda: self._clear_analysis_thread(thread))
        thread.start()

    def _finish_analysis(self, payload: object) -> None:
        self._stop_run_timer()
        self.analyze_btn.setEnabled(True)
        self.benchmark_btn.setEnabled(True)
        data = payload if isinstance(payload, dict) else {}
        if not data.get("ok"):
            message = str(data.get("error") or "analysis failed")
            self.status_label.setText(message)
            self.count_label.setText("Count: -")
            self.progress_bar.setVisible(False)
            self.statusBar().showMessage(message, 8000)
            return
        outputs = data.get("outputs")
        if isinstance(outputs, CrabBenchmarkOutputs):
            self._finish_benchmark(outputs)
            return
        if not isinstance(outputs, CrabCounterOutputs):
            self.status_label.setText("Crab counter returned an unexpected result.")
            return
        self._last_outputs = outputs
        self._last_output_dir = outputs.output_dir
        result = outputs.result
        self.count_label.setText(f"Count: {result.count}")
        self.status_label.setText(
            f"Accepted {result.count} of {len(result.candidates)} crab candidate(s) in {result.analysis_seconds:.1f}s. "
            f"Wrote {outputs.annotated_image.name} and {outputs.result_json.name} under {outputs.output_dir}."
        )
        self._populate_detection_list(result)
        self.preview.set_result(result)
        self.open_output_btn.setEnabled(True)
        self.open_annotated_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.statusBar().showMessage(f"European green crabs: {result.count}", 8000)

    def _finish_benchmark(self, outputs: CrabBenchmarkOutputs) -> None:
        self._last_benchmark_outputs = outputs
        self._last_output_dir = outputs.output_dir
        selected = self._benchmark_selected_run(outputs)
        self._last_outputs = selected
        result = selected.result
        self.count_label.setText(f"Count: {result.count}")
        self.status_label.setText(
            f"Benchmark complete. Wrote {outputs.summary_csv.name} and {outputs.summary_json.name} under {outputs.output_dir}."
        )
        self.detection_list.clear()
        for run in outputs.runs:
            run_result = run.result
            self.detection_list.addItem(
                f"{run_result.reasoning_effort}: {run_result.count}/{len(run_result.candidates)} "
                f"candidate(s) accepted in {run_result.analysis_seconds:.1f}s"
            )
        self.preview.set_result(result)
        self.open_output_btn.setEnabled(True)
        self.open_annotated_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.statusBar().showMessage(f"Benchmark complete: {outputs.output_dir}", 8000)

    def _benchmark_selected_run(self, outputs: CrabBenchmarkOutputs) -> CrabCounterOutputs:
        preferred = self.reasoning_effort_combo.currentText()
        for run in outputs.runs:
            if run.result.reasoning_effort == preferred:
                return run
        return outputs.runs[-1]

    def _populate_detection_list(self, result: CrabCountResult) -> None:
        self.detection_list.clear()
        for index, detection in enumerate(result.detections, start=1):
            x0, y0, x1, y1 = detection.bbox
            self.detection_list.addItem(
                f"{index}: [{x0:.0f}, {y0:.0f}, {x1:.0f}, {y1:.0f}] "
                f"target {detection.target_match_confidence:.2f}, margin {detection.decision_margin:.2f}, "
                f"vs {detection.closest_non_target}"
            )

    def _preview_tab_changed(self, index: int) -> None:
        self.preview.set_display_mode("all" if index == 1 else "accepted")

    def _handle_worker_progress(self, payload: object) -> None:
        data = payload if isinstance(payload, dict) else {}
        event = str(data.get("event") or "")
        if event == "request_started":
            effort = str(data.get("effort") or self.reasoning_effort_combo.currentText())
            self._set_running_status(f"OpenAI request sent ({effort} effort). Waiting for model response...")
            return
        if event == "effort_started":
            effort = str(data.get("effort") or "")
            index = int(data.get("index") or 0)
            total = int(data.get("total") or len(REASONING_EFFORTS))
            completed = int(data.get("completed") or 0)
            self.progress_bar.setRange(0, max(1, total))
            self.progress_bar.setValue(max(0, completed))
            self.progress_bar.setFormat(f"{completed}/{total} efforts")
            self._set_running_status(f"Benchmarking effort {index}/{total}: {effort}. Waiting for model response...")
            return
        if event == "effort_finished":
            effort = str(data.get("effort") or "")
            total = int(data.get("total") or len(REASONING_EFFORTS))
            completed = int(data.get("completed") or 0)
            seconds = float(data.get("analysis_seconds") or 0.0)
            self.progress_bar.setRange(0, max(1, total))
            self.progress_bar.setValue(max(0, completed))
            self.progress_bar.setFormat(f"{completed}/{total} efforts")
            self._set_running_status(f"Finished {effort} in {seconds:.1f}s. Continuing benchmark...")

    def _start_run_timer(self) -> None:
        self._run_started_at = time.monotonic()
        self._run_timer.start()
        self._update_running_status()

    def _stop_run_timer(self) -> None:
        self._run_timer.stop()
        self._run_started_at = 0.0
        self._run_base_status = ""

    def _set_running_status(self, message: str) -> None:
        self._run_base_status = message
        self.status_label.setText(message)
        self._update_running_status()

    def _update_running_status(self) -> None:
        if self._analysis_thread is None or not self._run_started_at or not self._run_base_status:
            return
        elapsed = max(0.0, time.monotonic() - self._run_started_at)
        elapsed_text = self._format_elapsed(elapsed)
        message = f"{self._run_base_status} Elapsed {elapsed_text}."
        self.status_label.setText(message)
        self.statusBar().showMessage(message)

    def _format_elapsed(self, seconds: float) -> str:
        total_seconds = int(round(seconds))
        minutes, remainder = divmod(total_seconds, 60)
        return f"{minutes}:{remainder:02d}"

    def _clear_analysis_thread(self, thread: QThread) -> None:
        if self._analysis_thread is thread:
            self._analysis_thread = None
            self._analysis_worker = None

    def _open_output_folder(self) -> None:
        if self._last_output_dir is None:
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._last_output_dir)))

    def _open_annotated_image(self) -> None:
        if self._last_outputs is None:
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._last_outputs.annotated_image)))
