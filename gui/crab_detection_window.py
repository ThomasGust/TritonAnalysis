"""Crab-detection GUI for archive stills and fixed-board snapshots."""

from __future__ import annotations

from pathlib import Path

import cv2
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from crab_detector import (
    CrabDetectionError,
    CrabDetectionResult,
    detection_summary_text,
    detect_european_green_crabs,
    draw_european_green_crab_detections,
)
from gui.image_preview import ImagePreviewPanel
from gui.responsive import resize_to_available_screen, vertical_scroll_area
from tools.crab_yolo_predict import (
    DEFAULT_BOARD_CROP_SCALES,
    DEFAULT_CONFIDENCE,
    YoloPrediction,
    _board_crop_sources,
    _draw_predictions,
    _import_yolo,
    _inference_region_text,
    _predict_from_sources,
    latest_trained_weights,
)
from tools.crab_yolo_train import choose_training_device


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".wmv"}
SUPPORTED_MEDIA_SUFFIXES = IMAGE_SUFFIXES | VIDEO_SUFFIXES


def is_supported_media_path(path: Path) -> bool:
    """Return whether ``path`` is a supported image or video file."""
    return path.is_file() and path.suffix.lower() in SUPPORTED_MEDIA_SUFFIXES


def collect_media_paths(inputs: list[str | Path]) -> list[Path]:
    """Collect supported media from files/folders while preserving stable order."""
    ordered_paths: list[Path] = []
    seen: set[Path] = set()

    for raw_value in inputs:
        path = Path(raw_value).expanduser()
        if not path.exists():
            continue

        if path.is_dir():
            candidates = sorted(
                (child for child in path.rglob("*") if is_supported_media_path(child)),
                key=lambda child: tuple(part.lower() for part in child.parts),
            )
        else:
            candidates = [path] if is_supported_media_path(path) else []

        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved not in seen:
                seen.add(resolved)
                ordered_paths.append(resolved)

    return ordered_paths


class CrabDetectionWindow(QMainWindow):
    """Reference-board European green crab detector."""

    def __init__(
        self,
        *,
        image_paths: list[str | Path] | None = None,
        media_paths: list[str | Path] | None = None,
        reference_image: str | Path | None = None,
        detector_mode: str = "auto",
        yolo_model: str | Path | None = None,
        yolo_confidence: float = DEFAULT_CONFIDENCE,
        parent=None,
        **_legacy_options,
    ):
        super().__init__(parent)
        self.setWindowTitle("Crab Detection")
        self._media_paths: list[Path] = []
        self._reference_image = str(reference_image) if reference_image else None
        self._yolo_model_path = Path(yolo_model).expanduser() if yolo_model else None
        self._yolo_model = None
        self._yolo_model_loaded_path: Path | None = None
        self._current_image = None
        self._current_result: CrabDetectionResult | None = None
        self._current_yolo_predictions: list[YoloPrediction] = []
        self._current_annotated = None

        self.status_label = QLabel("Load a crab-board image or folder to count European green crabs.")
        self.status_label.setObjectName("summaryCard")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setWordWrap(True)
        self.status_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        self.detail_label = QLabel("")
        self.detail_label.setObjectName("summaryHint")
        self.detail_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.detail_label.setWordWrap(True)
        self.detail_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        self.path_list = QListWidget()
        self.path_list.setAlternatingRowColors(True)
        self.path_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.path_list.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.add_media_btn = QPushButton("Add Media...")
        self.add_media_btn.clicked.connect(self._choose_media)
        self.add_folder_btn = QPushButton("Add Folder...")
        self.add_folder_btn.clicked.connect(self._choose_folder)
        self.run_btn = QPushButton("Run Detection")
        self.run_btn.clicked.connect(self.run_current_detection)
        self.save_btn = QPushButton("Save Annotated...")
        self.save_btn.clicked.connect(self._save_annotated)
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self.clear_media)
        self.model_btn = QPushButton("YOLO Model...")
        self.model_btn.clicked.connect(self._choose_yolo_model)

        self.detector_combo = QComboBox()
        self.detector_combo.addItem("Auto", "auto")
        self.detector_combo.addItem("YOLO model", "yolo")
        self.detector_combo.addItem("Board projection", "board")
        requested_mode = detector_mode if detector_mode in {"auto", "yolo", "board"} else "auto"

        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0.01, 0.99)
        self.confidence_spin.setSingleStep(0.01)
        self.confidence_spin.setDecimals(2)
        self.confidence_spin.setValue(float(max(0.01, min(0.99, yolo_confidence))))
        self.confidence_spin.setToolTip("YOLO confidence threshold")
        mode_index = self.detector_combo.findData(requested_mode)
        self.detector_combo.setCurrentIndex(max(0, mode_index))
        self.detector_combo.currentIndexChanged.connect(self._handle_detector_mode_changed)
        self.confidence_spin.valueChanged.connect(self._handle_confidence_changed)

        controls = QHBoxLayout()
        controls.addWidget(self.add_media_btn)
        controls.addWidget(self.add_folder_btn)
        controls.addWidget(self.run_btn)
        controls.addWidget(self.save_btn)
        controls.addWidget(self.clear_btn)
        controls.addStretch(1)

        detector_controls = QHBoxLayout()
        detector_controls.addWidget(QLabel("Detector"))
        detector_controls.addWidget(self.detector_combo)
        detector_controls.addWidget(self.model_btn)
        detector_controls.addWidget(QLabel("Confidence"))
        detector_controls.addWidget(self.confidence_spin)
        detector_controls.addStretch(1)

        self.original_panel = ImagePreviewPanel("Source Image")
        self.annotated_panel = ImagePreviewPanel("European Green Crab Boxes")
        preview_row = QHBoxLayout()
        preview_row.addWidget(self.original_panel, 1)
        preview_row.addWidget(self.annotated_panel, 1)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)
        layout.addWidget(self.status_label)
        layout.addWidget(self.detail_label)
        layout.addLayout(controls)
        layout.addLayout(detector_controls)
        layout.addWidget(self.path_list, 0)
        layout.addLayout(preview_row, 1)

        self.setCentralWidget(vertical_scroll_area(content))
        self.path_list.currentRowChanged.connect(self._handle_selection_changed)
        self.save_btn.setEnabled(False)
        self._refresh_detector_detail()
        self.statusBar().showMessage("Crab detection ready.")
        self.load_media(media_paths if media_paths is not None else image_paths or [])
        resize_to_available_screen(self, 1280, 820, min_width=900, min_height=620)

    @property
    def media_paths(self) -> list[Path]:
        """Return the currently staged media paths."""
        return list(self._media_paths)

    def load_media(self, paths: list[str | Path]) -> None:
        """Load supported media paths for detection."""
        self._media_paths = collect_media_paths(paths)
        self._refresh_path_list()
        if self._media_paths:
            self.path_list.setCurrentRow(0)
            self.run_current_detection()

    def clear_media(self) -> None:
        """Clear staged media."""
        self._media_paths = []
        self._current_image = None
        self._current_result = None
        self._current_yolo_predictions = []
        self._current_annotated = None
        self._refresh_path_list()
        self.original_panel.clear("No source image")
        self.annotated_panel.clear("No detection result")
        self.status_label.setText("Load a crab-board image or folder to count European green crabs.")
        self.save_btn.setEnabled(False)

    def _choose_media(self) -> None:
        files, _selected_filter = QFileDialog.getOpenFileNames(
            self,
            "Stage crab media",
            "",
            "Media files (*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp *.mp4 *.mov *.m4v *.avi *.mkv *.wmv);;All files (*)",
        )
        if not files:
            return

        combined = [*self._media_paths, *files]
        self._media_paths = collect_media_paths(combined)
        self._refresh_path_list()
        if self._media_paths and self.path_list.currentRow() < 0:
            self.path_list.setCurrentRow(0)
        self.run_current_detection()

    def _choose_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self,
            "Stage crab media folder",
            "",
        )
        if not folder:
            return

        combined = [*self._media_paths, folder]
        self._media_paths = collect_media_paths(combined)
        self._refresh_path_list()
        if self._media_paths and self.path_list.currentRow() < 0:
            self.path_list.setCurrentRow(0)
        self.run_current_detection()

    def _choose_yolo_model(self) -> None:
        selected_path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "Choose YOLO crab model",
            str((latest_trained_weights() or Path()).parent),
            "YOLO weights (*.pt);;All files (*)",
        )
        if not selected_path:
            return
        self._yolo_model_path = Path(selected_path).expanduser()
        self._yolo_model = None
        self._yolo_model_loaded_path = None
        yolo_index = self.detector_combo.findData("yolo")
        if yolo_index >= 0:
            self.detector_combo.setCurrentIndex(yolo_index)
        self._refresh_detector_detail()
        self.run_current_detection()

    def _handle_detector_mode_changed(self, *_args) -> None:
        self._refresh_detector_detail()
        if self._current_image is not None:
            self.run_current_detection()

    def _handle_confidence_changed(self, *_args) -> None:
        if self._current_image is not None and self._active_detector_mode() == "yolo":
            self.run_current_detection()

    def _handle_selection_changed(self, row: int) -> None:
        if row < 0 or row >= len(self._media_paths):
            return
        self._load_current_image(run_detection=True)

    def _load_current_image(self, *, run_detection: bool) -> None:
        row = self.path_list.currentRow()
        if row < 0 or row >= len(self._media_paths):
            return
        path = self._media_paths[row]
        if path.suffix.lower() in VIDEO_SUFFIXES:
            self._current_image = None
            self._current_result = None
            self._current_yolo_predictions = []
            self._current_annotated = None
            self.original_panel.clear("Video detection is not rebuilt yet")
            self.annotated_panel.clear("No detection result")
            self.status_label.setText("Video scanning is not available yet. Use saved image frames for this rebuild.")
            self.statusBar().showMessage("Video scanning is not available yet.", 5000)
            self.save_btn.setEnabled(False)
            return

        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            self._current_image = None
            self._current_result = None
            self._current_yolo_predictions = []
            self._current_annotated = None
            self.original_panel.clear("Could not read image")
            self.annotated_panel.clear("No detection result")
            self.status_label.setText(f"Could not read image: {path}")
            self.statusBar().showMessage("Could not read selected image.", 5000)
            self.save_btn.setEnabled(False)
            return

        self._current_image = image
        self._current_result = None
        self._current_yolo_predictions = []
        self._current_annotated = None
        self.original_panel.set_frame(image, placeholder_text="No source image")
        self.annotated_panel.clear("Run detection to draw boxes")
        self.status_label.setText(f"Loaded {path.name}")
        self.save_btn.setEnabled(False)
        self.statusBar().showMessage(f"Loaded {path}", 5000)
        if run_detection:
            self.run_current_detection()

    def run_current_detection(self) -> None:
        if self._current_image is None:
            self._load_current_image(run_detection=False)
        if self._current_image is None:
            return
        if self._active_detector_mode() == "yolo":
            self._run_yolo_detection()
            return

        self._run_board_projection_detection()

    def _run_board_projection_detection(self) -> None:
        if self._current_image is None:
            return
        try:
            result = detect_european_green_crabs(self._current_image, reference_image=self._reference_image)
        except CrabDetectionError as exc:
            self._current_result = None
            self._current_yolo_predictions = []
            self._current_annotated = draw_european_green_crab_detections(self._current_image, None)
            self.annotated_panel.set_frame(
                self._current_annotated,
                placeholder_text="No detection result",
            )
            self.status_label.setText(str(exc))
            self.statusBar().showMessage(str(exc), 8000)
            self.save_btn.setEnabled(False)
            return

        self._current_result = result
        self._current_yolo_predictions = []
        annotated = draw_european_green_crab_detections(self._current_image, result)
        self._current_annotated = annotated
        self.annotated_panel.set_frame(annotated, placeholder_text="No detection result")
        summary = detection_summary_text(result)
        self.status_label.setText(summary)
        self.statusBar().showMessage(summary, 8000)
        self.save_btn.setEnabled(True)

    def _run_yolo_detection(self) -> None:
        if self._current_image is None:
            return
        model_path = self._resolved_yolo_model_path()
        if model_path is None:
            self._current_result = None
            self._current_yolo_predictions = []
            self._current_annotated = None
            self.annotated_panel.clear("No YOLO model")
            self.status_label.setText("No YOLO crab model found.")
            self.statusBar().showMessage("Choose YOLO Model... or train a crab YOLO model first.", 8000)
            self.save_btn.setEnabled(False)
            return

        try:
            model = self._load_yolo_model(model_path)
            sources = _board_crop_sources(
                self._current_image,
                reference_image=self._reference_image,
                image_size=640,
                crop_scales=DEFAULT_BOARD_CROP_SCALES,
            )
            predictions = _predict_from_sources(
                model,
                self._current_image,
                sources,
                imgsz=640,
                conf=float(self.confidence_spin.value()),
                iou=0.25,
                device=choose_training_device(None),
            )
            inference_region = _inference_region_text(sources)
        except Exception as exc:
            self._current_result = None
            self._current_yolo_predictions = []
            self._current_annotated = None
            self.annotated_panel.clear("YOLO detection failed")
            self.status_label.setText(f"YOLO detection failed: {exc}")
            self.statusBar().showMessage("YOLO detection failed.", 8000)
            self.save_btn.setEnabled(False)
            return

        self._current_result = None
        self._current_yolo_predictions = predictions
        self._current_annotated = _draw_predictions(self._current_image, predictions)
        self.annotated_panel.set_frame(self._current_annotated, placeholder_text="No detection result")
        summary = f"European green crabs: {len(predictions)}"
        self.status_label.setText(summary)
        self.statusBar().showMessage(f"{summary} ({inference_region})", 8000)
        self.save_btn.setEnabled(True)

    def _active_detector_mode(self) -> str:
        selected = self.detector_combo.currentData() or "auto"
        if selected == "auto":
            return "yolo" if self._resolved_yolo_model_path() is not None else "board"
        return str(selected)

    def _resolved_yolo_model_path(self) -> Path | None:
        if self._yolo_model_path is not None and self._yolo_model_path.exists():
            return self._yolo_model_path.resolve()
        latest = latest_trained_weights()
        return latest.resolve() if latest is not None and latest.exists() else None

    def _load_yolo_model(self, model_path: Path):
        if self._yolo_model is None or self._yolo_model_loaded_path != model_path:
            YOLO = _import_yolo()
            self._yolo_model = YOLO(str(model_path))
            self._yolo_model_loaded_path = model_path
        return self._yolo_model

    def _refresh_detector_detail(self) -> None:
        active = self._active_detector_mode()
        if active == "yolo":
            model_path = self._resolved_yolo_model_path()
            model_name = model_path.name if model_path is not None else "No model selected"
            self.detail_label.setText(f"YOLO model: {model_name}")
            self.model_btn.setEnabled(True)
            self.confidence_spin.setEnabled(True)
            return
        self.detail_label.setText("Board projection detector")
        self.model_btn.setEnabled(True)
        self.confidence_spin.setEnabled(False)

    def _save_annotated(self) -> None:
        if self._current_image is None:
            return
        row = self.path_list.currentRow()
        stem = "crab_detection"
        if 0 <= row < len(self._media_paths):
            stem = self._media_paths[row].stem
        selected_path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save annotated crab image",
            f"{stem}_european_green_crabs.jpg",
            "JPEG image (*.jpg);;PNG image (*.png);;All files (*)",
        )
        if not selected_path:
            return
        annotated = self._current_annotated
        if annotated is None:
            annotated = draw_european_green_crab_detections(self._current_image, self._current_result)
        if cv2.imwrite(selected_path, annotated):
            self.statusBar().showMessage(f"Saved annotated image: {selected_path}", 8000)
            return
        self.statusBar().showMessage("Could not save annotated image.", 8000)

    def _refresh_path_list(self) -> None:
        self.path_list.clear()
        for path in self._media_paths:
            self.path_list.addItem(str(path))

        count = len(self._media_paths)
        if count:
            self.statusBar().showMessage(f"Loaded {count} media file(s).")
            return
        self.path_list.addItem("No media staged.")
        self.path_list.item(0).setFlags(Qt.ItemFlag.NoItemFlags)
        self.statusBar().showMessage("No media staged.")
