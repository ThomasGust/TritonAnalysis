"""GUI for OpenAI-assisted European green crab counting."""

from __future__ import annotations

import os
import time
import json
from dataclasses import replace
from pathlib import Path

from PyQt6.QtCore import QObject, QRectF, Qt, QThread, QTimer, QUrl, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QDesktopServices, QFont, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
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
    QSplitter,
    QSpinBox,
    QTabBar,
    QVBoxLayout,
    QWidget,
)

from triton_analysis.crab.counter import (
    DEFAULT_HOMOGRAPHY_MODEL,
    DEFAULT_HOMOGRAPHY_REASONING_EFFORT,
    DEFAULT_MODEL,
    DEFAULT_REASONING_EFFORT,
    DEFAULT_TARGET_MARGIN_THRESHOLD,
    DEFAULT_TARGET_MATCH_THRESHOLD,
    REFERENCE_CLASS_LABELS,
    REASONING_EFFORTS,
    TARGET_CLASS,
    UNCERTAIN_CLASS,
    CrabBenchmarkOutputs,
    CrabCandidateDetectionResult,
    CrabCountResult,
    CrabCounterConfig,
    CrabCounterOutputs,
    CrabDetection,
    CrabEnsembleImageRun,
    CrabEnsembleOutputs,
    CrabPreprocessResult,
    analyze_crab_image,
    analyze_crab_image_ensemble,
    analyze_crab_image_pipeline,
    auto_preprocess_crab_target_image,
    benchmark_crab_image,
    benchmark_crab_image_pipeline,
    default_output_dir,
    discover_crab_board_reference_paths,
    discover_crab_detector_reference_paths,
    discover_counter_reference_atlas_paths,
    discover_counter_reference_paths,
    missing_reference_classes,
    preprocess_crab_target_image,
    transform_crab_count_result,
    write_reference_atlas,
)
from triton_analysis.crab.synthetic import CRAB_CLASS_NAMES, IMAGE_EXTENSIONS
from triton_analysis.gui.file_dialogs import ThumbnailFileDialog as QFileDialog
from triton_analysis.gui.job_center import JobReporter
from triton_analysis.gui.responsive import resize_to_available_screen
from triton_analysis.workspace import fresh_output_subdir, latest_pilot_run_dir, recent_pilot_run_dirs, workspace_paths


INVASIVE_SPECIES_FORM_URL = "https://cbjfq.share.hsforms.com/2rHEWllQ5QO6D7Z4CwVM7IQ"
JUDGE_DISPLAY_TAB_INDEX = 3
JUDGE_COUNT_COLOR = QColor(0, 255, 90)
MIN_SAMPLE_IMAGES = 1
MAX_SAMPLE_IMAGES = 10
TRITONPILOT_RECORDINGS_ENV = "TRITON_PILOT_RECORDINGS"


class WheelGuardComboBox(QComboBox):
    """Combo box that will not change values from accidental scroll-wheel ticks."""

    def wheelEvent(self, event) -> None:
        event.ignore()


class WheelGuardDoubleSpinBox(QDoubleSpinBox):
    """Spin box that will not change values from accidental scroll-wheel ticks."""

    def wheelEvent(self, event) -> None:
        event.ignore()


class CrabCounterPreview(QWidget):
    """Scaled image preview with result boxes painted on top."""

    selectionChanged = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap = QPixmap()
        self._image_path: Path | None = None
        self._result: CrabCountResult | None = None
        self._candidate_result: CrabCountResult | None = None
        self._display_mode = "accepted"
        self._interaction_mode = "none"
        self._selection_visible = True
        self._crop_anchor: tuple[float, float] | None = None
        self._crop_rect: tuple[float, float, float, float] | None = None
        self._homography_points: list[tuple[float, float]] = []
        self.setMinimumSize(360, 280)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_image(self, path: str | Path | None) -> None:
        self._image_path = Path(path).expanduser() if path else None
        self._pixmap = QPixmap(str(self._image_path)) if self._image_path else QPixmap()
        self._result = None
        self._candidate_result = None
        self.clear_selection()
        self.update()

    def set_result(self, result: CrabCountResult | None, annotated_image: str | Path | None = None) -> None:
        self._result = result
        self._candidate_result = None
        if annotated_image:
            self._image_path = Path(annotated_image).expanduser()
            self._pixmap = QPixmap(str(self._image_path))
        self.update()

    def set_candidate_result(self, result: CrabCountResult | None, image_path: str | Path | None = None) -> None:
        self._result = None
        self._candidate_result = result
        if image_path:
            self._image_path = Path(image_path).expanduser()
            self._pixmap = QPixmap(str(self._image_path))
        self.update()

    def set_display_mode(self, mode: str) -> None:
        self._display_mode = mode if mode in {"accepted", "all", "judge"} else "accepted"
        self.update()

    def set_interaction_mode(self, mode: str) -> None:
        self._interaction_mode = mode if mode in {"crop", "homography"} else "none"
        cursor = Qt.CursorShape.CrossCursor if self._interaction_mode != "none" else Qt.CursorShape.ArrowCursor
        self.setCursor(cursor)
        self.update()

    def set_selection_visible(self, visible: bool) -> None:
        self._selection_visible = bool(visible)
        self.update()

    def clear_selection(self) -> None:
        self._crop_anchor = None
        self._crop_rect = None
        self._homography_points = []
        self.selectionChanged.emit()
        self.update()

    def crop_rect(self) -> tuple[float, float, float, float] | None:
        return self._crop_rect

    def homography_points(self) -> tuple[tuple[float, float], ...]:
        return tuple(self._homography_points)

    def set_homography_points(self, points: object) -> None:
        parsed: list[tuple[float, float]] = []
        if isinstance(points, (list, tuple)):
            for raw in points[:4]:
                try:
                    x = float(raw[0])
                    y = float(raw[1])
                except (TypeError, ValueError, IndexError):
                    continue
                parsed.append((x, y))
        self._homography_points = parsed[:4]
        self.selectionChanged.emit()
        self.update()

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(15, 15, 19))
        if self._pixmap.isNull():
            painter.setPen(QColor(180, 184, 196))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No sample image selected")
            return

        image_rect = self._image_rect()
        painter.drawPixmap(image_rect, self._pixmap, QRectF(self._pixmap.rect()))
        draw_result = self._result or self._candidate_result
        if draw_result is None:
            if self._selection_visible:
                self._draw_selection(painter, image_rect)
            return
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        stage_candidates = self._result is None and self._candidate_result is not None
        judge_mode = self._display_mode == "judge" and not stage_candidates
        detections = draw_result.candidates if (self._display_mode == "all" or stage_candidates) else draw_result.detections
        self._draw_detections(
            painter,
            image_rect,
            draw_result.image_size,
            detections,
            stage_candidates=stage_candidates,
            show_labels=not judge_mode,
        )
        if judge_mode:
            self._draw_judge_count(painter, image_rect, draw_result.image_size, draw_result.detections, draw_result.count)
        if self._selection_visible:
            self._draw_selection(painter, image_rect)

    def _draw_detections(
        self,
        painter: QPainter,
        image_rect: QRectF,
        image_size: tuple[int, int],
        detections,
        *,
        stage_candidates: bool = False,
        show_labels: bool = True,
    ) -> None:
        for index, detection in enumerate(detections, start=1):
            rect = self._detection_rect(image_rect, image_size, detection)
            color = QColor(80, 210, 255) if stage_candidates else self._candidate_color(detection)
            painter.setPen(QPen(QColor(0, 0, 0), 3.5))
            painter.drawRect(rect)
            painter.setPen(QPen(color, 2.0))
            painter.drawRect(rect)
            if not show_labels:
                continue
            label = f"C{index}" if stage_candidates else self._candidate_label(detection, index=index)
            if label:
                text_rect = QRectF(rect.left(), max(image_rect.top(), rect.top() - 16), 72, 15)
                self._draw_overlay_text(painter, text_rect, label, color)

    @staticmethod
    def _draw_overlay_text(
        painter: QPainter,
        text_rect: QRectF,
        text: str,
        color: QColor,
        *,
        point_size: int | None = None,
    ) -> None:
        old_font = painter.font()
        font = QFont(old_font)
        font.setPointSize(point_size if point_size is not None else max(8, old_font.pointSize() - 1))
        font.setBold(True)
        painter.setFont(font)
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            painter.setPen(QColor(0, 0, 0, 210))
            painter.drawText(
                text_rect.translated(dx, dy),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                text,
            )
        painter.setPen(color)
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, text)
        painter.setFont(old_font)

    def _draw_judge_count(
        self,
        painter: QPainter,
        image_rect: QRectF,
        image_size: tuple[int, int],
        detections,
        count: int,
    ) -> None:
        long_text = f"European Green crabs: {int(count)}"
        short_text = f"EGC: {int(count)}"
        old_font = painter.font()
        font = QFont(old_font)
        font.setPointSize(18)
        font.setBold(True)
        painter.setFont(font)
        metrics = painter.fontMetrics()
        margin = 8.0
        header_height = self._judge_count_header_height()
        header_rect = QRectF(
            image_rect.left(),
            max(float(self.rect().top()) + 4.0, image_rect.top() - header_height),
            image_rect.width(),
            max(28.0, header_height - 8.0),
        )
        max_width = max(80.0, header_rect.width() - margin * 2.0)
        text = short_text if metrics.horizontalAdvance(long_text) + 18 > max_width else long_text
        text_width = min(max_width, float(metrics.horizontalAdvance(text) + 18))
        count_rect = QRectF(
            header_rect.left() + margin,
            header_rect.top(),
            text_width,
            header_rect.height(),
        )
        painter.setFont(old_font)
        self._draw_overlay_text(painter, count_rect, text, JUDGE_COUNT_COLOR, point_size=18)

    def _judge_count_header_height(self) -> float:
        return 44.0

    @staticmethod
    def _detection_rect(image_rect: QRectF, image_size: tuple[int, int], detection) -> QRectF:
        scale_x = image_rect.width() / max(1, image_size[0])
        scale_y = image_rect.height() / max(1, image_size[1])
        x0, y0, x1, y1 = detection.bbox
        return QRectF(
            image_rect.left() + x0 * scale_x,
            image_rect.top() + y0 * scale_y,
            max(1.0, (x1 - x0) * scale_x),
            max(1.0, (y1 - y0) * scale_y),
        )

    def mousePressEvent(self, event) -> None:
        if self._pixmap.isNull():
            return
        point = self._widget_to_image_point(event.position())
        if point is None:
            return
        if self._interaction_mode == "crop" and event.button() == Qt.MouseButton.LeftButton:
            self._crop_anchor = point
            self._crop_rect = (point[0], point[1], point[0], point[1])
            self.selectionChanged.emit()
            self.update()
            return
        if self._interaction_mode == "homography":
            if event.button() == Qt.MouseButton.LeftButton:
                if len(self._homography_points) >= 4:
                    self._homography_points = []
                self._homography_points.append(point)
                self.selectionChanged.emit()
                self.update()
                return
            if event.button() == Qt.MouseButton.RightButton and self._homography_points:
                self._homography_points.pop()
                self.selectionChanged.emit()
                self.update()

    def mouseMoveEvent(self, event) -> None:
        if self._interaction_mode != "crop" or self._crop_anchor is None:
            return
        point = self._widget_to_image_point(event.position())
        if point is None:
            return
        self._crop_rect = self._normalized_rect((*self._crop_anchor, point[0], point[1]))
        self.selectionChanged.emit()
        self.update()

    def mouseReleaseEvent(self, event) -> None:
        if self._interaction_mode != "crop" or self._crop_anchor is None or event.button() != Qt.MouseButton.LeftButton:
            return
        point = self._widget_to_image_point(event.position())
        if point is not None:
            self._crop_rect = self._normalized_rect((*self._crop_anchor, point[0], point[1]))
        self._crop_anchor = None
        self.selectionChanged.emit()
        self.update()

    def _image_rect(self) -> QRectF:
        margin = 10.0
        top_inset = self._judge_count_header_height() if self._display_mode == "judge" and self._result is not None else 0.0
        available = QRectF(self.rect()).adjusted(margin, margin + top_inset, -margin, -margin)
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

    def _widget_to_image_point(self, point) -> tuple[float, float] | None:
        image_rect = self._image_rect()
        if not image_rect.contains(point):
            return None
        x = (point.x() - image_rect.left()) / max(1.0, image_rect.width()) * max(1, self._pixmap.width())
        y = (point.y() - image_rect.top()) / max(1.0, image_rect.height()) * max(1, self._pixmap.height())
        return (
            max(0.0, min(float(self._pixmap.width()), float(x))),
            max(0.0, min(float(self._pixmap.height()), float(y))),
        )

    def _image_to_widget_point(self, point: tuple[float, float], image_rect: QRectF) -> tuple[float, float]:
        scale_x = image_rect.width() / max(1, self._pixmap.width())
        scale_y = image_rect.height() / max(1, self._pixmap.height())
        return (image_rect.left() + point[0] * scale_x, image_rect.top() + point[1] * scale_y)

    def _image_to_widget_rect(
        self,
        rect: tuple[float, float, float, float],
        image_rect: QRectF,
    ) -> QRectF:
        x0, y0, x1, y1 = self._normalized_rect(rect)
        p0 = self._image_to_widget_point((x0, y0), image_rect)
        p1 = self._image_to_widget_point((x1, y1), image_rect)
        return QRectF(p0[0], p0[1], max(1.0, p1[0] - p0[0]), max(1.0, p1[1] - p0[1]))

    def _normalized_rect(self, rect: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        x0, y0, x1, y1 = rect
        return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))

    def _draw_selection(self, painter: QPainter, image_rect: QRectF) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        if self._crop_rect is not None:
            rect = self._image_to_widget_rect(self._crop_rect, image_rect)
            painter.setPen(QPen(QColor(255, 255, 255), 5.0))
            painter.drawRect(rect)
            painter.setPen(QPen(QColor(50, 190, 255), 2.5))
            painter.drawRect(rect)
        if self._homography_points:
            widget_points = [self._image_to_widget_point(point, image_rect) for point in self._homography_points]
            painter.setPen(QPen(QColor(0, 0, 0), 5.0))
            for first, second in zip(widget_points, widget_points[1:]):
                painter.drawLine(int(round(first[0])), int(round(first[1])), int(round(second[0])), int(round(second[1])))
            if len(widget_points) == 4:
                first, last = widget_points[0], widget_points[-1]
                painter.drawLine(int(round(last[0])), int(round(last[1])), int(round(first[0])), int(round(first[1])))
            painter.setPen(QPen(QColor(255, 220, 70), 2.5))
            for first, second in zip(widget_points, widget_points[1:]):
                painter.drawLine(int(round(first[0])), int(round(first[1])), int(round(second[0])), int(round(second[1])))
            if len(widget_points) == 4:
                first, last = widget_points[0], widget_points[-1]
                painter.drawLine(int(round(last[0])), int(round(last[1])), int(round(first[0])), int(round(first[1])))
            painter.setBrush(QBrush(QColor(255, 220, 70)))
            for index, point in enumerate(widget_points, start=1):
                center = QRectF(point[0] - 6, point[1] - 6, 12, 12)
                painter.setPen(QPen(QColor(0, 0, 0), 4.0))
                painter.drawEllipse(center)
                painter.setPen(QPen(QColor(255, 220, 70), 2.0))
                painter.drawEllipse(center)
                painter.setPen(QColor(255, 255, 255))
                painter.drawText(QRectF(point[0] + 8, point[1] - 14, 24, 20), Qt.AlignmentFlag.AlignLeft, str(index))

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

    def _candidate_label(self, detection, *, index: int | None = None) -> str:
        short = {
            TARGET_CLASS: "EGC",
            "native_rock_crab": "Rock",
            "jonah_crab": "Jonah",
            UNCERTAIN_CLASS: "?",
        }.get(detection.label, detection.label)
        if detection.label == TARGET_CLASS and detection.accepted_as_target and index is not None:
            return f"{short} {index}"
        return short


class CrabCounterWorker(QObject):
    """Run one crab-counter request off the UI thread."""

    progress = pyqtSignal(object)
    finished = pyqtSignal(object)

    def __init__(
        self,
        config: CrabCounterConfig,
        *,
        benchmark: bool = False,
        preprocess_mode: str = "none",
        preprocess_output_dir: Path | None = None,
        homography_model: str = DEFAULT_HOMOGRAPHY_MODEL,
        homography_effort: str = DEFAULT_HOMOGRAPHY_REASONING_EFFORT,
        board_reference_paths: tuple[Path, ...] = (),
        analysis_flow: str = "pipeline",
        ensemble_configs: tuple[CrabCounterConfig, ...] = (),
        ensemble_output_dir: Path | None = None,
    ):
        super().__init__()
        self._config = config
        self._benchmark = benchmark
        self._preprocess_mode = preprocess_mode
        self._preprocess_output_dir = preprocess_output_dir
        self._homography_model = homography_model
        self._homography_effort = homography_effort
        self._board_reference_paths = board_reference_paths
        self._analysis_flow = analysis_flow
        self._ensemble_configs = ensemble_configs
        self._ensemble_output_dir = ensemble_output_dir

    def run(self) -> None:
        preprocess_result: CrabPreprocessResult | None = None
        try:
            config = self._config
            if self._ensemble_configs:
                outputs = analyze_crab_image_ensemble(
                    self._ensemble_configs,
                    output_dir=self._ensemble_output_dir or config.output_dir,
                    progress_callback=self.progress.emit,
                    preprocess_mode=self._preprocess_mode,
                    homography_model=self._homography_model,
                    homography_effort=self._homography_effort,
                    board_reference_paths=self._board_reference_paths,
                    validation_model=config.model,
                    validation_reasoning_effort=config.reasoning_effort,
                )
                selected_preprocess = outputs.selected_run.preprocess_result
                self.finished.emit(
                    {
                        "ok": True,
                        "outputs": outputs,
                        "benchmark": False,
                        "preprocess_result": selected_preprocess,
                    }
                )
                return
            if self._preprocess_mode == "auto_homography":
                output_dir = self._preprocess_output_dir or (Path(config.output_dir).expanduser() / "preprocess")
                self.progress.emit({"event": "auto_homography_started", "effort": self._homography_effort})
                try:
                    preprocess_result = auto_preprocess_crab_target_image(
                        config.image_path,
                        output_dir,
                        model=self._homography_model,
                        reasoning_effort=self._homography_effort,
                        board_reference_paths=self._board_reference_paths,
                        artifact_root=config.output_dir,
                    )
                except Exception as exc:
                    self.progress.emit({"event": "auto_homography_failed", "error": str(exc)})
                    raise RuntimeError(f"auto homography failed: {exc}") from exc
                else:
                    self.progress.emit(
                        {
                            "event": "auto_homography_finished",
                            "preprocess_result": preprocess_result,
                            "points": preprocess_result.ordered_points or preprocess_result.selection_points,
                            "confidence": preprocess_result.board_confidence,
                            "seconds": preprocess_result.board_detection_seconds,
                            "processed_image": str(preprocess_result.processed_image),
                        }
                    )
                    config = replace(config, image_path=preprocess_result.processed_image)
            if self._benchmark:
                if self._analysis_flow == "pipeline":
                    outputs = benchmark_crab_image_pipeline(config, progress_callback=self.progress.emit)
                else:
                    outputs = benchmark_crab_image(config, progress_callback=self.progress.emit)
            else:
                if self._analysis_flow == "pipeline":
                    outputs = analyze_crab_image_pipeline(config, progress_callback=self.progress.emit)
                else:
                    self.progress.emit({"event": "request_started", "effort": config.reasoning_effort})
                    outputs = analyze_crab_image(config)
        except Exception as exc:  # pragma: no cover - surfaced in GUI
            self.finished.emit({"ok": False, "error": str(exc)})
            return
        self.finished.emit(
            {
                "ok": True,
                "outputs": outputs,
                "benchmark": self._benchmark,
                "preprocess_result": preprocess_result,
            }
        )


class CrabCounterWindow(JobReporter, QMainWindow):
    """Count European green crabs on a saved MATE board image."""

    def __init__(
        self,
        *,
        image_path: str | Path | None = None,
        workspace_root: str | Path | None = None,
        job_center=None,
        parent=None,
    ):
        super().__init__(parent)
        self.attach_job_center(job_center, "crab-counter")
        self.setWindowTitle("Crab Counter")
        self._workspace = workspace_paths(workspace_root, create=True)
        self._analysis_thread: QThread | None = None
        self._analysis_worker: CrabCounterWorker | None = None
        self._last_outputs: CrabCounterOutputs | None = None
        self._last_benchmark_outputs: CrabBenchmarkOutputs | None = None
        self._last_preprocess_result: CrabPreprocessResult | None = None
        self._last_display_result: CrabCountResult | None = None
        self._last_projected_result: CrabCountResult | None = None
        self._last_candidate_display_result: CrabCountResult | None = None
        self._last_candidate_projected_result: CrabCountResult | None = None
        self._last_output_dir: Path | None = None
        self._sample_run_states: dict[str, dict[str, object]] = {}
        self._active_run_id: str | None = None
        self._run_started_at = 0.0
        self._run_base_status = ""
        self._run_timer = QTimer(self)
        self._run_timer.setInterval(1000)
        self._run_timer.timeout.connect(self._update_running_status)

        self.preview = CrabCounterPreview()
        self.preview.selectionChanged.connect(self._update_preprocess_status)
        self.preview_tabs = QTabBar()
        self.preview_tabs.addTab("European Green")
        self.preview_tabs.addTab("All Candidates")
        self.preview_tabs.addTab("Projected Board")
        self.preview_tabs.addTab("Judge Display")
        self.preview_tabs.currentChanged.connect(self._preview_tab_changed)
        self.sample_edits: list[QLineEdit] = []
        self.sample_row_widgets: list[QWidget] = []
        self.sample_label_widgets: list[QLabel] = []
        self.sample_rows_layout: QVBoxLayout | None = None
        self.target_edit = QLineEdit(str(Path(image_path).expanduser()) if image_path else "")
        self.target_edit.editingFinished.connect(self._load_target_preview)
        self.sample_edits.append(self.target_edit)
        self.sample_count_spin = QSpinBox()
        self.sample_count_spin.setRange(MIN_SAMPLE_IMAGES, MAX_SAMPLE_IMAGES)
        self.sample_count_spin.setValue(1)
        self.sample_count_spin.valueChanged.connect(self._set_sample_row_count)
        self.sample_count_spin.setToolTip("Choose how many board captures to analyze, from 1 to 10.")
        self.sample_mode_label = QLabel("Single staged run")
        self.sample_mode_label.setWordWrap(True)
        self.sample_status_list = QListWidget()
        self.sample_status_list.setMaximumHeight(130)
        self.sample_status_list.currentItemChanged.connect(self._sample_status_selection_changed)
        self.preprocess_mode_combo = WheelGuardComboBox()
        self.preprocess_mode_combo.addItem("Auto Homography", "auto_homography")
        self.preprocess_mode_combo.addItem("Full Frame (Legacy)", "none")
        self.preprocess_mode_combo.addItem("Manual Crop (Legacy)", "manual_crop")
        self.preprocess_mode_combo.addItem("Manual Homography (Legacy)", "manual_homography")
        self.preprocess_mode_combo.currentIndexChanged.connect(self._preprocess_mode_changed)
        self.preprocess_status_label = QLabel("OpenAI will locate and rectify the board before classification.")
        self.preprocess_status_label.setWordWrap(True)
        self.model_edit = QLineEdit(DEFAULT_MODEL)
        self.reasoning_effort_combo = WheelGuardComboBox()
        self.reasoning_effort_combo.addItems(REASONING_EFFORTS)
        default_effort_index = self.reasoning_effort_combo.findText(DEFAULT_REASONING_EFFORT)
        self.reasoning_effort_combo.setCurrentIndex(max(0, default_effort_index))
        self.threshold_spin = WheelGuardDoubleSpinBox()
        self.threshold_spin.setRange(0.5, 0.99)
        self.threshold_spin.setDecimals(2)
        self.threshold_spin.setSingleStep(0.05)
        self.threshold_spin.setValue(DEFAULT_TARGET_MATCH_THRESHOLD)
        self.threshold_spin.setToolTip("Higher values reduce false positives but can miss ambiguous European green crabs.")
        self.margin_spin = WheelGuardDoubleSpinBox()
        self.margin_spin.setRange(0.0, 0.5)
        self.margin_spin.setDecimals(2)
        self.margin_spin.setSingleStep(0.05)
        self.margin_spin.setValue(DEFAULT_TARGET_MARGIN_THRESHOLD)
        self.margin_spin.setToolTip("Higher values require European green crab to beat the closest non-target by more.")
        self.analysis_flow_combo = WheelGuardComboBox()
        self.analysis_flow_combo.addItem("3-Stage Pipeline", "pipeline")
        self.analysis_flow_combo.addItem("Single Request (Legacy)", "single")
        self.output_root_edit = QLineEdit(str(self._workspace.results / "crab_counter"))
        self.unlock_params_check = QCheckBox("Unlock Params")
        self.unlock_params_check.setToolTip("Leave this off during normal runs to avoid accidental model, flow, and threshold changes.")
        self.unlock_params_check.toggled.connect(self._update_param_lock_state)
        self.status_label = QLabel("Set OPENAI_API_KEY, choose board samples, then analyze.")
        self.status_label.setWordWrap(True)
        self.count_label = QLabel("Count: -")
        self.count_label.setObjectName("crabCounterCount")
        self.count_label.setStyleSheet("QLabel#crabCounterCount { color: #00ff66; font-size: 24px; font-weight: 800; }")
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(True)
        self.reference_edits: dict[str, QLineEdit] = {}
        self.detection_list = QListWidget()
        self.analyze_btn = QPushButton("Analyze Samples")
        self.analyze_btn.clicked.connect(self._start_analysis)
        self.benchmark_btn = QPushButton("Benchmark Effort")
        self.benchmark_btn.clicked.connect(self._start_benchmark)
        self.open_output_btn = QPushButton("Open Output")
        self.open_output_btn.clicked.connect(self._open_output_folder)
        self.open_output_btn.setEnabled(False)
        self.open_annotated_btn = QPushButton("Open Annotated")
        self.open_annotated_btn.clicked.connect(self._open_annotated_image)
        self.open_annotated_btn.setEnabled(False)
        self.open_species_form_btn = QPushButton("Open Species Form")
        self.open_species_form_btn.setObjectName("crabCounterSpeciesForm")
        self.open_species_form_btn.setMinimumHeight(34)
        self.open_species_form_btn.setStyleSheet(
            "QPushButton#crabCounterSpeciesForm { font-weight: 700; padding: 6px 14px; }"
        )
        self.open_species_form_btn.clicked.connect(self._open_invasive_species_form)
        self.open_preprocessed_btn = QPushButton("Open Input")
        self.open_preprocessed_btn.clicked.connect(self._open_preprocessed_image)
        self.open_preprocessed_btn.setEnabled(False)
        self._locked_param_widgets = (
            self.preprocess_mode_combo,
            self.model_edit,
            self.reasoning_effort_combo,
            self.analysis_flow_combo,
            self.threshold_spin,
            self.margin_spin,
            self.output_root_edit,
        )

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

        self._update_param_lock_state()
        self._load_default_references()
        self._load_target_preview()
        self.statusBar().showMessage("Crab counter ready.")
        resize_to_available_screen(self, 1300, 840, min_width=720, min_height=620)

    def _build_preview_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        judge_row = QHBoxLayout()
        judge_row.setContentsMargins(4, 0, 4, 0)
        judge_row.addWidget(self.count_label)
        judge_row.addStretch(1)
        judge_row.addWidget(self.open_species_form_btn)
        layout.addLayout(judge_row)
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

        sample_form = QFormLayout()
        sample_form.addRow("Sample Count", self.sample_count_spin)
        sample_form.addRow("Run Mode", self.sample_mode_label)
        layout.addWidget(QLabel("Board Samples"))
        layout.addLayout(sample_form)
        self.sample_rows_layout = QVBoxLayout()
        self.sample_rows_layout.setContentsMargins(0, 0, 0, 0)
        self.sample_rows_layout.setSpacing(6)
        layout.addLayout(self.sample_rows_layout)
        self._add_sample_row_widget(self.target_edit)
        sample_row = QHBoxLayout()
        latest_three_btn = QPushButton("Latest 3")
        latest_three_btn.setToolTip("Set Sample Count to 3 and fill from the newest synced Pilot session.")
        latest_three_btn.clicked.connect(self._use_latest_three_pilot_images)
        latest_btn = QPushButton("Fill Latest Session")
        latest_btn.setToolTip("Fill rows from the newest image files in Workspace incoming Pilot sessions.")
        latest_btn.clicked.connect(self._use_latest_pilot_images)
        clear_samples_btn = QPushButton("Clear Samples")
        clear_samples_btn.clicked.connect(self._clear_sample_images)
        sample_row.addWidget(latest_three_btn)
        sample_row.addWidget(latest_btn)
        sample_row.addWidget(clear_samples_btn)
        sample_row.addStretch(1)
        layout.addLayout(sample_row)
        layout.addWidget(QLabel("Sample Progress"))
        layout.addWidget(self.sample_status_list)

        preprocess_form = QFormLayout()
        preprocess_form.addRow("Preprocess", self.preprocess_mode_combo)
        layout.addLayout(preprocess_form)
        preprocess_row = QHBoxLayout()
        clear_preprocess_btn = QPushButton("Clear Selection")
        clear_preprocess_btn.clicked.connect(self._clear_preprocess_selection)
        preview_preprocess_btn = QPushButton("Preview Input")
        preview_preprocess_btn.clicked.connect(self._preview_preprocessed_target)
        preprocess_row.addWidget(clear_preprocess_btn)
        preprocess_row.addWidget(preview_preprocess_btn)
        preprocess_row.addWidget(self.open_preprocessed_btn)
        preprocess_row.addStretch(1)
        layout.addLayout(preprocess_row)
        layout.addWidget(self.preprocess_status_label)

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
        preview_atlas_btn = QPushButton("Preview Atlas")
        preview_atlas_btn.clicked.connect(self._preview_reference_atlas)
        reference_row.addWidget(autofill_btn)
        reference_row.addWidget(preview_atlas_btn)
        reference_row.addStretch(1)
        layout.addLayout(reference_row)

        settings_form = QFormLayout()
        settings_form.addRow("", self.unlock_params_check)
        settings_form.addRow("Model", self.model_edit)
        settings_form.addRow("Reasoning Effort", self.reasoning_effort_combo)
        settings_form.addRow("Flow", self.analysis_flow_combo)
        settings_form.addRow("EGC Threshold", self.threshold_spin)
        settings_form.addRow("EGC Margin", self.margin_spin)
        settings_form.addRow("Output Root", self.output_root_edit)
        layout.addLayout(settings_form)

        action_row = QHBoxLayout()
        action_row.addWidget(self.analyze_btn)
        action_row.addWidget(self.benchmark_btn)
        action_row.addStretch(1)
        output_action_row = QHBoxLayout()
        output_action_row.addWidget(self.open_output_btn)
        output_action_row.addWidget(self.open_annotated_btn)
        output_action_row.addStretch(1)
        layout.addLayout(action_row)
        layout.addLayout(output_action_row)

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

    def _update_param_lock_state(self) -> None:
        unlocked = self.unlock_params_check.isChecked()
        for widget in self._locked_param_widgets:
            widget.setEnabled(unlocked)
        self.unlock_params_check.setText("Params Unlocked" if unlocked else "Unlock Params")

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

    def _add_sample_row_widget(self, edit: QLineEdit) -> None:
        if self.sample_rows_layout is None:
            return
        row_widget = QWidget()
        row = QHBoxLayout(row_widget)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        label = QLabel()
        label.setMinimumWidth(64)
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(lambda _checked=False, row_widget=row_widget: self._choose_sample_image_for_row(row_widget))
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(lambda _checked=False, edit=edit: self._clear_sample_image(edit))
        row.addWidget(label)
        row.addWidget(edit, 1)
        row.addWidget(browse_btn)
        row.addWidget(clear_btn)
        self.sample_rows_layout.addWidget(row_widget)
        self.sample_row_widgets.append(row_widget)
        self.sample_label_widgets.append(label)
        self._refresh_sample_controls()

    def _set_sample_row_count(self, count: int) -> None:
        target_count = max(MIN_SAMPLE_IMAGES, min(MAX_SAMPLE_IMAGES, int(count)))
        if self.sample_count_spin.value() != target_count:
            self.sample_count_spin.setValue(target_count)
            return
        while len(self.sample_edits) < target_count:
            edit = QLineEdit()
            edit.editingFinished.connect(self._load_target_preview)
            self.sample_edits.append(edit)
            self._add_sample_row_widget(edit)
        while len(self.sample_edits) > target_count:
            self._remove_last_sample_row()
        self._refresh_sample_controls()
        self._load_target_preview()

    def _remove_last_sample_row(self) -> None:
        if len(self.sample_edits) <= MIN_SAMPLE_IMAGES:
            return
        self.sample_edits.pop()
        self.sample_label_widgets.pop()
        row_widget = self.sample_row_widgets.pop()
        if self.sample_rows_layout is not None:
            self.sample_rows_layout.removeWidget(row_widget)
        row_widget.deleteLater()

    def _refresh_sample_controls(self) -> None:
        for index, label in enumerate(self.sample_label_widgets, start=1):
            label.setText(f"Sample {index}")
        count = len(self.sample_edits)
        if self.sample_count_spin.value() != count:
            self.sample_count_spin.blockSignals(True)
            self.sample_count_spin.setValue(count)
            self.sample_count_spin.blockSignals(False)
        if count <= 1:
            self.sample_mode_label.setText("Single staged run")
        else:
            self.sample_mode_label.setText(f"{count}-sample ensemble")

    def _choose_target_image(self) -> None:
        self._choose_sample_image(0)

    def _choose_sample_image_for_row(self, row_widget: QWidget) -> None:
        try:
            index = self.sample_row_widgets.index(row_widget)
        except ValueError:
            return
        self._choose_sample_image(index)

    def _choose_sample_image(self, index: int) -> None:
        if index < 0 or index >= len(self.sample_edits):
            return
        start = self._sample_start_dir(index)
        path, _filter = QFileDialog.getOpenFileName(
            self,
            f"Choose crab-board sample {index + 1}",
            str(start),
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)",
        )
        if not path:
            return
        self.sample_edits[index].setText(path)
        self._load_target_preview()

    def _clear_sample_image(self, edit: QLineEdit) -> None:
        edit.clear()
        self._load_target_preview()

    def _clear_sample_images(self) -> None:
        for edit in self.sample_edits:
            edit.clear()
        self._load_target_preview()

    def _reset_sample_run_states(self) -> None:
        self._sample_run_states.clear()
        self._active_run_id = None
        self.sample_status_list.clear()

    def _initialize_sample_run_states(self, sample_paths: tuple[Path, ...]) -> None:
        self._reset_sample_run_states()
        for index, path in enumerate(sample_paths, start=1):
            run_id = f"image_{index}"
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, run_id)
            self.sample_status_list.addItem(item)
            self._sample_run_states[run_id] = {
                "run_id": run_id,
                "index": index,
                "source_image": path,
                "status": "Queued",
                "detail": path.name,
                "item": item,
            }
            self._update_sample_status_item(run_id)
        if self.sample_status_list.count():
            self.sample_status_list.setCurrentRow(0)

    def _sample_status_selection_changed(self, current: QListWidgetItem | None, _previous: QListWidgetItem | None) -> None:
        if current is None:
            return
        run_id = current.data(Qt.ItemDataRole.UserRole)
        if isinstance(run_id, str):
            self._activate_sample_run(run_id)

    def _ensure_sample_run_state(self, run_id: str, index: int = 0) -> dict[str, object]:
        state = self._sample_run_states.get(run_id)
        if state is not None:
            return state
        item = QListWidgetItem()
        item.setData(Qt.ItemDataRole.UserRole, run_id)
        self.sample_status_list.addItem(item)
        state = {
            "run_id": run_id,
            "index": index or len(self._sample_run_states) + 1,
            "status": "Queued",
            "detail": "",
            "item": item,
        }
        self._sample_run_states[run_id] = state
        self._update_sample_status_item(run_id)
        return state

    def _update_sample_status_item(self, run_id: str) -> None:
        state = self._sample_run_states.get(run_id)
        if not state:
            return
        item = state.get("item")
        if not isinstance(item, QListWidgetItem):
            return
        index = int(state.get("index") or 0)
        status = str(state.get("status") or "Queued")
        detail = str(state.get("detail") or "")
        count_text = ""
        if "count" in state:
            count_text = f" | count {int(state.get('count') or 0)}"
        item.setText(f"Sample {index}: {status}{count_text} | {detail}".strip())

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

    def _current_reference_paths(self) -> dict[str, Path | None]:
        return {
            class_name: Path(edit.text().strip()).expanduser() if edit.text().strip() else None
            for class_name, edit in self.reference_edits.items()
        }

    def _preview_reference_atlas(self) -> None:
        references = self._current_reference_paths()
        missing = missing_reference_classes(references)
        if missing:
            self.status_label.setText("Missing reference images: " + ", ".join(REFERENCE_CLASS_LABELS[name] for name in missing))
            return
        try:
            output_root = Path(self.output_root_edit.text().strip()).expanduser()
            output_path = output_root / "crab_reference_atlas_preview.png"
            atlas_path = write_reference_atlas(
                references,
                output_path,
                atlas_paths=discover_counter_reference_atlas_paths(self._workspace.root),
            )
        except Exception as exc:
            self.status_label.setText(f"Could not build reference atlas: {exc}")
            return
        self.status_label.setText(f"Wrote reference atlas preview: {atlas_path}")
        self.statusBar().showMessage(f"Reference atlas preview: {atlas_path}", 8000)
        self._show_reference_atlas_dialog(atlas_path)

    def _show_reference_atlas_dialog(self, atlas_path: Path) -> None:
        pixmap = QPixmap(str(atlas_path))
        if pixmap.isNull():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(atlas_path)))
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("Crab Reference Atlas")
        layout = QVBoxLayout(dialog)
        label = QLabel()
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setPixmap(pixmap)
        scroll = QScrollArea()
        scroll.setWidget(label)
        scroll.setWidgetResizable(False)
        layout.addWidget(scroll, 1)
        dialog.resize(min(980, max(520, pixmap.width() + 48)), min(680, max(360, pixmap.height() + 72)))
        dialog.exec()

    def _current_preprocess_mode(self) -> str:
        return str(self.preprocess_mode_combo.currentData() or "none")

    def _current_analysis_flow(self) -> str:
        return "single" if self.analysis_flow_combo.currentData() == "single" else "pipeline"

    def _preprocess_mode_changed(self) -> None:
        mode = self._current_preprocess_mode()
        if mode == "manual_crop":
            self.preview.set_interaction_mode("crop")
        elif mode == "manual_homography":
            self.preview.set_interaction_mode("homography")
        else:
            self.preview.set_interaction_mode("none")
        self._update_preprocess_status()

    def _clear_preprocess_selection(self) -> None:
        self.preview.clear_selection()
        self._update_preprocess_status()

    def _update_preprocess_status(self) -> None:
        mode = self._current_preprocess_mode()
        if mode == "manual_crop":
            crop_rect = self.preview.crop_rect()
            if crop_rect is None:
                self.preprocess_status_label.setText("Drag a rectangle on the image to crop before analysis.")
                return
            x0, y0, x1, y1 = crop_rect
            self.preprocess_status_label.setText(
                f"Manual crop: [{x0:.0f}, {y0:.0f}, {x1:.0f}, {y1:.0f}] "
                f"({max(0.0, x1 - x0):.0f} x {max(0.0, y1 - y0):.0f}px)."
            )
            return
        if mode == "manual_homography":
            point_count = len(self.preview.homography_points())
            self.preprocess_status_label.setText(
                f"Click four board corners for homography ({point_count}/4). Right-click removes the last point."
            )
            return
        if mode == "auto_homography":
            details = ""
            if self._last_preprocess_result and self._last_preprocess_result.mode == "auto_homography":
                confidence = self._last_preprocess_result.board_confidence
                if confidence is not None:
                    details = f" Last outline confidence {confidence:.2f}."
            self.preprocess_status_label.setText(
                "Auto homography: OpenAI will choose the board corners, draw them here, "
                f"and send a rectified board image to classification.{details}"
            )
            return
        self.preprocess_status_label.setText("Full frame will be sent.")

    def _preview_preprocessed_target(self) -> None:
        image_path = Path(self.target_edit.text().strip()).expanduser()
        if not image_path.is_file():
            self.status_label.setText("Choose Sample 1 first.")
            return
        mode = self._current_preprocess_mode()
        if mode == "none":
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(image_path)))
            return
        if mode == "auto_homography":
            if self._last_preprocess_result and self._last_preprocess_result.mode == "auto_homography":
                self._open_preprocessed_image()
                return
            self.status_label.setText("Auto homography runs during Analyze/Benchmark. Use Open Input after the outline response.")
            return
        try:
            preview_dir = Path(self.output_root_edit.text().strip()).expanduser() / "preprocess_preview"
            result = self._preprocess_target_image(image_path, preview_dir)
        except Exception as exc:
            self.status_label.setText(f"Could not preprocess sample: {exc}")
            return
        self._set_preprocess_result(result)
        self.status_label.setText(f"Wrote {result.mode} sample preview: {result.processed_image}")
        self.statusBar().showMessage(f"Preprocessed sample: {result.processed_image}", 8000)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(result.processed_image)))

    def _preprocess_target_image(self, image_path: Path, output_dir: Path) -> CrabPreprocessResult:
        mode = self._current_preprocess_mode()
        if mode == "manual_crop":
            crop_rect = self.preview.crop_rect()
            if crop_rect is None:
                raise ValueError("drag a crop rectangle first")
            return preprocess_crab_target_image(image_path, output_dir, mode=mode, crop_rect=crop_rect)
        if mode == "manual_homography":
            points = self.preview.homography_points()
            if len(points) != 4:
                raise ValueError("click four board corners first")
            return preprocess_crab_target_image(image_path, output_dir, mode=mode, homography_points=points)
        raise ValueError(f"unsupported preprocessing mode: {mode}")

    def _target_start_dir(self) -> Path:
        return self._sample_start_dir(0)

    def _sample_start_dir(self, index: int) -> Path:
        edit = self.sample_edits[index] if 0 <= index < len(self.sample_edits) else self.target_edit
        text = edit.text().strip()
        if text:
            path = Path(text).expanduser()
            return path.parent if path.is_file() else path
        return self._latest_synced_pilot_session_dir()

    def _use_latest_pilot_image(self) -> None:
        self._use_latest_pilot_images()

    def _use_latest_three_pilot_images(self) -> None:
        self.sample_count_spin.setValue(3)
        self._use_latest_pilot_images()

    def _use_latest_pilot_images(self) -> None:
        session = self._latest_synced_pilot_session_with_images()
        if session is not None:
            images = self._latest_images_in_tritonpilot_session(session, limit=len(self.sample_edits))
            source_label = self._synced_pilot_source_label(session)
        else:
            session = self._latest_local_tritonpilot_session_dir()
            if session is not None:
                images = self._latest_images_in_tritonpilot_session(session, limit=len(self.sample_edits))
                source_label = f"local TritonPilot session {session.name}"
            else:
                root = latest_pilot_run_dir(self._workspace.root, create=True)
                images = self._latest_images_under(root, limit=len(self.sample_edits))
                source_label = self._synced_pilot_source_label(root)
        if not images:
            recordings = self._tritonpilot_recordings_dir()
            self.status_label.setText(
                f"No images found under the synced Pilot inbox ({self._workspace.pilot_incoming}) "
                f"or optional local fallback ({recordings})."
            )
            return
        for index, edit in enumerate(self.sample_edits):
            edit.setText(str(images[index]) if index < len(images) else "")
        if len(images) < len(self.sample_edits):
            self.status_label.setText(f"Filled {len(images)} of {len(self.sample_edits)} sample row(s) from {source_label}.")
        else:
            self.status_label.setText(f"Filled {len(images)} sample row(s) from {source_label}.")
        self._load_target_preview()

    def _tritonpilot_recordings_dir(self) -> Path:
        override = os.environ.get(TRITONPILOT_RECORDINGS_ENV, "").strip()
        if override:
            return Path(override).expanduser()
        repo_root = Path(__file__).resolve().parents[2]
        return repo_root.parent / "TritonPilot" / "recordings"

    def _latest_synced_pilot_session_dir(self) -> Path:
        return latest_pilot_run_dir(self._workspace.root, create=True)

    def _latest_synced_pilot_session_with_images(self) -> Path | None:
        for session in recent_pilot_run_dirs(self._workspace.root, create=False):
            if self._latest_images_in_tritonpilot_session(session, limit=1):
                return session
        inbox = self._workspace.pilot_incoming
        if self._latest_images_in_tritonpilot_session(inbox, limit=1):
            return inbox
        return None

    def _synced_pilot_source_label(self, path: Path) -> str:
        label = self._workspace.label_for(path)
        if path == self._workspace.pilot_incoming:
            return f"synced Pilot inbox {label}"
        return f"synced Pilot session {path.name}"

    def _latest_local_tritonpilot_session_dir(self) -> Path | None:
        recordings = self._tritonpilot_recordings_dir()
        if not recordings.is_dir():
            return None
        sessions = sorted(
            (path for path in recordings.iterdir() if path.is_dir()),
            key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        )
        for session in sessions:
            if self._latest_images_in_tritonpilot_session(session, limit=1):
                return session
        return None

    def _latest_images_in_tritonpilot_session(self, session: str | Path, *, limit: int) -> list[Path]:
        session_path = Path(session).expanduser()
        direct_images = self._image_files_under(session_path, recursive=False)
        if len(direct_images) >= limit:
            return direct_images[: max(0, int(limit))]
        recursive_images = self._image_files_under(session_path, recursive=True, exclude_dirs={"stereo_sessions"})
        merged: list[Path] = []
        seen: set[Path] = set()
        for path in [*direct_images, *recursive_images]:
            try:
                key = path.resolve()
            except OSError:
                key = path
            if key in seen:
                continue
            seen.add(key)
            merged.append(path)
        merged.sort(key=lambda path: path.stat().st_mtime_ns, reverse=True)
        return merged[: max(0, int(limit))]

    def _latest_image_under(self, root: str | Path) -> Path | None:
        images = self._latest_images_under(root, limit=1)
        return images[0] if images else None

    def _latest_images_under(self, root: str | Path, *, limit: int) -> list[Path]:
        return self._image_files_under(root, recursive=True)[: max(0, int(limit))]

    def _image_files_under(
        self,
        root: str | Path,
        *,
        recursive: bool,
        exclude_dirs: set[str] | None = None,
    ) -> list[Path]:
        root_path = Path(root).expanduser()
        if not root_path.exists():
            return []
        exclude = {name.lower() for name in (exclude_dirs or set())}
        iterator = root_path.rglob("*") if recursive else root_path.iterdir()
        candidates: list[Path] = []
        for path in iterator:
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            if exclude and any(part.lower() in exclude for part in path.relative_to(root_path).parts[:-1]):
                continue
            candidates.append(path)
        candidates.sort(key=lambda path: path.stat().st_mtime_ns, reverse=True)
        return candidates

    def _load_target_preview(self) -> None:
        text = self.target_edit.text().strip()
        path = Path(text).expanduser() if text else None
        self._last_preprocess_result = None
        self._last_display_result = None
        self._last_projected_result = None
        self._last_candidate_display_result = None
        self._last_candidate_projected_result = None
        self.open_preprocessed_btn.setEnabled(False)
        if path and path.is_file():
            self.preview.set_image(path)
            self.statusBar().showMessage(f"Loaded sample image: {path.name}", 4000)
        else:
            self.preview.set_image(None)

    def _current_sample_paths(self) -> tuple[Path, ...]:
        paths: list[Path] = []
        for index, edit in enumerate(self.sample_edits, start=1):
            text = edit.text().strip()
            if not text:
                raise ValueError(f"choose Sample {index} first")
            path = Path(text).expanduser()
            if not path.is_file():
                raise ValueError(f"Sample {index} does not exist: {path}")
            paths.append(path)
        return tuple(paths)

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
        try:
            sample_paths = self._current_sample_paths()
        except ValueError as exc:
            self.status_label.setText(str(exc))
            return
        if benchmark and len(sample_paths) != 1:
            self.status_label.setText("Benchmark Effort uses one sample. Set Sample Count to 1 before benchmarking.")
            return
        ensemble_images = sample_paths if len(sample_paths) > 1 else ()
        ensemble_run = bool(ensemble_images)
        if ensemble_run:
            preprocess_mode_for_ensemble = self._current_preprocess_mode()
            if preprocess_mode_for_ensemble not in {"auto_homography", "none"}:
                self.status_label.setText("Multi-sample analysis supports Auto Homography or Full Frame mode.")
                return
            if self._current_analysis_flow() != "pipeline":
                self.status_label.setText("Multi-sample analysis requires the 3-Stage Pipeline flow.")
                return
        image_path = sample_paths[0]
        references = self._current_reference_paths()
        missing = missing_reference_classes(references)
        if missing:
            self.status_label.setText("Missing reference images: " + ", ".join(REFERENCE_CLASS_LABELS[name] for name in missing))
            return

        output_root = Path(self.output_root_edit.text().strip()).expanduser()
        if ensemble_run:
            output_prefix = "crab_ensemble"
        else:
            output_prefix = "crab_benchmark" if benchmark else "crab_counter"
        output_dir = fresh_output_subdir(output_root, output_prefix, create=True)
        analysis_image_path = image_path
        preprocess_mode = self._current_preprocess_mode()
        worker_preprocess_mode = "none"
        self._last_preprocess_result = None
        self.open_preprocessed_btn.setEnabled(False)
        if preprocess_mode != "none" and not ensemble_run:
            if preprocess_mode == "auto_homography":
                worker_preprocess_mode = "auto_homography"
            else:
                try:
                    preprocess_result = self._preprocess_target_image(image_path, output_dir / "preprocess")
                except Exception as exc:
                    self.status_label.setText(f"Could not preprocess sample: {exc}")
                    return
                self._set_preprocess_result(preprocess_result)
                analysis_image_path = preprocess_result.processed_image
        elif ensemble_run and preprocess_mode == "auto_homography":
            worker_preprocess_mode = "auto_homography"
        config = CrabCounterConfig(
            image_path=analysis_image_path,
            reference_paths={name: path for name, path in references.items() if path is not None},
            output_dir=output_dir,
            model=self.model_edit.text().strip() or DEFAULT_MODEL,
            reasoning_effort=self.reasoning_effort_combo.currentText(),
            target_confidence_threshold=self.threshold_spin.value(),
            target_margin_threshold=self.margin_spin.value(),
            reference_atlas_paths=discover_counter_reference_atlas_paths(self._workspace.root),
            detector_reference_paths=discover_crab_detector_reference_paths(self._workspace.root),
        )
        ensemble_configs = tuple(
            replace(config, image_path=image, output_dir=output_dir / f"image_{index}")
            for index, image in enumerate(ensemble_images, start=1)
        )
        if ensemble_run:
            self._initialize_sample_run_states(tuple(ensemble_images))
        else:
            self._reset_sample_run_states()
        self._last_outputs = None
        self._last_benchmark_outputs = None
        self._last_output_dir = None
        self._last_display_result = None
        self._last_projected_result = None
        self._last_candidate_display_result = None
        self._last_candidate_projected_result = None
        self.open_output_btn.setEnabled(False)
        self.open_annotated_btn.setEnabled(False)
        self.count_label.setText("Count: ...")
        self.detection_list.clear()
        if ensemble_run:
            job_title = f"Crab Counter · {len(ensemble_images)}-sample ensemble"
        elif benchmark:
            job_title = "Crab Counter · benchmark"
        else:
            job_title = "Crab Counter"
        self._begin_job(job_title)
        if ensemble_run:
            names = ", ".join(path.name for path in ensemble_images)
            prep = f" via {preprocess_mode}" if preprocess_mode != "none" else ""
            self._set_running_status(
                f"Analyzing {len(ensemble_images)} samples{prep} with {config.model}, then validating the best judge result: {names}..."
            )
            self.progress_bar.setRange(0, len(ensemble_images))
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat(f"0/{len(ensemble_images)} images")
        elif benchmark:
            efforts = ", ".join(REASONING_EFFORTS)
            prep = f" via {preprocess_mode}" if preprocess_mode != "none" else ""
            flow = "3-stage pipeline" if self._current_analysis_flow() == "pipeline" else "single request"
            self._set_running_status(f"Benchmarking {image_path.name}{prep} with {config.model} ({flow}): {efforts}...")
            self.progress_bar.setRange(0, len(REASONING_EFFORTS))
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat("0/%m efforts")
        else:
            prep = f" via {preprocess_mode}" if preprocess_mode != "none" else ""
            flow = "3-stage pipeline" if self._current_analysis_flow() == "pipeline" else "single request"
            self._set_running_status(
                f"Analyzing {image_path.name}{prep} with {config.model} ({config.reasoning_effort} effort, {flow})..."
            )
            self.progress_bar.setRange(0, 0)
            self.progress_bar.setFormat("Working")
        self.progress_bar.setVisible(True)
        self.analyze_btn.setEnabled(False)
        self.benchmark_btn.setEnabled(False)
        self._start_run_timer()

        worker = CrabCounterWorker(
            config,
            benchmark=benchmark,
            preprocess_mode=worker_preprocess_mode,
            preprocess_output_dir=output_dir / "preprocess",
            homography_model=DEFAULT_HOMOGRAPHY_MODEL,
            homography_effort=DEFAULT_HOMOGRAPHY_REASONING_EFFORT,
            board_reference_paths=discover_crab_board_reference_paths(self._workspace.root),
            analysis_flow=self._current_analysis_flow(),
            ensemble_configs=ensemble_configs,
            ensemble_output_dir=output_dir if ensemble_run else None,
        )
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
            self._fail_job(message)
            return
        outputs = data.get("outputs")
        preprocess_result = data.get("preprocess_result")
        if isinstance(preprocess_result, CrabPreprocessResult):
            self._set_preprocess_result(preprocess_result)
        if isinstance(outputs, CrabBenchmarkOutputs):
            self._finish_benchmark(outputs)
            return
        if isinstance(outputs, CrabEnsembleOutputs):
            self._finish_ensemble(outputs)
            return
        if not isinstance(outputs, CrabCounterOutputs):
            self.status_label.setText("Crab counter returned an unexpected result.")
            self._fail_job("Unexpected result")
            return
        self._last_outputs = outputs
        self._last_output_dir = outputs.output_dir
        result = outputs.result
        display_result = self._preview_result_for_current_run(result)
        self._last_projected_result = result
        self._last_display_result = display_result
        self._last_candidate_projected_result = None
        self._last_candidate_display_result = None
        self.count_label.setText(f"Count: {result.count}")
        mapping_note = " Preview boxes are mapped back to the original frame." if self._last_preprocess_result else ""
        self.status_label.setText(
            f"Accepted {result.count} of {len(result.candidates)} crab candidate(s) in {result.analysis_seconds:.1f}s. "
            f"Wrote {outputs.annotated_image.name}, {outputs.result_json.name}, and run_manifest.json under {outputs.output_dir}."
            f"{mapping_note}"
        )
        self._populate_detection_list(display_result)
        self._show_judge_display()
        self.open_output_btn.setEnabled(True)
        self.open_annotated_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.statusBar().showMessage(f"European green crabs: {result.count}", 8000)
        self._finish_job(ok=True, detail=f"{result.count} European green crab(s)")

    def _finish_ensemble(self, outputs: CrabEnsembleOutputs) -> None:
        self._last_outputs = outputs.final_outputs
        self._last_output_dir = outputs.output_dir
        for run in outputs.runs:
            self._store_sample_image_run(run)
        selected_state = self._ensure_sample_run_state(outputs.selected_run.run_id, outputs.selected_run.index)
        selected_state["projected_result"] = outputs.final_outputs.result
        selected_state["display_result"] = self._preview_result_for_preprocess(
            outputs.final_outputs.result,
            outputs.selected_run.preprocess_result,
        )
        selected_state["status"] = "Selected"
        selected_state["count"] = outputs.final_outputs.result.count
        selected_state["detail"] = f"validator confidence {outputs.selection.confidence:.2f}"
        self._update_sample_status_item(outputs.selected_run.run_id)
        self._active_run_id = outputs.selected_run.run_id
        selected_item = selected_state.get("item")
        if isinstance(selected_item, QListWidgetItem):
            self.sample_status_list.setCurrentItem(selected_item)
        self._activate_sample_run(outputs.selected_run.run_id)
        result = outputs.final_outputs.result
        self.count_label.setText(f"Count: {result.count}")
        self.status_label.setText(
            f"Multi-sample ensemble selected {outputs.selection.selected_run_id} "
            f"(confidence {outputs.selection.confidence:.2f}). "
            f"Accepted {result.count} of {len(result.candidates)} crab candidate(s). "
            f"Wrote {outputs.final_outputs.result_json.name}, {outputs.validation_json.name}, "
            f"and run_manifest.json under {outputs.output_dir}."
        )
        if self._last_display_result is not None:
            self._populate_detection_list(self._last_display_result)
            self.detection_list.addItem(f"Selected {outputs.selection.selected_run_id}: {outputs.selection.rationale}")
        self._show_judge_display()
        self.open_output_btn.setEnabled(True)
        self.open_annotated_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.statusBar().showMessage(
            f"Ensemble selected {outputs.selection.selected_run_id}: {result.count} European green crab(s)",
            8000,
        )
        self._finish_job(ok=True, detail=f"{result.count} European green crab(s) (ensemble)")

    def _finish_benchmark(self, outputs: CrabBenchmarkOutputs) -> None:
        self._last_benchmark_outputs = outputs
        self._last_output_dir = outputs.output_dir
        selected = self._benchmark_selected_run(outputs)
        self._last_outputs = selected
        result = selected.result
        display_result = self._preview_result_for_current_run(result)
        self._last_projected_result = result
        self._last_display_result = display_result
        self._last_candidate_projected_result = None
        self._last_candidate_display_result = None
        self.count_label.setText(f"Count: {result.count}")
        self.status_label.setText(
            f"Benchmark complete. Wrote {outputs.summary_csv.name}, {outputs.summary_json.name}, "
            f"and run_manifest.json under {outputs.output_dir}."
        )
        self.detection_list.clear()
        for run in outputs.runs:
            run_result = run.result
            self.detection_list.addItem(
                f"{run_result.reasoning_effort}: {run_result.count}/{len(run_result.candidates)} "
                f"candidate(s) accepted in {run_result.analysis_seconds:.1f}s"
            )
        self._show_judge_display()
        self.open_output_btn.setEnabled(True)
        self.open_annotated_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.statusBar().showMessage(f"Benchmark complete: {outputs.output_dir}", 8000)
        self._finish_job(ok=True, detail="Benchmark complete")

    def _set_preprocess_result(self, result: CrabPreprocessResult) -> None:
        self._last_preprocess_result = result
        self.open_preprocessed_btn.setEnabled(True)
        points = result.ordered_points or result.selection_points
        if points:
            self.preview.set_homography_points(points)
        self._update_preprocess_status()

    def _set_candidate_detection_result(self, result: CrabCandidateDetectionResult) -> None:
        projected = self._candidate_result_from_detection_result(result)
        self._last_candidate_projected_result = projected
        self._last_candidate_display_result = self._preview_result_for_current_run(projected)
        self._refresh_preview_display()

    def _candidate_result_from_detection_result(self, result: CrabCandidateDetectionResult) -> CrabCountResult:
        detections = tuple(
            CrabDetection(
                label=UNCERTAIN_CLASS,
                bbox=candidate.bbox,
                confidence=candidate.confidence,
                target_match_confidence=0.0,
                class_scores={class_name: 0.0 for class_name in CRAB_CLASS_NAMES},
                closest_non_target="native_rock_crab",
                decision_margin=0.0,
                accepted_as_target=False,
                notes=f"candidate {candidate.candidate_id}",
            )
            for candidate in result.candidates
        )
        return CrabCountResult(
            image_path=result.image_path,
            image_size=result.image_size,
            count=0,
            detections=(),
            candidates=detections,
            model=result.model,
            reasoning_effort=result.reasoning_effort,
            target_confidence_threshold=self.threshold_spin.value(),
            target_margin_threshold=self.margin_spin.value(),
            analysis_seconds=result.analysis_seconds,
            summary=result.summary,
        )

    def _show_judge_display(self) -> None:
        if self.preview_tabs.currentIndex() == JUDGE_DISPLAY_TAB_INDEX:
            self.preview.set_display_mode("judge")
            self._refresh_preview_display()
            return
        self.preview_tabs.setCurrentIndex(JUDGE_DISPLAY_TAB_INDEX)

    def _refresh_preview_display(self) -> None:
        preview_index = self.preview_tabs.currentIndex()
        projected_tab = preview_index == 2
        judge_tab = preview_index == JUDGE_DISPLAY_TAB_INDEX
        self.preview.set_selection_visible(not (projected_tab or judge_tab))
        if projected_tab:
            projected = self._last_projected_result or self._last_candidate_projected_result
            if projected is None:
                return
            image_path = self._last_preprocess_result.processed_image if self._last_preprocess_result else projected.image_path
            if self._last_projected_result is not None:
                self.preview.set_result(projected, image_path)
            else:
                self.preview.set_candidate_result(projected, image_path)
            return

        display = self._last_display_result or self._last_candidate_display_result
        if display is None:
            return
        if self._last_display_result is not None:
            self.preview.set_result(display, display.image_path)
        else:
            self.preview.set_candidate_result(display, display.image_path)

    def _benchmark_selected_run(self, outputs: CrabBenchmarkOutputs) -> CrabCounterOutputs:
        preferred = self.reasoning_effort_combo.currentText()
        for run in outputs.runs:
            if run.result.reasoning_effort == preferred:
                return run
        return outputs.runs[-1]

    def _preview_result_for_current_run(self, result: CrabCountResult) -> CrabCountResult:
        return self._preview_result_for_preprocess(result, self._last_preprocess_result)

    def _preview_result_for_preprocess(
        self,
        result: CrabCountResult,
        preprocess_result: CrabPreprocessResult | None,
    ) -> CrabCountResult:
        if preprocess_result is None:
            return result
        try:
            metadata = json.loads(preprocess_result.metadata_json.read_text(encoding="utf-8"))
            matrix = metadata["processed_to_source_matrix"]
            source_size_values = metadata["source_size"]
            source_size = (int(source_size_values[0]), int(source_size_values[1]))
            source_image = metadata.get("source_image") or preprocess_result.source_image
            return transform_crab_count_result(
                result,
                matrix,
                source_image_path=source_image,
                source_size=source_size,
            )
        except Exception as exc:
            self.statusBar().showMessage(f"Could not map boxes to original frame: {exc}", 8000)
            return result

    def _activate_sample_run(self, run_id: str) -> None:
        state = self._sample_run_states.get(run_id)
        if not state:
            return
        self._active_run_id = run_id
        preprocess_result = state.get("preprocess_result")
        self._last_preprocess_result = preprocess_result if isinstance(preprocess_result, CrabPreprocessResult) else None
        self.open_preprocessed_btn.setEnabled(self._last_preprocess_result is not None)
        if self._last_preprocess_result is not None:
            self.preview.set_homography_points(
                self._last_preprocess_result.ordered_points or self._last_preprocess_result.selection_points
            )
        else:
            self.preview.set_homography_points(())
        projected = state.get("projected_result")
        display = state.get("display_result")
        candidate_projected = state.get("candidate_projected_result")
        candidate_display = state.get("candidate_display_result")
        self._last_projected_result = projected if isinstance(projected, CrabCountResult) else None
        self._last_display_result = display if isinstance(display, CrabCountResult) else None
        self._last_candidate_projected_result = candidate_projected if isinstance(candidate_projected, CrabCountResult) else None
        self._last_candidate_display_result = candidate_display if isinstance(candidate_display, CrabCountResult) else None
        if self._last_display_result is not None:
            self._populate_detection_list(self._last_display_result)
        elif self._last_candidate_display_result is not None:
            self._populate_detection_list(self._last_candidate_display_result)
        else:
            self.detection_list.clear()
            source = state.get("source_image")
            if isinstance(source, Path) and source.is_file():
                self.preview.set_image(source)
        self._refresh_preview_display()

    def _store_sample_candidate_result(
        self,
        run_id: str,
        detection_result: CrabCandidateDetectionResult,
    ) -> None:
        state = self._ensure_sample_run_state(run_id)
        projected = self._candidate_result_from_detection_result(detection_result)
        preprocess_result = state.get("preprocess_result")
        display = self._preview_result_for_preprocess(
            projected,
            preprocess_result if isinstance(preprocess_result, CrabPreprocessResult) else None,
        )
        state["candidate_projected_result"] = projected
        state["candidate_display_result"] = display
        state["detail"] = f"{len(detection_result.candidates)} candidate(s) detected"
        self._update_sample_status_item(run_id)
        if self._active_run_id == run_id:
            self._activate_sample_run(run_id)

    def _store_sample_image_run(self, image_run: CrabEnsembleImageRun) -> None:
        state = self._ensure_sample_run_state(image_run.run_id, image_run.index)
        state["source_image"] = image_run.source_image
        state["preprocess_result"] = image_run.preprocess_result
        state["outputs"] = image_run.outputs
        state["projected_result"] = image_run.outputs.result
        state["display_result"] = self._preview_result_for_preprocess(image_run.outputs.result, image_run.preprocess_result)
        state["count"] = image_run.outputs.result.count
        state["status"] = "Done"
        state["detail"] = f"{image_run.outputs.result.count}/{len(image_run.outputs.result.candidates)} accepted"
        self._update_sample_status_item(image_run.run_id)
        if self._active_run_id == image_run.run_id:
            self._activate_sample_run(image_run.run_id)

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
        if index == JUDGE_DISPLAY_TAB_INDEX:
            mode = "judge"
        elif index in (1, 2):
            mode = "all"
        else:
            mode = "accepted"
        self.preview.set_display_mode(mode)
        self._refresh_preview_display()

    def _handle_sample_run_progress(self, data: dict[str, object]) -> None:
        run_id = str(data.get("run_id") or "")
        if not run_id:
            return
        index = int(data.get("index") or 0)
        state = self._ensure_sample_run_state(run_id, index)
        event = str(data.get("event") or "")
        if self._active_run_id is None:
            self._active_run_id = run_id
        if event in {"ensemble_image_started", "ensemble_image_pipeline_started"}:
            state["status"] = "Running"
            self._set_running_status(f"Sample {state.get('index')} started. Running image pipelines in parallel...")
        elif event == "auto_homography_started":
            state["status"] = "Finding board"
            effort = str(data.get("effort") or DEFAULT_HOMOGRAPHY_REASONING_EFFORT)
            self._set_running_status(f"Sample {state.get('index')} locating board corners ({effort} effort)...")
        elif event == "auto_homography_finished":
            preprocess_result = data.get("preprocess_result")
            if isinstance(preprocess_result, CrabPreprocessResult):
                state["preprocess_result"] = preprocess_result
                state["detail"] = f"board confidence {float(preprocess_result.board_confidence or 0.0):.2f}"
            state["status"] = "Board ready"
            if self._active_run_id == run_id:
                self._activate_sample_run(run_id)
        elif event == "candidate_detection_started":
            effort = str(data.get("effort") or "low")
            state["status"] = "Detecting"
            state["detail"] = f"{effort} effort"
            self._set_running_status(f"Sample {state.get('index')} detecting printed crab candidate boxes...")
        elif event == "candidate_detection_finished":
            detection_result = data.get("detection_result")
            if isinstance(detection_result, CrabCandidateDetectionResult):
                self._store_sample_candidate_result(run_id, detection_result)
            state["status"] = "Candidates ready"
            count = int(data.get("count") or 0)
            state["detail"] = f"{count} candidate box(es)"
            self._set_running_status(f"Sample {state.get('index')} detected {count} candidate box(es).")
        elif event == "candidate_classification_started":
            count = int(data.get("candidate_count") or 0)
            state["status"] = "Classifying"
            state["detail"] = f"{count} crop(s)"
            self._set_running_status(f"Sample {state.get('index')} classifying {count} candidate crop(s)...")
        elif event == "candidate_classification_finished":
            state["status"] = "Writing result"
            seconds = float(data.get("analysis_seconds") or 0.0)
            state["detail"] = f"classifier {seconds:.1f}s"
        elif event == "ensemble_image_finished":
            image_run = data.get("image_run")
            if isinstance(image_run, CrabEnsembleImageRun):
                self._store_sample_image_run(image_run)
            completed = int(data.get("completed") or 0)
            total = int(data.get("total") or max(1, self.sample_status_list.count()))
            count = int(data.get("count") or state.get("count") or 0)
            candidates = int(data.get("candidate_count") or 0)
            state["status"] = "Done"
            state["count"] = count
            state["detail"] = f"{count}/{candidates} accepted"
            self.progress_bar.setRange(0, max(1, total))
            self.progress_bar.setValue(max(0, completed))
            self.progress_bar.setFormat(f"{completed}/{total} images")
            self._set_running_status(
                f"Sample {state.get('index')} finished with {count}/{candidates} accepted candidate(s)."
            )
        elif event == "ensemble_image_failed":
            state["status"] = "Failed"
            state["detail"] = str(data.get("error") or "unknown error")
            self.progress_bar.setFormat("Image failed")
            self._set_running_status(f"Sample {state.get('index')} failed: {state['detail']}")
        self._update_sample_status_item(run_id)
        if self._active_run_id == run_id:
            self._activate_sample_run(run_id)

    def _handle_worker_progress(self, payload: object) -> None:
        data = payload if isinstance(payload, dict) else {}
        event = str(data.get("event") or "")
        if data.get("run_id"):
            self._handle_sample_run_progress(data)
            return
        if event == "auto_homography_started":
            effort = str(data.get("effort") or DEFAULT_HOMOGRAPHY_REASONING_EFFORT)
            self.progress_bar.setRange(0, 0)
            self.progress_bar.setFormat("Finding board")
            self._set_running_status(f"Locating board corners with OpenAI ({effort} effort)...")
            return
        if event == "auto_homography_finished":
            preprocess_result = data.get("preprocess_result")
            if isinstance(preprocess_result, CrabPreprocessResult):
                self._set_preprocess_result(preprocess_result)
            points = data.get("points")
            if not isinstance(preprocess_result, CrabPreprocessResult):
                self.preview.set_homography_points(points)
            confidence = data.get("confidence")
            seconds = float(data.get("seconds") or 0.0)
            confidence_text = f"{float(confidence):.2f}" if confidence is not None else "unknown"
            self.progress_bar.setRange(0, 0)
            self.progress_bar.setFormat("Classifying")
            self._set_running_status(
                f"Board outline found in {seconds:.1f}s (confidence {confidence_text}). Sending rectified board to classifier..."
            )
            return
        if event == "auto_homography_failed":
            error = str(data.get("error") or "unknown error")
            self.progress_bar.setRange(0, 1)
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat("Homography failed")
            self._set_running_status(
                f"Auto homography could not find a usable board outline ({error}). Run stopped before classification."
            )
            return
        if event == "candidate_detection_started":
            effort = str(data.get("effort") or "low")
            self.progress_bar.setRange(0, 0)
            self.progress_bar.setFormat("Detecting")
            self._set_running_status(f"Detecting printed crab candidate boxes ({effort} effort)...")
            return
        if event == "candidate_detection_finished":
            detection_result = data.get("detection_result")
            if isinstance(detection_result, CrabCandidateDetectionResult):
                self._set_candidate_detection_result(detection_result)
            count = int(data.get("count") or 0)
            seconds = float(data.get("analysis_seconds") or 0.0)
            self.progress_bar.setRange(0, 0)
            self.progress_bar.setFormat("Classifying")
            self._set_running_status(f"Detected {count} candidate box(es) in {seconds:.1f}s. Building crop sheet...")
            return
        if event == "candidate_classification_started":
            effort = str(data.get("effort") or self.reasoning_effort_combo.currentText())
            count = int(data.get("candidate_count") or 0)
            self.progress_bar.setRange(0, 0)
            self.progress_bar.setFormat("Classifying")
            self._set_running_status(f"Classifying {count} candidate crop(s) ({effort} effort). Waiting for model response...")
            return
        if event == "candidate_classification_finished":
            seconds = float(data.get("analysis_seconds") or 0.0)
            self._set_running_status(f"Crop classifier finished in {seconds:.1f}s. Writing results...")
            return
        if event == "request_started":
            effort = str(data.get("effort") or self.reasoning_effort_combo.currentText())
            self._set_running_status(f"OpenAI request sent ({effort} effort). Waiting for model response...")
            return
        if event == "ensemble_started":
            total = int(data.get("total") or 3)
            workers = int(data.get("max_workers") or total)
            self.progress_bar.setRange(0, max(1, total))
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat(f"0/{total} images")
            self._set_running_status(f"Running {total} image pipelines with up to {workers} parallel worker(s)...")
            return
        if event == "ensemble_image_started":
            index = int(data.get("index") or 0)
            total = int(data.get("total") or 3)
            run_id = str(data.get("run_id") or f"image_{index}")
            self.progress_bar.setRange(0, max(1, total))
            self._set_running_status(f"{run_id} started ({index}/{total}). Waiting for image pipelines...")
            return
        if event == "ensemble_image_finished":
            completed = int(data.get("completed") or 0)
            total = int(data.get("total") or 3)
            run_id = str(data.get("run_id") or "image")
            count = int(data.get("count") or 0)
            candidates = int(data.get("candidate_count") or 0)
            self.progress_bar.setRange(0, max(1, total))
            self.progress_bar.setValue(max(0, completed))
            self.progress_bar.setFormat(f"{completed}/{total} images")
            self._set_running_status(
                f"{run_id} finished with {count}/{candidates} accepted candidate(s). Waiting for remaining images..."
            )
            return
        if event == "ensemble_image_failed":
            run_id = str(data.get("run_id") or "image")
            error = str(data.get("error") or "unknown error")
            self.progress_bar.setFormat("Image failed")
            self._set_running_status(f"{run_id} failed before ensemble validation: {error}")
            return
        if event == "ensemble_validation_started":
            effort = str(data.get("effort") or self.reasoning_effort_combo.currentText())
            self.progress_bar.setRange(0, 0)
            self.progress_bar.setFormat("Validating")
            self._set_running_status(f"All image pipelines finished. Running ensemble sensibility check ({effort} effort)...")
            return
        if event == "ensemble_validation_finished":
            run_id = str(data.get("selected_run_id") or "image")
            confidence = float(data.get("confidence") or 0.0)
            seconds = float(data.get("analysis_seconds") or 0.0)
            self._set_running_status(
                f"Ensemble validator selected {run_id} in {seconds:.1f}s (confidence {confidence:.2f}). Writing final outputs..."
            )
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
        self._report_progress(message)
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

    def _open_preprocessed_image(self) -> None:
        if self._last_preprocess_result is None:
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._last_preprocess_result.processed_image)))

    def _open_invasive_species_form(self) -> None:
        if QDesktopServices.openUrl(QUrl(INVASIVE_SPECIES_FORM_URL)):
            self.statusBar().showMessage("Opened invasive species reporting form.", 5000)
        else:
            self.status_label.setText(f"Could not open invasive species form: {INVASIVE_SPECIES_FORM_URL}")
