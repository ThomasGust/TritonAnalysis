"""GUI wrapper for the Triton stereo RealityScan reconstruction pipeline."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

from PyQt6.QtCore import QProcess, QProcessEnvironment, Qt, QTimer, QUrl
from PyQt6.QtGui import QDesktopServices, QTextCursor
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from triton_analysis.gui.file_dialogs import ThumbnailFileDialog as QFileDialog

from triton_analysis.gui.realityscan_model_viewer_window import RealityScanModelViewerPanel
from triton_analysis.gui.responsive import resize_to_available_screen, vertical_scroll_area
from triton_analysis.workspace import fresh_output_subdir, latest_pilot_stereo_sessions_dir, safe_output_slug, workspace_paths


REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_MODULE = "triton_analysis.realityscan.underwater_pipeline"
PIPELINE_PACKAGE_PATH = Path("triton_analysis") / "realityscan" / "underwater_pipeline.py"
PRESETS = ("balanced", "high-detail", "max-detail")
DEFAULT_MIN_GOOD_COMPONENT_RATIO = 0.12
FAST_VARIANTS = (
    ("Flat Luma K+", "flat_luma_kplus"),
    ("Caustic Stable Luma K+", "caustic_luma"),
    ("Legacy Enhanced Brown4", "legacy_enhanced"),
)


def default_results_dir(*, create: bool = False) -> Path:
    path = workspace_paths(create=create).realityscan_results
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def find_default_pipeline_root() -> Path:
    """Return the most likely checkout that contains the current pipeline."""
    candidates: list[Path] = []
    for env_name in ("TRITON_ANALYSIS_ROOT", "TRITON_REALITYSCAN_PIPELINE_ROOT"):
        value = os.environ.get(env_name, "").strip()
        if value:
            candidates.append(Path(value).expanduser())
    candidates.extend(
        [
            REPO_ROOT,
            REPO_ROOT.parent / "TritonAnalysis",
        ]
    )
    for candidate in candidates:
        if (candidate / PIPELINE_PACKAGE_PATH).exists():
            return candidate.resolve()
    return REPO_ROOT.resolve()


def find_realityscan_exe() -> Path | None:
    """Locate RealityScan/RealityCapture without importing the pipeline module."""
    env_path = os.environ.get("REALITYSCAN_EXE", "").strip()
    if env_path:
        candidate = Path(env_path).expanduser()
        if candidate.exists():
            return candidate.resolve()

    patterns = [
        r"C:\Program Files\Epic Games\RealityScan*\RealityScan.exe",
        r"C:\Program Files\Capturing Reality\RealityScan*\RealityScan.exe",
        r"C:\Program Files\Capturing Reality\RealityCapture*\RealityCapture.exe",
    ]
    from glob import glob
    import shutil

    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(Path(item) for item in glob(pattern))
    candidates = [path for path in candidates if path.exists()]
    if candidates:
        return sorted(candidates, key=lambda path: path.parent.name, reverse=True)[0].resolve()

    found = shutil.which("RealityScan.exe") or shutil.which("RealityCapture.exe")
    return Path(found).resolve() if found else None


def stereo_manifest_path(path: Path) -> Path | None:
    """Return the manifest path if *path* looks like a TritonPilot stereo session."""
    candidate = Path(path).expanduser()
    if candidate.is_dir():
        manifest = candidate / "manifest.json"
        return manifest if manifest.exists() else None
    if candidate.is_file() and candidate.name.lower() == "manifest.json":
        return candidate
    return None


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


class RealityScanReconstructionWindow(QMainWindow):
    """Run and monitor the stereo photogrammetry pipeline from TritonAnalysis."""

    def __init__(self, session_path: str | None = None, calibration_path: str | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("RealityScan Stereo Reconstruction")
        self._process: QProcess | None = None
        self._line_buffer = ""
        self._workspace_path: Path | None = None
        self._output_paths: dict[str, Path] = {}
        self._output_labels: dict[str, QLabel] = {}
        self._output_buttons: dict[str, QPushButton] = {}

        self._build_ui()
        if session_path:
            self.session_edit.setText(str(Path(session_path)))
            self._autoload_session_calibration()
        if calibration_path:
            self.calibration_edit.setText(str(Path(calibration_path)))
        self._refresh_command_preview()
        self._set_running(False)
        self._set_status("Ready", "idle")
        resize_to_available_screen(self, 1500, 900, min_width=1040, min_height=680)

    def closeEvent(self, event) -> None:
        self.shutdown()
        super().closeEvent(event)

    def shutdown(self) -> None:
        if self._process is not None and self._process.state() != QProcess.ProcessState.NotRunning:
            self._process.terminate()
            if not self._process.waitForFinished(1500):
                self._process.kill()
                self._process.waitForFinished(1500)
        if hasattr(self, "model_viewer_panel"):
            self.model_viewer_panel.shutdown()

    def _build_ui(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)
        self.setCentralWidget(central)

        self.tabs = QTabWidget()
        root.addWidget(self.tabs, 1)

        reconstruction_tab = QWidget()
        reconstruction_layout = QVBoxLayout(reconstruction_tab)
        reconstruction_layout.setContentsMargins(0, 0, 0, 0)
        reconstruction_layout.setSpacing(8)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        reconstruction_layout.addWidget(splitter, 1)

        controls_panel = QWidget()
        controls_panel.setMinimumWidth(420)
        controls_panel.setMaximumWidth(560)
        controls_layout = QVBoxLayout(controls_panel)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(8)
        splitter.addWidget(vertical_scroll_area(controls_panel))

        title = QLabel("Stereo Photogrammetry")
        title.setStyleSheet("font-size: 18px; font-weight: 700;")
        controls_layout.addWidget(title)

        input_card = _SectionCard("Inputs")
        self.pipeline_root_edit = QLineEdit(str(find_default_pipeline_root()))
        self.pipeline_root_edit.setClearButtonEnabled(True)
        self.pipeline_root_edit.textChanged.connect(self._refresh_command_preview)
        pipeline_row = QHBoxLayout()
        pipeline_row.addWidget(self.pipeline_root_edit, 1)
        self.pipeline_browse_btn = QPushButton("Browse...")
        self.pipeline_browse_btn.clicked.connect(self._choose_pipeline_root)
        pipeline_row.addWidget(self.pipeline_browse_btn)
        input_card.body.addWidget(QLabel("Pipeline Root"))
        input_card.body.addLayout(pipeline_row)

        self.session_edit = QLineEdit()
        self.session_edit.setPlaceholderText("TritonPilot stereo session folder or manifest.json")
        self.session_edit.setClearButtonEnabled(True)
        self.session_edit.textChanged.connect(self._on_session_changed)
        session_row = QHBoxLayout()
        session_row.addWidget(self.session_edit, 1)
        self.session_browse_btn = QPushButton("Session...")
        self.session_browse_btn.clicked.connect(self._choose_session_dir)
        self.manifest_browse_btn = QPushButton("Manifest...")
        self.manifest_browse_btn.clicked.connect(self._choose_manifest)
        session_row.addWidget(self.session_browse_btn)
        session_row.addWidget(self.manifest_browse_btn)
        input_card.body.addWidget(QLabel("Stereo Session"))
        input_card.body.addLayout(session_row)

        self.calibration_edit = QLineEdit()
        self.calibration_edit.setPlaceholderText("stereo_calibration.json")
        self.calibration_edit.setClearButtonEnabled(True)
        self.calibration_edit.textChanged.connect(self._refresh_command_preview)
        calibration_row = QHBoxLayout()
        calibration_row.addWidget(self.calibration_edit, 1)
        self.calibration_browse_btn = QPushButton("Browse...")
        self.calibration_browse_btn.clicked.connect(self._choose_calibration)
        self.calibration_find_btn = QPushButton("Find Recent")
        self.calibration_find_btn.clicked.connect(self._find_recent_calibration)
        calibration_row.addWidget(self.calibration_browse_btn)
        calibration_row.addWidget(self.calibration_find_btn)
        input_card.body.addWidget(QLabel("Stereo Calibration"))
        input_card.body.addLayout(calibration_row)

        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText(str(default_results_dir() / "<new subfolder each run>"))
        self.output_edit.setClearButtonEnabled(True)
        self.output_edit.textChanged.connect(self._refresh_command_preview)
        output_row = QHBoxLayout()
        output_row.addWidget(self.output_edit, 1)
        self.output_browse_btn = QPushButton("Output...")
        self.output_browse_btn.clicked.connect(self._choose_output_dir)
        output_row.addWidget(self.output_browse_btn)
        input_card.body.addWidget(QLabel("Output Workspace"))
        input_card.body.addLayout(output_row)

        self.realityscan_edit = QLineEdit()
        self.realityscan_edit.setPlaceholderText("auto detect")
        self.realityscan_edit.setClearButtonEnabled(True)
        self.realityscan_edit.textChanged.connect(self._refresh_command_preview)
        realityscan_row = QHBoxLayout()
        realityscan_row.addWidget(self.realityscan_edit, 1)
        self.realityscan_browse_btn = QPushButton("Browse...")
        self.realityscan_browse_btn.clicked.connect(self._choose_realityscan)
        self.realityscan_detect_btn = QPushButton("Detect")
        self.realityscan_detect_btn.clicked.connect(self._detect_realityscan)
        realityscan_row.addWidget(self.realityscan_browse_btn)
        realityscan_row.addWidget(self.realityscan_detect_btn)
        input_card.body.addWidget(QLabel("RealityScan"))
        input_card.body.addLayout(realityscan_row)
        controls_layout.addWidget(input_card)

        run_card = _SectionCard("Configuration")
        config_grid = QGridLayout()
        config_grid.setHorizontalSpacing(8)
        config_grid.setVerticalSpacing(6)
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(PRESETS)
        self.preset_combo.setCurrentText("max-detail")
        self.preset_combo.currentIndexChanged.connect(self._refresh_command_preview)
        self.alignment_combo = QComboBox()
        for label, value in (("Off", "off"), ("Standard", "standard"), ("Thorough", "thorough")):
            self.alignment_combo.addItem(label, value)
        self.alignment_combo.currentIndexChanged.connect(self._on_alignment_changed)
        self.fast_variant_combo = QComboBox()
        for label, value in FAST_VARIANTS:
            self.fast_variant_combo.addItem(label, value)
        self.fast_variant_combo.currentIndexChanged.connect(self._refresh_command_preview)
        config_grid.addWidget(QLabel("Preset"), 0, 0)
        config_grid.addWidget(self.preset_combo, 0, 1)
        config_grid.addWidget(QLabel("Alignment Tournament"), 1, 0)
        config_grid.addWidget(self.alignment_combo, 1, 1)
        config_grid.addWidget(QLabel("Fast Variant"), 2, 0)
        config_grid.addWidget(self.fast_variant_combo, 2, 1)
        run_card.body.addLayout(config_grid)

        flags_grid = QGridLayout()
        flags_grid.setHorizontalSpacing(6)
        flags_grid.setVerticalSpacing(4)
        self.metric_scale_check = QCheckBox("Metric Scale From Stereo")
        self.metric_scale_check.setChecked(True)
        self.metric_required_check = QCheckBox("Metric Scale Required")
        self.metric_required_check.setChecked(True)
        self.xmp_priors_check = QCheckBox("Stereo XMP Priors")
        self.xmp_priors_check.setChecked(True)
        self.texture_layers_check = QCheckBox("Color Texture Layers")
        self.texture_layers_check.setChecked(True)
        self.bridge_frames_check = QCheckBox("Bridge Frames")
        self.bridge_frames_check.setChecked(True)
        self.rig_priors_check = QCheckBox("Rig Priors")
        self.prepare_only_check = QCheckBox("Prepare Only")
        self.alignment_only_check = QCheckBox("Alignment Only")
        self.overwrite_check = QCheckBox("Overwrite Output")
        self.large_face_filter_check = QCheckBox("Large Face Filter")
        self.large_face_filter_check.setChecked(True)
        self.clean_model_check = QCheckBox("Clean Model")
        self.connectivity_report_check = QCheckBox("Connectivity Report")
        self.connectivity_report_check.setChecked(True)
        self.component_diagnostics_check = QCheckBox("Component Diagnostics")
        self.try_merge_check = QCheckBox("Try Merge Components")
        self.fail_poor_alignment_check = QCheckBox("Fail On Poor Alignment")
        self.fail_poor_alignment_check.setChecked(True)
        flags = [
            self.metric_scale_check,
            self.metric_required_check,
            self.xmp_priors_check,
            self.texture_layers_check,
            self.bridge_frames_check,
            self.rig_priors_check,
            self.prepare_only_check,
            self.alignment_only_check,
            self.overwrite_check,
            self.large_face_filter_check,
            self.clean_model_check,
            self.connectivity_report_check,
            self.component_diagnostics_check,
            self.try_merge_check,
            self.fail_poor_alignment_check,
        ]
        for index, checkbox in enumerate(flags):
            flags_grid.addWidget(checkbox, index // 2, index % 2)
            checkbox.toggled.connect(self._refresh_command_preview)
        self.alignment_only_check.toggled.connect(self._ensure_alignment_for_alignment_only)
        run_card.body.addLayout(flags_grid)
        controls_layout.addWidget(run_card)

        advanced_card = _SectionCard("Advanced")
        self.advanced_group = QGroupBox("Override Individual CLI Settings")
        self.advanced_group.setCheckable(True)
        self.advanced_group.setChecked(False)
        self.advanced_group.toggled.connect(self._refresh_command_preview)
        advanced_grid = QGridLayout(self.advanced_group)
        advanced_grid.setHorizontalSpacing(8)
        advanced_grid.setVerticalSpacing(6)

        self.target_fps_spin = self._double_spin(0.1, 60.0, 1, 0.5, 8.0, " fps")
        self.max_frames_spin = self._int_spin(1, 5000, 420, 10)
        self.min_frames_spin = self._int_spin(1, 5000, 120, 10)
        self.quality_quantile_spin = self._double_spin(0.0, 0.9, 3, 0.01, 0.05, "")
        self.crop_fraction_spin = self._double_spin(0.0, 0.5, 3, 0.005, 0.04, "")
        self.pair_delta_spin = self._double_spin(0.0, 1000.0, 1, 5.0, 75.0, " ms")
        self.wb_gain_spin = self._double_spin(0.1, 10.0, 2, 0.1, 2.4, "")
        self.clahe_spin = self._double_spin(0.1, 10.0, 2, 0.1, 2.0, "")
        self.sharpen_spin = self._double_spin(0.0, 5.0, 2, 0.05, 0.22, "")
        self.features_mpx_spin = self._int_spin(1000, 200000, 20000, 1000)
        self.features_image_spin = self._int_spin(1000, 500000, 80000, 5000)
        self.preselector_spin = self._int_spin(1000, 200000, 20000, 1000)
        self.texture_count_spin = self._int_spin(1, 32, 4, 1)
        self.texture_resolution_spin = self._int_spin(512, 16384, 4096, 512)
        self.normal_downscale_spin = self._int_spin(1, 8, 2, 1)
        self.simplify_spin = self._int_spin(0, 10000000, 1500000, 100000)
        self.timeout_spin = self._double_spin(0.1, 72.0, 1, 0.5, 8.0, " h")
        self.metric_min_pairs_spin = self._int_spin(1, 500, 3, 1)
        self.min_component_ratio_spin = self._double_spin(0.01, 1.0, 2, 0.01, DEFAULT_MIN_GOOD_COMPONENT_RATIO, "")

        self.model_quality_combo = self._combo(("preview", "normal", "high"), "normal")
        self.detector_combo = self._combo(("Low", "Medium", "High", "Ultra"), "Ultra")
        self.overlap_combo = self._combo(("Low", "Medium", "High"), "Low")
        self.pose_prior_combo = self._combo(("initial", "exact", "locked"), "exact")
        self.calibration_prior_combo = self._combo(("initial", "exact", "locked"), "exact")

        rows = [
            ("Target FPS", self.target_fps_spin),
            ("Max Frames", self.max_frames_spin),
            ("Min Frames", self.min_frames_spin),
            ("Quality Quantile", self.quality_quantile_spin),
            ("Crop Fraction", self.crop_fraction_spin),
            ("Pair Delta", self.pair_delta_spin),
            ("WB Gain", self.wb_gain_spin),
            ("CLAHE", self.clahe_spin),
            ("Sharpen", self.sharpen_spin),
            ("Features / MPx", self.features_mpx_spin),
            ("Features / Image", self.features_image_spin),
            ("Preselector", self.preselector_spin),
            ("Texture Count", self.texture_count_spin),
            ("Texture Resolution", self.texture_resolution_spin),
            ("Normal Downscale", self.normal_downscale_spin),
            ("Simplify Tris", self.simplify_spin),
            ("Timeout", self.timeout_spin),
            ("Metric Min Pairs", self.metric_min_pairs_spin),
            ("Min Component Ratio", self.min_component_ratio_spin),
            ("Model Quality", self.model_quality_combo),
            ("Detector", self.detector_combo),
            ("Overlap", self.overlap_combo),
            ("Pose Prior", self.pose_prior_combo),
            ("Calibration Prior", self.calibration_prior_combo),
        ]
        for index, (label, widget) in enumerate(rows):
            row = index // 2
            col = (index % 2) * 2
            advanced_grid.addWidget(QLabel(label), row, col)
            advanced_grid.addWidget(widget, row, col + 1)
        advanced_card.body.addWidget(self.advanced_group)
        controls_layout.addWidget(advanced_card)

        command_card = _SectionCard("Command Preview")
        self.command_preview = QPlainTextEdit()
        self.command_preview.setReadOnly(True)
        self.command_preview.setMaximumBlockCount(24)
        self.command_preview.setMinimumHeight(95)
        self.command_preview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        command_card.body.addWidget(self.command_preview)
        copy_row = QHBoxLayout()
        self.copy_command_btn = QPushButton("Copy Command")
        self.copy_command_btn.clicked.connect(self._copy_command)
        copy_row.addWidget(self.copy_command_btn)
        copy_row.addStretch(1)
        command_card.body.addLayout(copy_row)
        controls_layout.addWidget(command_card)
        controls_layout.addStretch(1)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([500, 1000])

        status_card = _SectionCard("Run Status")
        status_row = QHBoxLayout()
        self.status_label = QLabel("Ready")
        self.status_label.setObjectName("summaryCard")
        self.stage_label = QLabel("-")
        self.stage_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.stage_label.setWordWrap(True)
        status_row.addWidget(self.status_label, 0)
        status_row.addWidget(self.stage_label, 1)
        status_card.body.addLayout(status_row)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        status_card.body.addWidget(self.progress_bar)
        run_row = QHBoxLayout()
        self.run_btn = QPushButton("Run Reconstruction")
        self.run_btn.clicked.connect(self._start_pipeline)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self._cancel_pipeline)
        self.clear_log_btn = QPushButton("Clear Log")
        self.clear_log_btn.clicked.connect(lambda: self.log_text.clear())
        run_row.addWidget(self.run_btn)
        run_row.addWidget(self.cancel_btn)
        run_row.addWidget(self.clear_log_btn)
        run_row.addStretch(1)
        status_card.body.addLayout(run_row)
        right_layout.addWidget(status_card, 0)

        outputs_card = _SectionCard("Outputs")
        outputs_grid = QGridLayout()
        outputs_grid.setHorizontalSpacing(8)
        outputs_grid.setVerticalSpacing(5)
        for row, (key, label) in enumerate(
            [
                ("workspace", "Workspace"),
                ("contact_sheet", "Contact Sheet"),
                ("report", "Report"),
                ("components", "Components"),
                ("model", "OBJ"),
                ("metric_model", "Metric OBJ"),
                ("metric_scale", "Metric Scale"),
                ("manifest", "Manifest"),
            ]
        ):
            value = QLabel("-")
            value.setWordWrap(True)
            value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            button = QPushButton("Open")
            button.setEnabled(False)
            button.clicked.connect(lambda _checked=False, output_key=key: self._open_output(output_key))
            self._output_labels[key] = value
            self._output_buttons[key] = button
            outputs_grid.addWidget(QLabel(label), row, 0)
            outputs_grid.addWidget(value, row, 1)
            outputs_grid.addWidget(button, row, 2)
        outputs_card.body.addLayout(outputs_grid)
        view_row = QHBoxLayout()
        self.view_metric_model_btn = QPushButton("View Metric Model")
        self.view_metric_model_btn.clicked.connect(self._launch_model_viewer)
        self.view_metric_model_btn.setEnabled(False)
        view_row.addWidget(self.view_metric_model_btn)
        view_row.addStretch(1)
        outputs_card.body.addLayout(view_row)
        right_layout.addWidget(outputs_card, 0)

        log_card = _SectionCard("Live Log")
        self.log_text = QPlainTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumBlockCount(7000)
        self.log_text.setStyleSheet("font-family: Consolas, Menlo, monospace;")
        log_card.body.addWidget(self.log_text, 1)
        right_layout.addWidget(log_card, 1)

        self.tabs.addTab(reconstruction_tab, "Reconstruction")
        self.model_viewer_panel = RealityScanModelViewerPanel()
        self.tabs.addTab(self.model_viewer_panel, "Model Viewer")

    def _double_spin(
        self,
        minimum: float,
        maximum: float,
        decimals: int,
        step: float,
        value: float,
        suffix: str,
    ) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(decimals)
        spin.setSingleStep(step)
        spin.setValue(value)
        if suffix:
            spin.setSuffix(suffix)
        spin.valueChanged.connect(self._refresh_command_preview)
        return spin

    def _int_spin(self, minimum: int, maximum: int, value: int, step: int) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setSingleStep(step)
        spin.setValue(value)
        spin.valueChanged.connect(self._refresh_command_preview)
        return spin

    def _combo(self, values: tuple[str, ...], current: str) -> QComboBox:
        combo = QComboBox()
        combo.addItems(values)
        combo.setCurrentText(current)
        combo.currentIndexChanged.connect(self._refresh_command_preview)
        return combo

    def _clean_path_text(self, text: str) -> str:
        return str(text or "").strip().strip('"')

    def _path_arg(self, text: str) -> str:
        raw = self._clean_path_text(text)
        if not raw:
            return ""
        path = Path(raw).expanduser()
        try:
            if path.exists():
                return str(path.resolve())
        except Exception:
            pass
        return str(path)

    def _combo_value(self, combo: QComboBox) -> str:
        data = combo.currentData()
        return str(data if data is not None else combo.currentText())

    def _pipeline_root(self) -> Path:
        text = self._clean_path_text(self.pipeline_root_edit.text())
        return Path(text).expanduser() if text else find_default_pipeline_root()

    def _output_workspace_for_command(self, *, preview: bool = False) -> str:
        explicit = self._path_arg(self.output_edit.text())
        if explicit:
            return explicit
        results_dir = default_results_dir(create=not preview)
        if preview:
            return str(results_dir / "<new subfolder each run>")
        session = self._clean_path_text(self.session_edit.text())
        stem = Path(session).parent.name if Path(session).name.lower() == "manifest.json" else Path(session).name
        return str(fresh_output_subdir(results_dir, safe_output_slug(stem, fallback="scan")).resolve())

    def build_command(self, *, preview: bool = False, output_override: str | None = None) -> list[str]:
        session = self._path_arg(self.session_edit.text()) or ("<stereo-session>" if preview else "")
        command = [sys.executable, "-u", "-m", PIPELINE_MODULE, session]
        output_path = output_override if output_override is not None else self._output_workspace_for_command(preview=preview)
        if output_path:
            command.extend(["--output", output_path])

        calibration = self._path_arg(self.calibration_edit.text())
        if calibration:
            command.extend(["--stereo-calibration", calibration])

        command.extend(["--reconstruction-preset", self.preset_combo.currentText()])
        fast_variant = self._combo_value(self.fast_variant_combo)
        if fast_variant == "legacy_enhanced":
            command.append("--legacy-enhanced-default")
        elif fast_variant == "caustic_luma":
            command.extend(["--base-geometry-mode", "caustic_luma"])
        alignment = self._combo_value(self.alignment_combo)
        if alignment != "off":
            command.extend(["--alignment-tournament", alignment])
        if self.prepare_only_check.isChecked():
            command.append("--prepare-only")
        if self.alignment_only_check.isChecked():
            command.append("--alignment-only")
        command.append("--metric-scale-from-stereo" if self.metric_scale_check.isChecked() else "--no-metric-scale-from-stereo")
        if self.metric_required_check.isChecked():
            command.append("--metric-scale-required")
        command.append("--stereo-xmp-priors" if self.xmp_priors_check.isChecked() else "--no-stereo-xmp-priors")
        command.append("--texture-layers" if self.texture_layers_check.isChecked() else "--no-texture-layers")
        if not self.bridge_frames_check.isChecked():
            command.append("--no-connectivity-bridge-selection")
        if self.rig_priors_check.isChecked():
            command.append("--stereo-xmp-rig-priors")
        if self.overwrite_check.isChecked():
            command.append("--overwrite")
        if not self.large_face_filter_check.isChecked():
            command.append("--no-mesh-large-face-filter")
        if self.clean_model_check.isChecked():
            command.append("--clean-model")
        if self.connectivity_report_check.isChecked():
            command.append("--connectivity-report")
        if self.component_diagnostics_check.isChecked():
            command.append("--export-component-diagnostics")
        if self.try_merge_check.isChecked():
            command.append("--try-merge-components")
        if self.fail_poor_alignment_check.isChecked():
            command.extend(["--min-good-component-ratio", f"{self.min_component_ratio_spin.value():g}"])
            command.append("--fail-on-poor-alignment")

        realityscan = self._path_arg(self.realityscan_edit.text())
        if realityscan:
            command.extend(["--realityscan-exe", realityscan])

        if self.advanced_group.isChecked():
            command.extend(
                [
                    "--target-fps",
                    f"{self.target_fps_spin.value():g}",
                    "--max-frames",
                    str(self.max_frames_spin.value()),
                    "--min-frames",
                    str(self.min_frames_spin.value()),
                    "--quality-quantile",
                    f"{self.quality_quantile_spin.value():g}",
                    "--crop-fraction",
                    f"{self.crop_fraction_spin.value():g}",
                    "--stereo-max-pair-delta-ms",
                    f"{self.pair_delta_spin.value():g}",
                    "--wb-gain",
                    f"{self.wb_gain_spin.value():g}",
                    "--clahe-clip",
                    f"{self.clahe_spin.value():g}",
                    "--sharpen",
                    f"{self.sharpen_spin.value():g}",
                    "--max-features-per-mpx",
                    str(self.features_mpx_spin.value()),
                    "--max-features-per-image",
                    str(self.features_image_spin.value()),
                    "--preselector-features",
                    str(self.preselector_spin.value()),
                    "--texture-count",
                    str(self.texture_count_spin.value()),
                    "--texture-resolution",
                    str(self.texture_resolution_spin.value()),
                    "--normal-downscale",
                    str(self.normal_downscale_spin.value()),
                    "--simplify-triangles",
                    str(self.simplify_spin.value()),
                    "--timeout-hours",
                    f"{self.timeout_spin.value():g}",
                    "--metric-scale-min-pairs",
                    str(self.metric_min_pairs_spin.value()),
                    "--model-quality",
                    self.model_quality_combo.currentText(),
                    "--detector-sensitivity",
                    self.detector_combo.currentText(),
                    "--images-overlap",
                    self.overlap_combo.currentText(),
                    "--stereo-xmp-pose-prior",
                    self.pose_prior_combo.currentText(),
                    "--stereo-xmp-calibration-prior",
                    self.calibration_prior_combo.currentText(),
                ]
            )
        return [part for part in command if part]

    def command_preview_text(self) -> str:
        return subprocess.list2cmdline(self.build_command(preview=True))

    def _refresh_command_preview(self, *_args) -> None:
        if hasattr(self, "command_preview"):
            self.command_preview.setPlainText(self.command_preview_text())

    def _on_session_changed(self, *_args) -> None:
        self._autoload_session_calibration()
        self._refresh_command_preview()

    def _choose_pipeline_root(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, "Select pipeline checkout root", str(self._pipeline_root()))
        if selected:
            self.pipeline_root_edit.setText(str(Path(selected)))

    def _session_dialog_start(self) -> Path:
        session_text = self._clean_path_text(self.session_edit.text())
        if session_text:
            path = Path(session_text).expanduser()
            if path.exists():
                return path.parent if path.is_file() else path
        workspace = workspace_paths(create=True)
        synced_sessions = latest_pilot_stereo_sessions_dir(create=True)
        if synced_sessions.exists():
            return synced_sessions
        if workspace.pilot_incoming.exists():
            return workspace.pilot_incoming
        pilot_sessions = self._pipeline_root() / "recordings" / "stereo_sessions"
        return pilot_sessions if pilot_sessions.exists() else REPO_ROOT

    def _choose_session_dir(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, "Select stereo session", str(self._session_dialog_start()))
        if selected:
            self.session_edit.setText(str(Path(selected)))

    def _choose_manifest(self) -> None:
        path, _filter = QFileDialog.getOpenFileName(
            self,
            "Select stereo manifest",
            str(self._session_dialog_start()),
            "Stereo manifest (manifest.json);;JSON files (*.json);;All files (*)",
        )
        if path:
            self.session_edit.setText(str(Path(path)))

    def _choose_calibration(self) -> None:
        start = self._calibration_dialog_start()
        path, _filter = QFileDialog.getOpenFileName(
            self,
            "Select stereo calibration",
            str(start),
            "Stereo calibration (*.json);;JSON files (*.json);;All files (*)",
        )
        if path:
            self.calibration_edit.setText(str(Path(path)))

    def _calibration_dialog_start(self) -> Path:
        candidate = self._default_calibration_candidate()
        if candidate is not None and candidate.exists():
            return candidate.parent
        calibration_dir = workspace_paths(create=True).calibrations
        return calibration_dir if calibration_dir.exists() else self._session_dialog_start()

    def _choose_output_dir(self) -> None:
        start = Path(self._output_workspace_for_command(preview=False)).parent
        selected = QFileDialog.getExistingDirectory(self, "Select output workspace", str(start))
        if selected:
            self.output_edit.setText(str(Path(selected)))

    def _choose_realityscan(self) -> None:
        start = Path(os.environ.get("ProgramFiles", "C:\\Program Files"))
        path, _filter = QFileDialog.getOpenFileName(
            self,
            "Select RealityScan executable",
            str(start),
            "RealityScan (RealityScan.exe RealityCapture.exe);;Executables (*.exe);;All files (*)",
        )
        if path:
            self.realityscan_edit.setText(str(Path(path)))

    def _detect_realityscan(self) -> None:
        found = find_realityscan_exe()
        if found is None:
            self.statusBar().showMessage("RealityScan executable was not found.", 5000)
            return
        self.realityscan_edit.setText(str(found))
        self.statusBar().showMessage(f"RealityScan detected: {found}", 5000)

    def _session_dir_from_text(self) -> Path | None:
        text = self._clean_path_text(self.session_edit.text())
        if not text:
            return None
        path = Path(text).expanduser()
        return path.parent if path.is_file() else path

    def _autoload_session_calibration(self) -> None:
        if self.calibration_edit.text().strip():
            return
        candidate = self._default_calibration_candidate()
        if candidate is not None and candidate.exists():
            self.calibration_edit.setText(str(candidate))

    def _find_recent_calibration(self) -> None:
        candidates: list[Path] = []
        default_candidate = self._default_calibration_candidate()
        if default_candidate is not None:
            candidates.append(default_candidate)
        calibration_dir = workspace_paths(create=True).calibrations
        if calibration_dir.exists():
            candidates.extend(calibration_dir.glob("*_stereo_calibration.json"))
            candidates.extend(calibration_dir.glob("stereo_calibration.json"))
        pilot_incoming = workspace_paths(create=True).pilot_incoming
        if pilot_incoming.exists():
            candidates.extend(pilot_incoming.glob("*/stereo_sessions/*/stereo_calibration.json"))
        for root in (self._pipeline_root() / "recordings" / "stereo_sessions", REPO_ROOT.parent / "TritonPilot" / "recordings" / "stereo_sessions"):
            if root.exists():
                candidates.extend(root.glob("*/stereo_calibration.json"))
        candidates = [path for path in candidates if path.exists()]
        if not candidates:
            self.statusBar().showMessage("No stereo_calibration.json files found.", 5000)
            return
        selected = sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)[0]
        self.calibration_edit.setText(str(selected))
        self.statusBar().showMessage(f"Calibration selected: {selected}", 5000)

    def _default_calibration_candidate(self) -> Path | None:
        session_dir = self._session_dir_from_text()
        if session_dir is None:
            return None
        calibration_dir = workspace_paths(create=True).calibrations
        candidates = [
            session_dir / "stereo_calibration.json",
            calibration_dir / f"{session_dir.name}_stereo_calibration.json",
            calibration_dir / "stereo_calibration.json",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[1]

    def _on_alignment_changed(self, *_args) -> None:
        self._refresh_command_preview()

    def _ensure_alignment_for_alignment_only(self, checked: bool) -> None:
        self._refresh_command_preview()

    def _copy_command(self) -> None:
        QApplication.clipboard().setText(self.command_preview_text())
        self.statusBar().showMessage("Command copied.", 2500)

    def _validate_before_run(self, *, output_workspace: str | None = None) -> bool:
        pipeline_root = self._pipeline_root()
        if not (pipeline_root / PIPELINE_PACKAGE_PATH).exists():
            QMessageBox.warning(
                self,
                "RealityScan Reconstruction",
                f"Pipeline file was not found:\n{pipeline_root / PIPELINE_PACKAGE_PATH}",
            )
            return False

        session = Path(self._clean_path_text(self.session_edit.text())).expanduser()
        if not session.exists():
            QMessageBox.warning(self, "RealityScan Reconstruction", f"Stereo session does not exist:\n{session}")
            return False
        if stereo_manifest_path(session) is None:
            QMessageBox.warning(self, "RealityScan Reconstruction", f"Not a stereo session or manifest:\n{session}")
            return False

        needs_calibration = self.metric_scale_check.isChecked() or self.metric_required_check.isChecked() or self.xmp_priors_check.isChecked()
        calibration_text = self._clean_path_text(self.calibration_edit.text())
        if needs_calibration and not calibration_text:
            QMessageBox.warning(self, "RealityScan Reconstruction", "Select a stereo calibration first.")
            return False
        if calibration_text and not Path(calibration_text).expanduser().exists():
            QMessageBox.warning(self, "RealityScan Reconstruction", f"Stereo calibration does not exist:\n{calibration_text}")
            return False

        output = Path(output_workspace or self._output_workspace_for_command(preview=False)).expanduser()
        if output.exists() and any(output.iterdir()) and not self.overwrite_check.isChecked():
            QMessageBox.warning(
                self,
                "RealityScan Reconstruction",
                "Output workspace already contains files. Enable overwrite or choose another workspace.",
            )
            return False
        return True

    def _start_pipeline(self) -> None:
        if self._process is not None and self._process.state() != QProcess.ProcessState.NotRunning:
            return
        output_workspace = self._output_workspace_for_command(preview=False)
        if not self._validate_before_run(output_workspace=output_workspace):
            return

        command = self.build_command(output_override=output_workspace)
        self._clear_outputs()
        self._line_buffer = ""
        self.log_text.clear()
        self._append_log("$ " + subprocess.list2cmdline(command) + "\n")

        process = QProcess(self)
        process.setProgram(command[0])
        process.setArguments(command[1:])
        process.setWorkingDirectory(str(self._pipeline_root()))
        process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        process.setProcessEnvironment(env)
        process.readyReadStandardOutput.connect(self._read_process_output)
        process.finished.connect(self._on_process_finished)
        process.errorOccurred.connect(self._on_process_error)
        self._process = process
        self._set_running(True)
        self._set_status("Running", "running")
        self._set_stage("Starting pipeline", 2)
        process.start()

    def _cancel_pipeline(self) -> None:
        if self._process is None or self._process.state() == QProcess.ProcessState.NotRunning:
            return
        self._append_log("\nCancel requested.\n")
        self._set_stage("Stopping pipeline", max(self.progress_bar.value(), 1))
        self._process.terminate()
        QTimer.singleShot(5000, self._kill_if_still_running)

    def _kill_if_still_running(self) -> None:
        if self._process is not None and self._process.state() != QProcess.ProcessState.NotRunning:
            self._process.kill()

    def _read_process_output(self) -> None:
        if self._process is None:
            return
        text = bytes(self._process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if not text:
            return
        self._append_log(text)
        self._line_buffer += text
        while "\n" in self._line_buffer:
            line, self._line_buffer = self._line_buffer.split("\n", 1)
            self._handle_log_line(line.rstrip("\r"))

    def _append_log(self, text: str) -> None:
        self.log_text.moveCursor(QTextCursor.MoveOperation.End)
        self.log_text.insertPlainText(text)
        self.log_text.moveCursor(QTextCursor.MoveOperation.End)

    def _handle_log_line(self, line: str) -> None:
        text = str(line or "").strip()
        if not text:
            return
        if text.startswith("Output workspace:"):
            self._set_workspace(Path(text.split(":", 1)[1].strip()))
        elif text.startswith("Scoring stereo pairs"):
            self._set_stage("Scoring stereo pairs", 8)
        elif text.startswith("Stereo session:"):
            self._set_stage(text, 12)
        elif text.startswith("Selected "):
            self._set_stage(text, 16)
        elif text.startswith("Cropping "):
            self._set_stage(text, 18)
        elif text.startswith("Writing "):
            self._set_stage("Writing selected frames", 24)
        elif text.startswith("Command file:"):
            self._set_stage("RealityScan command written", 30)
        elif text.startswith("Contact sheet:"):
            self._set_output_path("contact_sheet", Path(text.split(":", 1)[1].strip()))
        elif text.startswith("Component summary:"):
            self._set_output_path("components", Path(text.split(":", 1)[1].strip()))
        elif text.startswith("Component diagnostics summary:"):
            self._set_output_path("components", Path(text.split(":", 1)[1].strip()))
        elif text.startswith("Launching RealityScan"):
            self._set_stage("RealityScan reconstruction running", 36)
        elif text.startswith("Detected "):
            self._set_stage("Aligning images", max(self.progress_bar.value(), 44))
        elif "Feature detection completed" in text:
            self._set_stage("Feature detection complete", 50)
        elif text.startswith("Finalizing ") or "Reconstruction completed" in text:
            self._set_stage("Alignment complete", 56)
        elif "calculateHighModel" in text or "calculateNormalModel" in text or "calculatePreviewModel" in text:
            self._set_stage("Building mesh", 62)
        elif "Unwrapping Model completed" in text or "Executing command 'unwrap'" in text:
            self._set_stage("Unwrapping model", 72)
        elif "calculateTexture" in text or "Texturing Model completed" in text:
            self._set_stage("Texturing model", 82)
        elif "exportModel" in text or "Exporting Textured and Colored Mesh" in text:
            self._set_stage("Exporting model", 90)
        elif text.startswith("RealityScan progress:"):
            self._handle_realityscan_progress(text)
        elif text.startswith("Model exported:"):
            self._set_output_path("model", Path(text.split(":", 1)[1].strip()))
            self._set_stage("OBJ exported", 92)
        elif text.startswith("Metric model exported:"):
            match = re.match(r"Metric model exported:\s*(.*?)\s*(?:\(|$)", text)
            model_text = match.group(1).strip() if match else text.split(":", 1)[1].strip()
            self._set_output_path("metric_model", Path(model_text))
            self._set_stage("Metric OBJ exported", 96)
        elif "Metric stereo scaling failed:" in text:
            self._set_status("Metric Scale Failed", "warn")
            self._set_stage(text, max(self.progress_bar.value(), 92))
        elif text.startswith("Elapsed:"):
            self._set_stage(text, 100)

    def _handle_realityscan_progress(self, text: str) -> None:
        if "#completed" in text:
            self._set_stage("RealityScan completed", max(self.progress_bar.value(), 94))
            return
        for token in text.split()[2:]:
            try:
                value = float(token)
            except ValueError:
                continue
            if 0.0 <= value <= 1.0:
                self._set_stage("RealityScan running", max(self.progress_bar.value(), 36 + int(value * 54)))
                return

    def _on_process_finished(self, exit_code: int, _status: QProcess.ExitStatus) -> None:
        if self._line_buffer.strip():
            self._handle_log_line(self._line_buffer.strip())
            self._line_buffer = ""
        self._populate_known_workspace_outputs()
        self._set_running(False)
        if int(exit_code) == 0:
            self._set_status("Complete", "ok")
            self._set_stage("Pipeline complete", 100)
            self.statusBar().showMessage("RealityScan reconstruction complete.", 6000)
        else:
            self._set_status(f"Failed ({int(exit_code)})", "alert")
            self._set_stage("Pipeline failed", max(self.progress_bar.value(), 1))
            self.statusBar().showMessage(f"RealityScan reconstruction failed: exit {int(exit_code)}", 8000)

    def _on_process_error(self, error: QProcess.ProcessError) -> None:
        self._set_running(False)
        self._set_status("Process Error", "alert")
        self._set_stage(str(error), max(self.progress_bar.value(), 1))

    def _set_workspace(self, path: Path) -> None:
        self._workspace_path = Path(path)
        self._set_output_path("workspace", self._workspace_path)
        self._populate_known_workspace_outputs()
        self._set_stage("Workspace ready", 5)

    def _populate_known_workspace_outputs(self) -> None:
        if self._workspace_path is None:
            return
        workspace = self._workspace_path
        known = {
            "contact_sheet": workspace / "selection_contact_sheet.jpg",
            "report": workspace / "reports" / "final_overview.html",
            "components": workspace / "reports" / "alignment_components.csv",
            "model": workspace / "underwater_model.obj",
            "metric_model": workspace / "underwater_model_metric.obj",
            "metric_scale": workspace / "reports" / "metric_scale.json",
            "manifest": workspace / "manifest.json",
        }
        for key, path in known.items():
            if path.exists():
                self._set_output_path(key, path)

    def _set_output_path(self, key: str, path: Path) -> None:
        path = Path(path)
        self._output_paths[key] = path
        label = self._output_labels.get(key)
        if label is not None:
            label.setText(str(path))
            label.setToolTip(str(path))
        button = self._output_buttons.get(key)
        if button is not None:
            button.setEnabled(path.exists())
        if key in {"model", "metric_model"} and hasattr(self, "view_metric_model_btn"):
            metric = self._output_paths.get("metric_model")
            model = self._output_paths.get("model")
            self.view_metric_model_btn.setEnabled(bool((metric and metric.exists()) or (model and model.exists())))

    def _clear_outputs(self) -> None:
        self._workspace_path = None
        self._output_paths.clear()
        for key, label in self._output_labels.items():
            label.setText("-")
            label.setToolTip("")
            if key in self._output_buttons:
                self._output_buttons[key].setEnabled(False)
        if hasattr(self, "view_metric_model_btn"):
            self.view_metric_model_btn.setEnabled(False)

    def _open_output(self, key: str) -> None:
        path = self._output_paths.get(key)
        if path is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def _launch_model_viewer(self) -> None:
        path = self._output_paths.get("metric_model") or self._output_paths.get("model")
        if path is None or not path.exists():
            QMessageBox.information(self, "Model Viewer", "No exported OBJ model is available yet.")
            return
        self.model_viewer_panel.model_edit.setText(str(path))
        self.tabs.setCurrentWidget(self.model_viewer_panel)
        QTimer.singleShot(0, self.model_viewer_panel._start_viewer)

    def _set_running(self, running: bool) -> None:
        self.run_btn.setEnabled(not running)
        self.cancel_btn.setEnabled(running)
        self.clear_log_btn.setEnabled(not running)
        for widget in (
            self.pipeline_root_edit,
            self.pipeline_browse_btn,
            self.session_edit,
            self.session_browse_btn,
            self.manifest_browse_btn,
            self.calibration_edit,
            self.calibration_browse_btn,
            self.calibration_find_btn,
            self.output_edit,
            self.output_browse_btn,
            self.realityscan_edit,
            self.realityscan_browse_btn,
            self.realityscan_detect_btn,
            self.preset_combo,
            self.alignment_combo,
            self.advanced_group,
        ):
            widget.setEnabled(not running)

    def _set_status(self, text: str, tone: str) -> None:
        self.status_label.setText(text)
        self.status_label.setProperty("tone", tone)
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)
        self.status_label.update()

    def _set_stage(self, text: str, progress: int | None = None) -> None:
        self.stage_label.setText(str(text))
        self.stage_label.setToolTip(str(text))
        if progress is not None:
            self.progress_bar.setValue(max(0, min(100, int(progress))))
