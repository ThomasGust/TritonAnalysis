"""Unified TritonAnalysis competition window."""

from __future__ import annotations

import os
import time
from pathlib import Path

from PyQt6.QtCore import QSettings, Qt, QThread, QTimer, QUrl
from PyQt6.QtGui import QAction, QDesktopServices
from PyQt6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from triton_analysis.gui.file_dialogs import ThumbnailFileDialog as QFileDialog

from triton_analysis.workspace import AnalysisWorkspace, set_active_workspace_root, workspace_paths
from triton_analysis.gui.crab_detection_window import CrabDetectionWindow
from triton_analysis.gui.edna_analysis_window import EDNAAnalysisWindow
from triton_analysis.gui.iceberg_measurement_window import IcebergMeasurementWindow
from triton_analysis.gui.iceberg_tracking_window import IcebergTrackingWindow
from triton_analysis.gui.multi_rect_length_measurement_window import MultiRectLengthMeasurementWindow
from triton_analysis.gui.pilot_transfer_sync import PilotTransferSyncWorker, SyncFn
from triton_analysis.gui.realityscan_reconstruction_window import RealityScanReconstructionWindow
from triton_analysis.gui.responsive import resize_to_available_screen
from triton_analysis.gui.stereo_calibration_window import StereoCalibrationWindow
from triton_analysis.gui.stereo_iceberg_measurement_window import StereoIcebergMeasurementWindow
from triton_analysis.sync.pilot_transfer import DEFAULT_PILOT_TRANSFER_URL


_TRANSIENT_WORKSPACE_PARTS = {".pytest-work", ".pytest-tmp"}
_TRANSIENT_WORKSPACE_PREFIXES = ("pytest-", "pytest-of-")


def _looks_like_transient_workspace(path: str | Path) -> bool:
    """Return whether a saved workspace points at a test temp directory."""
    if not path:
        return False
    try:
        parts = Path(path).expanduser().parts
    except (OSError, RuntimeError, ValueError):
        return False
    for part in parts:
        lowered = part.lower()
        if lowered in _TRANSIENT_WORKSPACE_PARTS:
            return True
        if any(lowered.startswith(prefix) for prefix in _TRANSIENT_WORKSPACE_PREFIXES):
            return True
    return False


class TritonAnalysisWindow(QMainWindow):
    """Tabbed field workflow for the MATE ROV analysis station."""

    TAB_KEYS = (
        "coral-reconstruction",
        "crab-detection",
        "stereo-iceberg-length",
        "iceberg-tracking",
        "edna-analysis",
        "stereo-calibration",
        "backup-coral-measurement",
        "backup-iceberg-measurement",
    )

    def __init__(
        self,
        *,
        crab_paths: list[str | Path] | None = None,
        crab_reference_image: str | Path | None = None,
        backup_coral_paths: list[str | Path] | None = None,
        backup_iceberg_paths: list[str | Path] | None = None,
        stereo_manifest_path: str | Path | None = None,
        stereo_calibration_path: str | Path | None = None,
        reconstruction_session_path: str | Path | None = None,
        use_sample_edna: bool = False,
        initial_tab: str | None = None,
        pilot_transfer_url: str | None = None,
        pilot_transfer_output: str | Path | None = None,
        pilot_transfer_auto_sync: bool | None = None,
        pilot_transfer_sync_fn: SyncFn | None = None,
        workspace_root: str | Path | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("TritonAnalysis")
        self._windows: dict[str, QMainWindow] = {}
        self._settings = QSettings("TritonAnalysis", "UnifiedApp")
        stored_workspace = str(self._settings.value("workspace/root", "") or "").strip()
        if _looks_like_transient_workspace(stored_workspace):
            stored_workspace = ""
        env_workspace = os.environ.get("TRITON_ANALYSIS_WORKSPACE", "").strip()
        self._workspace: AnalysisWorkspace = workspace_paths(
            workspace_root or env_workspace or stored_workspace or None,
            create=True,
        )
        set_active_workspace_root(self._workspace.root)
        self._pilot_sync_url = str(
            pilot_transfer_url
            or self._settings.value("pilot_transfer/base_url", DEFAULT_PILOT_TRANSFER_URL)
            or DEFAULT_PILOT_TRANSFER_URL
        ).strip()
        env_sync_output = os.environ.get("TRITON_ANALYSIS_INBOX", "").strip()
        default_sync_output = env_sync_output or str(self._workspace.pilot_incoming)
        stored_sync_output = str(self._settings.value("pilot_transfer/output", "") or "").strip()
        if _looks_like_transient_workspace(stored_sync_output):
            stored_sync_output = ""
        if pilot_transfer_output:
            sync_output = pilot_transfer_output
        elif env_sync_output:
            sync_output = env_sync_output
        elif workspace_root or env_workspace:
            sync_output = default_sync_output
        else:
            sync_output = stored_sync_output or default_sync_output
        self._pilot_sync_output = Path(sync_output).expanduser()
        if pilot_transfer_auto_sync is None:
            stored_auto_sync = self._settings.value("pilot_transfer/auto_sync", "1")
            pilot_transfer_auto_sync = self._setting_truthy(stored_auto_sync, default=True)
        env_auto_sync = os.environ.get("TRITON_ANALYSIS_AUTO_SYNC", "").strip().lower()
        if env_auto_sync in {"0", "false", "no", "off"}:
            pilot_transfer_auto_sync = False
        elif env_auto_sync in {"1", "true", "yes", "on"}:
            pilot_transfer_auto_sync = True
        self._pilot_sync_enabled = bool(pilot_transfer_auto_sync)
        self._pilot_sync_timeout_s = float(os.environ.get("TRITON_ANALYSIS_SYNC_TIMEOUT_S", "2.0") or "2.0")
        self._pilot_sync_interval_ms = max(
            1000,
            int(float(os.environ.get("TRITON_ANALYSIS_SYNC_INTERVAL_S", "10.0") or "10.0") * 1000),
        )
        self._pilot_sync_busy = False
        self._pilot_sync_thread: QThread | None = None
        self._pilot_sync_worker: PilotTransferSyncWorker | None = None
        self._pilot_sync_fn = pilot_transfer_sync_fn
        self._pilot_sync_last_ok_ts = 0.0
        self._pilot_sync_last_error = ""
        self._pilot_sync_auto_act: QAction | None = None

        central = QWidget()
        central_layout = QVBoxLayout(central)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(6)

        self._pilot_sync_panel = self._build_pilot_sync_panel()
        central_layout.addWidget(self._pilot_sync_panel, 0)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(False)
        self.tabs.setMovable(True)
        self.tabs.setUsesScrollButtons(True)
        central_layout.addWidget(self.tabs, 1)
        self.setCentralWidget(central)

        calibration_text = str(stereo_calibration_path) if stereo_calibration_path else None
        manifest_text = str(stereo_manifest_path) if stereo_manifest_path else None
        reconstruction_text = str(reconstruction_session_path) if reconstruction_session_path else manifest_text

        self._add_window(
            "coral-reconstruction",
            "Coral Reconstruction",
            RealityScanReconstructionWindow(
                session_path=reconstruction_text,
                calibration_path=calibration_text,
                parent=self,
            ),
        )
        self._add_window(
            "crab-detection",
            "Crab Detection",
            CrabDetectionWindow(
                image_paths=[str(path) for path in crab_paths or []],
                reference_image=crab_reference_image,
                parent=self,
            ),
        )
        self._add_window(
            "stereo-iceberg-length",
            "Stereo Iceberg Length",
            StereoIcebergMeasurementWindow(
                manifest_path=manifest_text,
                calibration_path=calibration_text,
                parent=self,
            ),
        )
        self._add_window(
            "iceberg-tracking",
            "Iceberg Tracking",
            IcebergTrackingWindow(parent=self),
        )
        self._add_window(
            "edna-analysis",
            "eDNA Analysis",
            EDNAAnalysisWindow(
                use_sample=use_sample_edna,
                parent=self,
            ),
        )
        self._add_window(
            "stereo-calibration",
            "Stereo Calibration",
            StereoCalibrationWindow(
                manifest_path=manifest_text,
                parent=self,
            ),
        )
        self._add_window(
            "backup-coral-measurement",
            "Backup Coral Measurement",
            MultiRectLengthMeasurementWindow(
                media_paths=[str(path) for path in backup_coral_paths or []],
                parent=self,
            ),
        )
        self._add_window(
            "backup-iceberg-measurement",
            "Backup Iceberg Measurement",
            IcebergMeasurementWindow(
                media_paths=[str(path) for path in backup_iceberg_paths or []],
                parent=self,
            ),
        )

        self._tab_aliases = {
            "crab": "crab-detection",
            "iceberg-measurement": "stereo-iceberg-length",
            "iceberg-length": "stereo-iceberg-length",
            "edna": "edna-analysis",
            "backup-coral": "backup-coral-measurement",
            "multi-rect": "backup-coral-measurement",
            "multi-rect-length": "backup-coral-measurement",
            "backup-iceberg": "backup-iceberg-measurement",
        }

        self._pilot_sync_label = QLabel()
        self._pilot_sync_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._pilot_sync_label.setMinimumWidth(0)
        self.statusBar().addPermanentWidget(self._pilot_sync_label, 1)

        self._pilot_sync_timer = QTimer(self)
        self._pilot_sync_timer.setInterval(self._pilot_sync_interval_ms)
        self._pilot_sync_timer.timeout.connect(self._start_pilot_sync)
        self._make_menu()
        self._update_pilot_sync_label("ready" if self._pilot_sync_enabled else "off")
        if self._pilot_sync_enabled:
            self._pilot_sync_timer.start()
            QTimer.singleShot(1000, self._start_pilot_sync)

        self.focus_tab(initial_tab or "coral-reconstruction")
        self.statusBar().showMessage("TritonAnalysis unified app ready.")
        resize_to_available_screen(self, 1720, 980, min_width=1120, min_height=720)

    @staticmethod
    def _setting_truthy(value, *, default: bool = True) -> bool:
        if value is None:
            return bool(default)
        text = str(value).strip().lower()
        if text in {"0", "false", "no", "off"}:
            return False
        if text in {"1", "true", "yes", "on"}:
            return True
        return bool(default)

    @staticmethod
    def _format_bytes(value: int | float) -> str:
        amount = float(value or 0)
        units = ("B", "KB", "MB", "GB", "TB")
        for unit in units:
            if abs(amount) < 1024.0 or unit == units[-1]:
                if unit == "B":
                    return f"{int(amount)} {unit}"
                return f"{amount:.1f} {unit}"
            amount /= 1024.0
        return f"{amount:.1f} TB"

    def _build_pilot_sync_panel(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName("pilotSyncPanel")
        panel.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        panel.setStyleSheet(
            """
            QFrame#pilotSyncPanel {
                background: #202028;
                border-bottom: 1px solid #343442;
            }
            QLabel#pilotSyncState {
                font-weight: 700;
                padding: 2px 4px;
            }
            QLabel#pilotSyncState[tone="ok"] { color: #9be7b0; }
            QLabel#pilotSyncState[tone="warn"] { color: #f4cf7a; }
            QLabel#pilotSyncState[tone="alert"] { color: #ffaaa5; }
            QLabel#pilotSyncProgress {
                color: #ffffff;
                padding: 2px 4px;
            }
            QLabel#pilotSyncMeta {
                color: #c7cad7;
                padding: 1px 4px;
            }
            """
        )

        layout = QGridLayout(panel)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setHorizontalSpacing(10)
        layout.setVerticalSpacing(2)

        self._pilot_sync_state_panel_label = QLabel("Pilot Sync: ready")
        self._pilot_sync_state_panel_label.setObjectName("pilotSyncState")
        self._pilot_sync_progress_label = QLabel("No transfer running.")
        self._pilot_sync_progress_label.setObjectName("pilotSyncProgress")
        self._pilot_sync_source_label = QLabel()
        self._pilot_sync_source_label.setObjectName("pilotSyncMeta")
        self._pilot_sync_destination_label = QLabel()
        self._pilot_sync_destination_label.setObjectName("pilotSyncMeta")
        self._pilot_sync_last_label = QLabel("Last sync: never")
        self._pilot_sync_last_label.setObjectName("pilotSyncMeta")

        for label in (
            self._pilot_sync_state_panel_label,
            self._pilot_sync_progress_label,
            self._pilot_sync_source_label,
            self._pilot_sync_destination_label,
            self._pilot_sync_last_label,
        ):
            label.setWordWrap(True)
            label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            label.setMinimumWidth(0)

        button_row = QHBoxLayout()
        button_row.setContentsMargins(0, 0, 0, 0)
        button_row.setSpacing(6)
        sync_now_btn = QPushButton("Sync Now")
        sync_now_btn.setToolTip("Check TritonPilot now and receive any missing or changed files.")
        sync_now_btn.clicked.connect(lambda _checked=False: self._start_pilot_sync(force=True))
        open_folder_btn = QPushButton("Open Folder")
        open_folder_btn.setToolTip("Open the folder where TritonPilot files are received.")
        open_folder_btn.clicked.connect(self._open_pilot_sync_folder)
        button_row.addWidget(sync_now_btn)
        button_row.addWidget(open_folder_btn)

        layout.addWidget(self._pilot_sync_state_panel_label, 0, 0)
        layout.addWidget(self._pilot_sync_progress_label, 0, 1)
        layout.addLayout(button_row, 0, 2)
        layout.addWidget(self._pilot_sync_source_label, 1, 0)
        layout.addWidget(self._pilot_sync_destination_label, 1, 1)
        layout.addWidget(self._pilot_sync_last_label, 1, 2)
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 2)
        layout.setColumnStretch(2, 0)
        return panel

    def _make_menu(self) -> None:
        workspace_menu = self.menuBar().addMenu("&Workspace")

        choose_workspace_act = QAction("Set Workspace Root...", self)
        choose_workspace_act.triggered.connect(self._choose_workspace_root)
        workspace_menu.addAction(choose_workspace_act)

        open_workspace_act = QAction("Open Workspace Folder", self)
        open_workspace_act.triggered.connect(self._open_workspace_folder)
        workspace_menu.addAction(open_workspace_act)

        transfer_menu = self.menuBar().addMenu("&Pilot Sync")

        self._pilot_sync_auto_act = QAction("Auto Sync", self)
        self._pilot_sync_auto_act.setCheckable(True)
        self._pilot_sync_auto_act.setChecked(bool(self._pilot_sync_enabled))
        self._pilot_sync_auto_act.toggled.connect(self._set_pilot_sync_enabled)
        transfer_menu.addAction(self._pilot_sync_auto_act)

        sync_now_act = QAction("Sync Now", self)
        sync_now_act.triggered.connect(lambda _checked=False: self._start_pilot_sync(force=True))
        transfer_menu.addAction(sync_now_act)

        folder_act = QAction("Set Sync Folder...", self)
        folder_act.triggered.connect(self._choose_pilot_sync_output)
        transfer_menu.addAction(folder_act)

        url_act = QAction("Set Pilot URL...", self)
        url_act.triggered.connect(self._choose_pilot_sync_url)
        transfer_menu.addAction(url_act)

    def _set_pilot_sync_label_tone(self, tone: str | None) -> None:
        for label in (self._pilot_sync_label, self._pilot_sync_state_panel_label):
            label.setProperty("tone", tone or "")
            label.style().unpolish(label)
            label.style().polish(label)
            label.update()

    def _update_pilot_sync_label(self, state: str, detail: str = "") -> None:
        destination = self._workspace.label_for(self._pilot_sync_output)
        if state == "syncing":
            text = f"Pilot Sync: SYNCING {self._pilot_sync_url} -> {destination}"
            panel_state = "Pilot Sync: SYNCING"
            progress = detail or "Checking TritonPilot for new recordings..."
            tone = "warn"
        elif state == "ok":
            text = f"Pilot Sync: OK {detail} -> {destination}"
            panel_state = "Pilot Sync: OK"
            progress = detail or "Sync complete."
            tone = "ok"
        elif state == "lost":
            text = f"Pilot Sync: LOST {detail} | {self._pilot_sync_url} -> {destination}"
            panel_state = "Pilot Sync: LOST"
            progress = f"Connection problem: {detail}" if detail else "Connection problem."
            tone = "alert"
        elif state == "off":
            text = f"Pilot Sync: OFF {self._pilot_sync_url} -> {destination}"
            panel_state = "Pilot Sync: OFF"
            progress = "Automatic sync is off. Use Sync Now to check for missing files."
            tone = "warn"
        else:
            text = f"Pilot Sync: ready {self._pilot_sync_url} -> {destination}"
            panel_state = "Pilot Sync: ready"
            progress = "Ready to receive recordings."
            tone = ""
        self._pilot_sync_label.setText(text)
        self._pilot_sync_label.setToolTip(
            f"{text}\nWorkspace root: {self._workspace.root}\nSync folder: {self._pilot_sync_output}"
        )
        self._pilot_sync_state_panel_label.setText(panel_state)
        self._pilot_sync_progress_label.setText(progress)
        self._pilot_sync_source_label.setText(f"Source: {self._pilot_sync_url}")
        self._pilot_sync_destination_label.setText(f"Receiving to: {destination}")
        panel_tooltip = (
            f"Source: {self._pilot_sync_url}\n"
            f"Workspace root: {self._workspace.root}\n"
            f"Sync folder: {self._pilot_sync_output}"
        )
        for label in (
            self._pilot_sync_state_panel_label,
            self._pilot_sync_progress_label,
            self._pilot_sync_source_label,
            self._pilot_sync_destination_label,
            self._pilot_sync_last_label,
        ):
            label.setToolTip(panel_tooltip)
        self._set_pilot_sync_label_tone(tone)

    @staticmethod
    def _short_transfer_path(path_text: object, *, max_chars: int = 82) -> str:
        text = str(path_text or "")
        if len(text) <= max_chars:
            return text
        return "..." + text[-max(0, max_chars - 3) :]

    def _handle_pilot_sync_progress(self, payload: object) -> None:
        data = payload if isinstance(payload, dict) else {}
        event = str(data.get("event") or "")
        if not event:
            return
        total_files = int(data.get("total_files") or data.get("scanned") or 0)
        index = int(data.get("index") or 0)
        path = self._short_transfer_path(data.get("path", ""))

        if event in {"sync_start", "index_start"}:
            self._update_pilot_sync_label("syncing", "Requesting the file list from TritonPilot...")
            return
        if event == "local_fallback":
            source = self._short_transfer_path(data.get("source", ""), max_chars=72)
            self._pilot_sync_progress_label.setText(
                f"Pilot URL unavailable. Checking local TritonPilot recordings at {source}..."
            )
            return
        if event == "index_done":
            scanned = int(data.get("scanned") or 0)
            total_bytes = int(data.get("total_bytes") or 0)
            source_label = "Local Pilot folder" if str(data.get("base_url") or "").startswith("local:") else "Pilot"
            self._pilot_sync_progress_label.setText(
                f"{source_label} has {scanned} file(s), {self._format_bytes(total_bytes)}. Checking local folder..."
            )
            return
        if event == "skipped":
            skipped = int(data.get("skipped") or 0)
            if index == total_files or index % 25 == 0:
                self._pilot_sync_progress_label.setText(
                    f"Checking local files: {index}/{total_files} already received ({skipped} current)."
                )
            return
        if event == "would_copy":
            self._pilot_sync_progress_label.setText(f"Would receive {path} ({self._format_bytes(data.get('size') or 0)}).")
            return
        if event == "copy_start":
            prefix = f"{index}/{total_files}: " if total_files else ""
            self._pilot_sync_progress_label.setText(
                f"Receiving {prefix}{path} ({self._format_bytes(data.get('size') or 0)})..."
            )
            return
        if event == "copy_progress":
            file_size = int(data.get("size") or 0)
            file_bytes = int(data.get("file_bytes_copied") or 0)
            percent = (file_bytes / file_size * 100.0) if file_size > 0 else 0.0
            prefix = f"{index}/{total_files}: " if total_files else ""
            self._pilot_sync_progress_label.setText(
                f"Receiving {prefix}{path} {percent:.0f}% "
                f"({self._format_bytes(file_bytes)} / {self._format_bytes(file_size)})"
            )
            return
        if event == "copy_done":
            copied = int(data.get("copied") or 0)
            self._pilot_sync_progress_label.setText(
                f"Received {copied} file(s). Last received: {path} ({self._format_bytes(data.get('size') or 0)})."
            )
            return
        if event == "complete":
            copied = int(data.get("copied") or 0)
            skipped = int(data.get("skipped") or 0)
            scanned = int(data.get("scanned") or 0)
            bytes_copied = int(data.get("bytes_copied") or 0)
            self._pilot_sync_progress_label.setText(
                f"Sync complete: received {copied}, already current {skipped}, scanned {scanned}, "
                f"{self._format_bytes(bytes_copied)} transferred."
            )
            return
        if event == "sync_error":
            self._pilot_sync_progress_label.setText(f"Connection problem: {data.get('error') or 'transfer failed'}")

    def _choose_workspace_root(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "Choose TritonAnalysis workspace root",
            str(self._workspace.root),
        )
        if not selected:
            return
        self._set_workspace_root(Path(selected))

    def _set_workspace_root(self, root: str | Path) -> None:
        self._workspace = workspace_paths(root, create=True)
        set_active_workspace_root(self._workspace.root)
        self._settings.setValue("workspace/root", str(self._workspace.root))
        self._pilot_sync_output = self._workspace.pilot_incoming
        self._settings.setValue("pilot_transfer/output", str(self._pilot_sync_output))
        self._update_pilot_sync_label("ready" if self._pilot_sync_enabled else "off")
        self.statusBar().showMessage(f"Workspace set: {self._workspace.root}", 5000)
        if self._pilot_sync_enabled:
            self._start_pilot_sync()

    def _open_workspace_folder(self) -> None:
        self._workspace.ensure()
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._workspace.root)))

    def _open_pilot_sync_folder(self) -> None:
        self._pilot_sync_output.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._pilot_sync_output)))

    def _set_pilot_sync_enabled(self, enabled: bool) -> None:
        self._pilot_sync_enabled = bool(enabled)
        self._settings.setValue("pilot_transfer/auto_sync", "1" if self._pilot_sync_enabled else "0")
        if self._pilot_sync_auto_act is not None and self._pilot_sync_auto_act.isChecked() != self._pilot_sync_enabled:
            self._pilot_sync_auto_act.setChecked(self._pilot_sync_enabled)
        if self._pilot_sync_enabled:
            if not self._pilot_sync_timer.isActive():
                self._pilot_sync_timer.start()
            self._update_pilot_sync_label("ready")
            self._start_pilot_sync()
        else:
            self._pilot_sync_timer.stop()
            self._update_pilot_sync_label("off")

    def _choose_pilot_sync_output(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "Choose TritonPilot sync folder",
            str(self._pilot_sync_output),
        )
        if not selected:
            return
        self._pilot_sync_output = Path(selected).expanduser()
        self._settings.setValue("pilot_transfer/output", str(self._pilot_sync_output))
        self._update_pilot_sync_label("ready" if self._pilot_sync_enabled else "off")
        if self._pilot_sync_enabled:
            self._start_pilot_sync()

    def _choose_pilot_sync_url(self) -> None:
        text, ok = QInputDialog.getText(
            self,
            "Pilot Sync URL",
            "TritonPilot transfer URL:",
            text=self._pilot_sync_url,
        )
        if not ok:
            return
        text = str(text).strip().rstrip("/")
        if not text:
            return
        self._pilot_sync_url = text
        self._settings.setValue("pilot_transfer/base_url", self._pilot_sync_url)
        self._update_pilot_sync_label("ready" if self._pilot_sync_enabled else "off")
        if self._pilot_sync_enabled:
            self._start_pilot_sync()

    def _start_pilot_sync(self, *, force: bool = False) -> None:
        if not self._pilot_sync_enabled and not force:
            return
        if self._pilot_sync_busy:
            if force:
                self._pilot_sync_progress_label.setText("A Pilot sync is already running.")
            return
        self._pilot_sync_busy = True
        self._update_pilot_sync_label("syncing")

        worker_kwargs = {
            "base_url": self._pilot_sync_url,
            "destination": self._pilot_sync_output,
            "timeout": self._pilot_sync_timeout_s,
        }
        if self._pilot_sync_fn is not None:
            worker_kwargs["sync_fn"] = self._pilot_sync_fn
        worker = PilotTransferSyncWorker(**worker_kwargs)
        thread = QThread(self)
        self._pilot_sync_worker = worker
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._handle_pilot_sync_progress)
        worker.finished.connect(self._finish_pilot_sync)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(lambda: self._clear_pilot_sync_thread(thread))
        self._pilot_sync_thread = thread
        thread.start()

    def _clear_pilot_sync_thread(self, thread: QThread) -> None:
        if self._pilot_sync_thread is thread:
            self._pilot_sync_thread = None
        if self._pilot_sync_busy and self._pilot_sync_worker is not None:
            self._pilot_sync_busy = False
            self._pilot_sync_worker = None
            self._update_pilot_sync_label("lost", "sync worker stopped before reporting completion")

    def _finish_pilot_sync(self, payload: object) -> None:
        self._pilot_sync_busy = False
        self._pilot_sync_worker = None
        data = payload if isinstance(payload, dict) else {}
        if data.get("ok"):
            summary = data.get("summary")
            copied = int(getattr(summary, "copied", 0) or 0)
            skipped = int(getattr(summary, "skipped", 0) or 0)
            scanned = int(getattr(summary, "scanned", 0) or 0)
            bytes_copied = int(getattr(summary, "bytes_copied", 0) or 0)
            self._pilot_sync_last_ok_ts = time.time()
            self._pilot_sync_last_error = ""
            source = "local Pilot" if str(getattr(summary, "base_url", "") or "").startswith("local:") else "Pilot"
            detail = (
                f"{source}: scanned {scanned}, received {copied}, already current {skipped}, "
                f"{self._format_bytes(bytes_copied)} transferred"
            )
            self._update_pilot_sync_label("ok", detail)
            self._pilot_sync_last_label.setText(time.strftime("Last sync: %H:%M:%S"))
            if copied:
                self.statusBar().showMessage(
                    f"Pilot sync copied {copied} file(s) to {self._workspace.label_for(self._pilot_sync_output)}",
                    6000,
                )
            return

        error = str(data.get("error") or "transfer failed")
        self._pilot_sync_last_error = error
        self._update_pilot_sync_label("lost", error)
        self._pilot_sync_last_label.setText(time.strftime("Last attempt: %H:%M:%S"))

    def focus_tab(self, key: str) -> bool:
        """Select a tab by stable key."""
        normalized = str(key or "").strip().lower()
        normalized = self._tab_aliases.get(normalized, normalized)
        window = self._windows.get(normalized)
        if window is None:
            return False
        index = self.tabs.indexOf(window)
        if index < 0:
            return False
        self.tabs.setCurrentIndex(index)
        return True

    def closeEvent(self, event) -> None:
        self._pilot_sync_timer.stop()
        thread = self._pilot_sync_thread
        if thread is not None and thread.isRunning():
            thread.quit()
            thread.wait(max(500, int((self._pilot_sync_timeout_s + 0.5) * 1000)))
        for window in reversed(list(self._windows.values())):
            shutdown = getattr(window, "shutdown", None)
            if callable(shutdown):
                shutdown()
        super().closeEvent(event)

    def _add_window(self, key: str, label: str, window: QMainWindow) -> None:
        normalized = str(key).strip().lower()
        window.setWindowFlags(Qt.WindowType.Widget)
        window.statusBar().setSizeGripEnabled(False)
        self._windows[normalized] = window
        self.tabs.addTab(window, label)
