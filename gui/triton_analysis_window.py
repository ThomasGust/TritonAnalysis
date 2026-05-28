"""Unified TritonAnalysis competition window."""

from __future__ import annotations

import os
import time
from pathlib import Path

from PyQt6.QtCore import QSettings, Qt, QThread, QTimer, QUrl
from PyQt6.QtGui import QAction, QDesktopServices
from PyQt6.QtWidgets import QFileDialog, QInputDialog, QLabel, QMainWindow, QTabWidget

from analysis_workspace import AnalysisWorkspace, set_active_workspace_root, workspace_paths
from crab_detector_cv import DEFAULT_UNWRAP_SIZE
from gui.crab_detection_window import CrabDetectionWindow
from gui.edna_analysis_window import EDNAAnalysisWindow
from gui.iceberg_measurement_window import IcebergMeasurementWindow
from gui.iceberg_tracking_window import IcebergTrackingWindow
from gui.multi_rect_length_measurement_window import MultiRectLengthMeasurementWindow
from gui.pilot_transfer_sync import PilotTransferSyncWorker
from gui.realityscan_reconstruction_window import RealityScanReconstructionWindow
from gui.responsive import resize_to_available_screen
from gui.stereo_calibration_window import StereoCalibrationWindow
from gui.stereo_iceberg_measurement_window import StereoIcebergMeasurementWindow
from pilot_transfer import DEFAULT_PILOT_TRANSFER_URL


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
        self._pilot_sync_last_ok_ts = 0.0
        self._pilot_sync_last_error = ""
        self._pilot_sync_auto_act: QAction | None = None

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(False)
        self.tabs.setMovable(True)
        self.tabs.setUsesScrollButtons(True)
        self.setCentralWidget(self.tabs)

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
                force_square=True,
                unwrap_size=DEFAULT_UNWRAP_SIZE,
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
        self._pilot_sync_label.setMinimumWidth(560)
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
        self._pilot_sync_label.setProperty("tone", tone or "")
        self._pilot_sync_label.style().unpolish(self._pilot_sync_label)
        self._pilot_sync_label.style().polish(self._pilot_sync_label)
        self._pilot_sync_label.update()

    def _update_pilot_sync_label(self, state: str, detail: str = "") -> None:
        destination = self._workspace.label_for(self._pilot_sync_output)
        if state == "syncing":
            text = f"Pilot Sync: SYNCING {self._pilot_sync_url} -> {destination}"
            tone = "warn"
        elif state == "ok":
            text = f"Pilot Sync: OK {detail} -> {destination}"
            tone = "ok"
        elif state == "lost":
            text = f"Pilot Sync: LOST {detail} | {self._pilot_sync_url} -> {destination}"
            tone = "alert"
        elif state == "off":
            text = f"Pilot Sync: OFF {self._pilot_sync_url} -> {destination}"
            tone = "warn"
        else:
            text = f"Pilot Sync: ready {self._pilot_sync_url} -> {destination}"
            tone = ""
        self._pilot_sync_label.setText(text)
        self._pilot_sync_label.setToolTip(
            f"{text}\nWorkspace root: {self._workspace.root}\nSync folder: {self._pilot_sync_output}"
        )
        self._set_pilot_sync_label_tone(tone)

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
        if (not self._pilot_sync_enabled and not force) or self._pilot_sync_busy:
            return
        self._pilot_sync_busy = True
        self._update_pilot_sync_label("syncing")

        worker = PilotTransferSyncWorker(
            base_url=self._pilot_sync_url,
            destination=self._pilot_sync_output,
            timeout=self._pilot_sync_timeout_s,
        )
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
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

    def _finish_pilot_sync(self, payload: object) -> None:
        self._pilot_sync_busy = False
        data = payload if isinstance(payload, dict) else {}
        if data.get("ok"):
            summary = data.get("summary")
            copied = int(getattr(summary, "copied", 0) or 0)
            skipped = int(getattr(summary, "skipped", 0) or 0)
            scanned = int(getattr(summary, "scanned", 0) or 0)
            bytes_copied = int(getattr(summary, "bytes_copied", 0) or 0)
            copied_mb = bytes_copied / (1024 * 1024)
            self._pilot_sync_last_ok_ts = time.time()
            self._pilot_sync_last_error = ""
            detail = f"scanned {scanned}, copied {copied}, skipped {skipped}, {copied_mb:.1f} MB"
            self._update_pilot_sync_label("ok", detail)
            if copied:
                self.statusBar().showMessage(
                    f"Pilot sync copied {copied} file(s) to {self._workspace.label_for(self._pilot_sync_output)}",
                    6000,
                )
            return

        error = str(data.get("error") or "transfer failed")
        self._pilot_sync_last_error = error
        self._update_pilot_sync_label("lost", error)

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
