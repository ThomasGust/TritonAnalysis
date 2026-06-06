import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import QSettings, Qt
from PyQt6.QtWidgets import QApplication, QScrollArea, QTabWidget

from triton_analysis.gui.style import apply_modern_style
from triton_analysis.workspace import set_active_workspace_root


def _app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
        apply_modern_style(app)
    return app


@pytest.fixture(autouse=True)
def _disable_auto_pilot_sync(monkeypatch):
    monkeypatch.setenv("TRITON_ANALYSIS_AUTO_SYNC", "0")


@pytest.fixture(autouse=True)
def _isolate_unified_app_settings():
    settings = QSettings("TritonAnalysis", "UnifiedApp")
    snapshot = {key: settings.value(key) for key in settings.allKeys()}
    settings.clear()
    settings.sync()
    del settings
    set_active_workspace_root(None)
    try:
        yield
    finally:
        set_active_workspace_root(None)
        settings = QSettings("TritonAnalysis", "UnifiedApp")
        settings.clear()
        for key, value in snapshot.items():
            settings.setValue(key, value)
        settings.sync()
        del settings


@pytest.mark.parametrize(
    ("window_path", "min_scroll_areas"),
    [
        ("triton_analysis.gui.triton_analysis_window.TritonAnalysisWindow", 6),
        ("triton_analysis.gui.crab_detection_window.CrabDetectionWindow", 1),
        ("triton_analysis.gui.edna_analysis_window.EDNAAnalysisWindow", 1),
        ("triton_analysis.gui.iceberg_tracking_window.IcebergTrackingWindow", 1),
        ("triton_analysis.gui.coral_garden_model_window.CoralGardenModelWindow", 0),
        ("triton_analysis.gui.iceberg_measurement_window.IcebergMeasurementWindow", 2),
        ("triton_analysis.gui.planar_height_measurement_window.PlanarHeightMeasurementWindow", 2),
        ("triton_analysis.gui.multi_rect_length_measurement_window.MultiRectLengthMeasurementWindow", 2),
        ("triton_analysis.gui.stereo_calibration_window.StereoCalibrationWindow", 1),
        ("triton_analysis.gui.stereo_depth_window.StereoDepthWindow", 1),
        ("triton_analysis.gui.stereo_segment_measurement_window.StereoSegmentMeasurementWindow", 1),
        ("triton_analysis.gui.stereo_iceberg_measurement_window.StereoIcebergMeasurementWindow", 1),
        ("triton_analysis.gui.realityscan_reconstruction_window.RealityScanReconstructionWindow", 1),
        ("triton_analysis.gui.realityscan_model_viewer_window.RealityScanModelViewerWindow", 1),
        ("triton_analysis.apps.color_corr.MainWindow", 3),
    ],
)
def test_analysis_windows_fit_available_screen(window_path: str, min_scroll_areas: int):
    app = _app()
    module_name, class_name = window_path.rsplit(".", 1)
    module = pytest.importorskip(module_name)
    window_cls = getattr(module, class_name)

    window = window_cls()
    try:
        window.show()
        app.processEvents()

        screen = window.screen() or app.primaryScreen()
        assert screen is not None
        available = screen.availableGeometry()
        assert window.width() <= available.width()
        assert window.height() <= available.height()
        assert len(window.findChildren(QScrollArea)) >= min_scroll_areas
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()


def test_multi_rect_actions_and_anchor_canvases_are_visible():
    app = _app()
    from triton_analysis.gui.multi_rect_length_measurement_window import MultiRectLengthMeasurementWindow

    window = MultiRectLengthMeasurementWindow()
    try:
        window.show()
        app.processEvents()

        for attr_name in (
            "add_measure_btn",
            "remove_measure_btn",
            "undo_btn",
            "delete_btn",
            "clear_btn",
        ):
            button = getattr(window, attr_name)
            top_left = button.mapTo(window, button.rect().topLeft())
            bottom_right = button.mapTo(window, button.rect().bottomRight())
            assert top_left.x() >= 0
            assert top_left.y() >= 0
            assert bottom_right.x() <= window.width()
            assert bottom_right.y() <= window.height()

        sizes = window.setup_splitter.sizes()
        assert len(sizes) == 2
        assert min(sizes) > 0
        assert min(sizes) / max(sizes) > 0.75
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()


def test_unified_analysis_window_contains_competition_tabs():
    app = _app()
    from triton_analysis.gui.triton_analysis_window import TritonAnalysisWindow

    window = TritonAnalysisWindow()
    try:
        window.show()
        app.processEvents()

        tabs = window.findChild(QTabWidget)
        assert tabs is not None
        assert tabs.count() == 9
        assert [tabs.tabText(index) for index in range(tabs.count())] == [
            "Coral Reconstruction",
            "Crab Detection",
            "Stereo Iceberg Length",
            "Iceberg Tracking",
            "eDNA Analysis",
            "Stereo Calibration",
            "Backup Coral Measurement",
            "Backup Iceberg Measurement",
            "SSH",
        ]
        assert window.focus_tab("backup-coral-measurement") is True
        assert tabs.currentIndex() == 6
        assert window.focus_tab("crab") is True
        assert tabs.currentIndex() == 1
        assert window.focus_tab("terminal") is True
        assert tabs.currentIndex() == 8
        assert window.focus_tab("missing") is False
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()


def test_unified_analysis_window_shows_pilot_sync_destination(tmp_path: Path):
    app = _app()
    from triton_analysis.gui.triton_analysis_window import TritonAnalysisWindow

    destination = tmp_path / "incoming"
    window = TritonAnalysisWindow(
        pilot_transfer_auto_sync=False,
        pilot_transfer_url="http://10.77.0.1:8765",
        pilot_transfer_output=destination,
    )
    try:
        window.show()
        app.processEvents()

        assert "Pilot Sync: OFF" in window._pilot_sync_label.text()
        assert "http://10.77.0.1:8765" in window._pilot_sync_label.text()
        assert str(destination) in window._pilot_sync_label.text()
        assert "Automatic sync is off" in window._pilot_sync_progress_label.text()
        assert str(destination) in window._pilot_sync_destination_label.text()
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()


def test_unified_analysis_window_uses_workspace_relative_sync_label(tmp_path: Path):
    app = _app()
    from triton_analysis.gui.triton_analysis_window import TritonAnalysisWindow

    workspace = tmp_path / "analysis-workspace"
    window = TritonAnalysisWindow(
        pilot_transfer_auto_sync=False,
        pilot_transfer_url="http://10.77.0.1:8765",
        workspace_root=workspace,
    )
    try:
        window.show()
        app.processEvents()

        label = window._pilot_sync_label.text()
        assert "Workspace" in label
        assert str(Path("Workspace") / "incoming" / "pilot") in label
        assert str(workspace) in window._pilot_sync_label.toolTip()
        assert (workspace / "results").exists()
        assert (workspace / "results" / "realityscan").exists()
        assert (workspace / "reports").exists()

        new_workspace = tmp_path / "new-workspace"
        window._set_workspace_root(new_workspace)
        assert window._pilot_sync_output == new_workspace / "incoming" / "pilot"
        assert str(Path("Workspace") / "incoming" / "pilot") in window._pilot_sync_label.text()
        assert (new_workspace / "results").exists()
        assert (new_workspace / "calibrations").exists()
        assert str(Path("Workspace") / "incoming" / "pilot") in window._pilot_sync_destination_label.text()
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()


def test_unified_analysis_window_ignores_pytest_saved_workspace(tmp_path: Path):
    app = _app()
    from triton_analysis.workspace import REPO_ROOT
    from triton_analysis.gui.triton_analysis_window import TritonAnalysisWindow

    stale_workspace = Path(".pytest-work") / "pytest-of-Thoma" / "pytest-1" / "test_case" / "workspace"
    settings = QSettings("TritonAnalysis", "UnifiedApp")
    settings.setValue("workspace/root", str(stale_workspace))
    settings.setValue("pilot_transfer/output", str(stale_workspace / "incoming" / "pilot"))
    settings.sync()

    window = TritonAnalysisWindow(pilot_transfer_auto_sync=False)
    try:
        window.show()
        app.processEvents()

        assert window._workspace.root == REPO_ROOT / "Workspace"
        assert window._pilot_sync_output == REPO_ROOT / "Workspace" / "incoming" / "pilot"
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()


def test_unified_analysis_window_shows_pilot_sync_transfer_progress(tmp_path: Path):
    app = _app()
    from triton_analysis.gui.triton_analysis_window import TritonAnalysisWindow

    window = TritonAnalysisWindow(
        pilot_transfer_auto_sync=False,
        pilot_transfer_url="http://10.77.0.1:8765",
        pilot_transfer_output=tmp_path / "incoming",
    )
    try:
        window.show()
        app.processEvents()

        window._handle_pilot_sync_progress(
            {
                "event": "copy_start",
                "path": "run_01/stereo_sessions/left_frame.png",
                "size": 4096,
                "index": 2,
                "total_files": 8,
            }
        )
        assert "Receiving 2/8" in window._pilot_sync_progress_label.text()
        assert "left_frame.png" in window._pilot_sync_progress_label.text()

        window._handle_pilot_sync_progress(
            {
                "event": "copy_progress",
                "path": "run_01/stereo_sessions/left_frame.png",
                "size": 4096,
                "file_bytes_copied": 2048,
                "index": 2,
                "total_files": 8,
            }
        )
        assert "50%" in window._pilot_sync_progress_label.text()

        window._handle_pilot_sync_progress(
            {
                "event": "complete",
                "scanned": 8,
                "copied": 2,
                "skipped": 6,
                "bytes_copied": 8192,
            }
        )
        assert "Sync complete" in window._pilot_sync_progress_label.text()
        assert "received 2" in window._pilot_sync_progress_label.text()
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()


def test_unified_analysis_window_uses_clear_pilot_sync_states(tmp_path: Path):
    app = _app()
    from triton_analysis.gui.triton_analysis_window import TritonAnalysisWindow

    window = TritonAnalysisWindow(
        pilot_transfer_auto_sync=False,
        pilot_transfer_url="http://10.77.0.1:8765",
        pilot_transfer_output=tmp_path / "incoming",
    )
    try:
        window.show()
        app.processEvents()

        window._handle_pilot_sync_progress({"event": "watch_start"})
        assert "Pilot Sync: LIVE" in window._pilot_sync_label.text()
        assert window._pilot_sync_state_panel_label.text() == "Pilot Sync: LIVE"
        assert "Waiting for new Pilot recordings" in window._pilot_sync_progress_label.text()
        assert "SYNCING" not in window._pilot_sync_label.text()

        window._handle_pilot_sync_progress({"event": "watch_done", "changed": False})
        assert "Pilot Sync: LIVE" in window._pilot_sync_label.text()
        assert "No new Pilot recordings" in window._pilot_sync_progress_label.text()

        window._handle_pilot_sync_progress({"event": "sync_start"})
        assert "Pilot Sync: CHECKING" in window._pilot_sync_label.text()

        window._handle_pilot_sync_progress(
            {
                "event": "copy_start",
                "path": "run_01/video.mp4",
                "size": 4096,
                "index": 1,
                "total_files": 1,
            }
        )
        assert "Pilot Sync: RECEIVING" in window._pilot_sync_label.text()
        assert "video.mp4" in window._pilot_sync_progress_label.text()
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()


def test_unified_analysis_sync_now_finishes_from_background_worker(tmp_path: Path):
    app = _app()
    from triton_analysis.sync.pilot_transfer import PilotTransferSummary
    from triton_analysis.gui.triton_analysis_window import TritonAnalysisWindow

    def _fake_sync(base_url, destination, *, overwrite=False, timeout=10.0, progress_callback=None):
        if progress_callback is not None:
            progress_callback({"event": "index_done", "scanned": 2, "total_bytes": 42})
            progress_callback({"event": "complete", "scanned": 2, "copied": 1, "skipped": 1, "bytes_copied": 42})
        return PilotTransferSummary(
            base_url=base_url,
            destination=destination,
            scanned=2,
            copied=1,
            skipped=1,
            bytes_copied=42,
        )

    window = TritonAnalysisWindow(
        pilot_transfer_auto_sync=False,
        pilot_transfer_url="http://pilot.test:8765",
        pilot_transfer_output=tmp_path / "incoming",
        pilot_transfer_sync_fn=_fake_sync,
    )
    try:
        window.show()
        app.processEvents()

        window._start_pilot_sync(force=True)
        for _attempt in range(100):
            app.processEvents()
            if not window._pilot_sync_busy and window._pilot_sync_thread is None:
                break

        assert window._pilot_sync_busy is False
        assert window._pilot_sync_thread is None
        assert "Pilot Sync: OK" in window._pilot_sync_label.text()
        assert "received 1" in window._pilot_sync_label.text()
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()


def test_unified_analysis_stays_usable_when_pilot_network_is_unavailable(tmp_path: Path):
    app = _app()
    from triton_analysis.gui.triton_analysis_window import TritonAnalysisWindow

    def _offline_sync(base_url, destination, *, overwrite=False, timeout=10.0, progress_callback=None):
        raise OSError("network unavailable")

    window = TritonAnalysisWindow(
        pilot_transfer_auto_sync=False,
        pilot_transfer_url="http://10.77.0.1:8765",
        pilot_transfer_output=tmp_path / "incoming",
        pilot_transfer_sync_fn=_offline_sync,
    )
    try:
        window.show()
        app.processEvents()

        window._start_pilot_sync(force=True)
        for _attempt in range(100):
            app.processEvents()
            if not window._pilot_sync_busy and window._pilot_sync_thread is None:
                break

        assert window._pilot_sync_busy is False
        assert window._pilot_sync_thread is None
        assert "Pilot Sync: LOST" in window._pilot_sync_label.text()
        assert "network unavailable" in window._pilot_sync_label.text()
        assert window.tabs.count() == len(TritonAnalysisWindow.TAB_KEYS)
        assert window.focus_tab("edna")
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()


def test_edna_count_entry_rows_are_visible_without_table_scroll():
    app = _app()
    from triton_analysis.edna.analysis import DEFAULT_SPECIES
    from triton_analysis.gui.edna_analysis_window import EDNAAnalysisWindow

    window = EDNAAnalysisWindow()
    try:
        window.show()
        app.processEvents()

        table = window.input_table
        assert table.verticalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        assert table.verticalScrollBar().maximum() == 0
        assert table.rowCount() == len(DEFAULT_SPECIES)
        last_row = table.rowCount() - 1
        last_row_bottom = table.rowViewportPosition(last_row) + table.rowHeight(last_row)
        assert last_row_bottom <= table.viewport().height()
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()
