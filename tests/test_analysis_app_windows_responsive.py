import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import QSettings, Qt
from PyQt6.QtWidgets import QApplication, QScrollArea, QTabWidget, QToolButton

from triton_analysis.gui.style import apply_modern_style
from triton_analysis.workspace import set_active_workspace_root


def _app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
        apply_modern_style(app)
    return app


def _assert_table_rows_visible_without_scroll(table) -> None:
    assert table.verticalScrollBar().maximum() == 0
    assert table.horizontalScrollBar().maximum() == 0
    last_row = table.rowCount() - 1
    last_row_bottom = table.rowViewportPosition(last_row) + table.rowHeight(last_row)
    assert last_row_bottom <= table.viewport().height()

    metrics = table.fontMetrics()
    species_column_width = table.columnWidth(0)
    for row_index in range(table.rowCount()):
        item = table.item(row_index, 0)
        assert item is not None
        for line in item.text().splitlines():
            assert metrics.horizontalAdvance(line) <= species_column_width


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
        ("triton_analysis.gui.triton_analysis_window.TritonAnalysisWindow", 5),
        ("triton_analysis.gui.edna_analysis_window.EDNAAnalysisWindow", 1),
        ("triton_analysis.gui.crab_counter_window.CrabCounterWindow", 1),
        ("triton_analysis.gui.crab_dataset_generator_window.CrabDatasetGeneratorWindow", 1),
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


def test_crab_counter_params_are_locked_by_default(tmp_path: Path):
    app = _app()
    from triton_analysis.gui.crab_counter_window import CrabCounterWindow

    window = CrabCounterWindow(workspace_root=tmp_path / "workspace")
    try:
        window.show()
        app.processEvents()

        locked_widgets = (
            window.preprocess_mode_combo,
            window.model_edit,
            window.reasoning_effort_combo,
            window.analysis_flow_combo,
            window.threshold_spin,
            window.margin_spin,
            window.output_root_edit,
        )
        assert window.unlock_params_check.isChecked() is False
        assert all(widget.isEnabled() is False for widget in locked_widgets)
        assert window.target_edit.isEnabled() is True

        window.unlock_params_check.setChecked(True)
        app.processEvents()

        assert all(widget.isEnabled() is True for widget in locked_widgets)
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
        assert tabs.count() == 10
        all_tab_names = [
            "Coral Reconstruction",
            "Stereo Iceberg Length",
            "Iceberg Tracking",
            "eDNA Analysis",
            "Crab Counter",
            "Crab Dataset",
            "Stereo Calibration",
            "Backup Coral Measurement",
            "Backup Iceberg Measurement",
            "SSH",
        ]
        competition_tab_names = all_tab_names[:5]
        assert [tabs.tabText(index) for index in range(tabs.count())] == all_tab_names
        assert [tabs.tabText(index) for index in range(tabs.count()) if tabs.isTabVisible(index)] == competition_tab_names

        more_btn = window.findChild(QToolButton, "advancedTabsButton")
        assert more_btn is not None
        assert more_btn.isChecked() is False
        more_btn.click()
        app.processEvents()

        assert more_btn.isChecked() is True
        assert [tabs.tabText(index) for index in range(tabs.count()) if tabs.isTabVisible(index)] == all_tab_names

        more_btn.click()
        app.processEvents()

        assert more_btn.isChecked() is False
        assert [tabs.tabText(index) for index in range(tabs.count()) if tabs.isTabVisible(index)] == competition_tab_names
        assert window.focus_tab("crab") is True
        assert tabs.currentIndex() == 4
        assert window.focus_tab("crab-dataset") is True
        assert tabs.currentIndex() == 5
        assert tabs.isTabVisible(5) is True
        assert window.focus_tab("backup-coral-measurement") is True
        assert tabs.currentIndex() == 7
        assert window.focus_tab("terminal") is True
        assert tabs.currentIndex() == 9
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


def test_edna_input_and_output_rows_are_visible_without_table_scroll():
    app = _app()
    from triton_analysis.edna.analysis import DEFAULT_SPECIES
    from triton_analysis.gui.edna_analysis_window import EDNAAnalysisWindow

    window = EDNAAnalysisWindow()
    try:
        window.resize(1366, 768)
        window.show()
        app.processEvents()

        assert window.input_table.verticalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        assert window.judge_preview.table.verticalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        assert window.input_table.rowCount() == len(DEFAULT_SPECIES)
        assert window.judge_preview.table.rowCount() == len(DEFAULT_SPECIES)
        _assert_table_rows_visible_without_scroll(window.input_table)
        _assert_table_rows_visible_without_scroll(window.judge_preview.table)
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()
