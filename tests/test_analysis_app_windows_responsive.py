import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QScrollArea, QTabWidget

from gui.style import apply_modern_style


def _app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
        apply_modern_style(app)
    return app


@pytest.fixture(autouse=True)
def _disable_auto_pilot_sync(monkeypatch):
    monkeypatch.setenv("TRITON_ANALYSIS_AUTO_SYNC", "0")


@pytest.mark.parametrize(
    ("window_path", "min_scroll_areas"),
    [
        ("gui.triton_analysis_window.TritonAnalysisWindow", 8),
        ("gui.crab_detection_window.CrabDetectionWindow", 3),
        ("gui.edna_analysis_window.EDNAAnalysisWindow", 1),
        ("gui.iceberg_tracking_window.IcebergTrackingWindow", 1),
        ("gui.coral_garden_model_window.CoralGardenModelWindow", 0),
        ("gui.iceberg_measurement_window.IcebergMeasurementWindow", 2),
        ("gui.planar_height_measurement_window.PlanarHeightMeasurementWindow", 2),
        ("gui.multi_rect_length_measurement_window.MultiRectLengthMeasurementWindow", 2),
        ("gui.stereo_calibration_window.StereoCalibrationWindow", 1),
        ("gui.stereo_depth_window.StereoDepthWindow", 1),
        ("gui.stereo_segment_measurement_window.StereoSegmentMeasurementWindow", 1),
        ("gui.stereo_iceberg_measurement_window.StereoIcebergMeasurementWindow", 1),
        ("gui.realityscan_reconstruction_window.RealityScanReconstructionWindow", 1),
        ("gui.realityscan_model_viewer_window.RealityScanModelViewerWindow", 1),
        ("color_corr.MainWindow", 3),
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
    from gui.multi_rect_length_measurement_window import MultiRectLengthMeasurementWindow

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
    from gui.triton_analysis_window import TritonAnalysisWindow

    window = TritonAnalysisWindow()
    try:
        window.show()
        app.processEvents()

        tabs = window.findChild(QTabWidget)
        assert tabs is not None
        assert tabs.count() == 8
        assert [tabs.tabText(index) for index in range(tabs.count())] == [
            "Coral Reconstruction",
            "Crab Detection",
            "Stereo Iceberg Length",
            "Iceberg Tracking",
            "eDNA Analysis",
            "Stereo Calibration",
            "Backup Coral Measurement",
            "Backup Iceberg Measurement",
        ]
        assert window.focus_tab("backup-coral-measurement") is True
        assert tabs.currentIndex() == 6
        assert window.focus_tab("crab") is True
        assert tabs.currentIndex() == 1
        assert window.focus_tab("missing") is False
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()


def test_unified_analysis_window_shows_pilot_sync_destination(tmp_path: Path):
    app = _app()
    from gui.triton_analysis_window import TritonAnalysisWindow

    destination = tmp_path / "incoming"
    window = TritonAnalysisWindow(
        pilot_transfer_auto_sync=False,
        pilot_transfer_url="http://127.0.0.1:8765",
        pilot_transfer_output=destination,
    )
    try:
        window.show()
        app.processEvents()

        assert "Pilot Sync: OFF" in window._pilot_sync_label.text()
        assert "http://127.0.0.1:8765" in window._pilot_sync_label.text()
        assert str(destination) in window._pilot_sync_label.text()
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()


def test_unified_analysis_window_uses_workspace_relative_sync_label(tmp_path: Path):
    app = _app()
    from gui.triton_analysis_window import TritonAnalysisWindow

    workspace = tmp_path / "analysis-workspace"
    window = TritonAnalysisWindow(
        pilot_transfer_auto_sync=False,
        pilot_transfer_url="http://127.0.0.1:8765",
        workspace_root=workspace,
    )
    try:
        window.show()
        app.processEvents()

        label = window._pilot_sync_label.text()
        assert "Workspace" in label
        assert str(Path("Workspace") / "incoming" / "pilot") in label
        assert str(workspace) in window._pilot_sync_label.toolTip()

        new_workspace = tmp_path / "new-workspace"
        window._set_workspace_root(new_workspace)
        assert window._pilot_sync_output == new_workspace / "incoming" / "pilot"
        assert str(Path("Workspace") / "incoming" / "pilot") in window._pilot_sync_label.text()
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()


def test_edna_count_entry_rows_are_visible_without_table_scroll():
    app = _app()
    from edna_analysis import DEFAULT_SPECIES
    from gui.edna_analysis_window import EDNAAnalysisWindow

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
