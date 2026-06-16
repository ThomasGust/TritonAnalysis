import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication

from triton_analysis.gui.realityscan_reconstruction_window import RealityScanReconstructionWindow
from triton_analysis.gui.realityscan_model_viewer_window import RealityScanModelViewerPanel
from triton_analysis.gui.style import apply_modern_style


def _app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
        apply_modern_style(app)
    return app


def _pipeline_root(tmp_path: Path) -> Path:
    root = tmp_path / "TritonAnalysis"
    pipeline = root / "triton_analysis" / "realityscan"
    pipeline.mkdir(parents=True)
    (pipeline / "underwater_pipeline.py").write_text("# test pipeline\n", encoding="utf-8")
    return root


def _stereo_session(tmp_path: Path) -> tuple[Path, Path]:
    session = tmp_path / "recordings" / "stereo_sessions" / "session-a"
    session.mkdir(parents=True)
    (session / "manifest.json").write_text('{"frames": []}\n', encoding="utf-8")
    calibration = session / "stereo_calibration.json"
    calibration.write_text("{}\n", encoding="utf-8")
    return session, calibration


def test_realityscan_gui_builds_stereo_pipeline_command(tmp_path: Path):
    app = _app()
    pipeline_root = _pipeline_root(tmp_path)
    session, calibration = _stereo_session(tmp_path)
    output = tmp_path / "analysis-results" / "scan"

    window = RealityScanReconstructionWindow()
    try:
        assert window.tabs.tabText(0) == "Reconstruction"
        assert window.tabs.tabText(1) == "Model Viewer"
        assert isinstance(window.model_viewer_panel, RealityScanModelViewerPanel)

        window.pipeline_root_edit.setText(str(pipeline_root))
        window.session_edit.setText(str(session))
        window.calibration_edit.setText(str(calibration))
        window.output_edit.setText(str(output))

        command = window.build_command()

        assert command[:4] == [
            os.sys.executable,
            "-u",
            "-m",
            "triton_analysis.realityscan.underwater_pipeline",
        ]
        assert command[4] == str(session.resolve())
        assert command[command.index("--output") + 1] == str(output)
        assert command[command.index("--stereo-calibration") + 1] == str(calibration.resolve())
        assert command[command.index("--reconstruction-preset") + 1] == "max-detail"
        assert "--alignment-tournament" not in command
        assert "--legacy-enhanced-default" not in command
        assert "--metric-scale-from-stereo" in command
        assert "--metric-scale-required" in command
        assert "--texture-layers" in command
        assert "--connectivity-report" in command
        assert command[command.index("--min-good-component-ratio") + 1] == "0.12"
        assert "--fail-on-poor-alignment" in command
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()


def test_realityscan_gui_blank_output_uses_fresh_subfolder(tmp_path: Path, monkeypatch):
    app = _app()
    session, calibration = _stereo_session(tmp_path)
    monkeypatch.setenv("TRITON_ANALYSIS_WORKSPACE", str(tmp_path / "workspace"))

    window = RealityScanReconstructionWindow()
    try:
        window.session_edit.setText(str(session))
        window.calibration_edit.setText(str(calibration))

        first = Path(window._output_workspace_for_command(preview=False))
        first.mkdir(parents=True)
        second = Path(window._output_workspace_for_command(preview=False))

        assert first.parent == second.parent
        assert first != second
        assert first.name.startswith("session-a_")
        assert second.name.startswith("session-a_")
        assert "<new subfolder each run>" in window.command_preview_text()
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()


def test_realityscan_gui_alignment_only_keeps_fast_single_variant(tmp_path: Path):
    app = _app()
    window = RealityScanReconstructionWindow()
    try:
        window.alignment_only_check.setChecked(True)
        command = window.build_command(preview=True)

        assert "--alignment-only" in command
        assert "--alignment-tournament" not in command
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()


def test_realityscan_gui_can_disable_color_texture_layers(tmp_path: Path):
    app = _app()
    window = RealityScanReconstructionWindow()
    try:
        window.texture_layers_check.setChecked(False)
        command = window.build_command(preview=True)

        assert "--no-texture-layers" in command
        assert "--texture-layers" not in command
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()


def test_realityscan_gui_can_enable_component_diagnostics(tmp_path: Path):
    app = _app()
    window = RealityScanReconstructionWindow()
    try:
        window.component_diagnostics_check.setChecked(True)
        command = window.build_command(preview=True)

        assert "--export-component-diagnostics" in command
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()


def test_realityscan_gui_legacy_fast_variant_switches_default(tmp_path: Path):
    app = _app()
    window = RealityScanReconstructionWindow()
    try:
        index = window.fast_variant_combo.findData("legacy_enhanced")
        assert index >= 0
        window.fast_variant_combo.setCurrentIndex(index)

        command = window.build_command(preview=True)

        assert "--legacy-enhanced-default" in command
        assert "--alignment-tournament" not in command
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()


def test_realityscan_gui_populates_known_workspace_outputs(tmp_path: Path):
    app = _app()
    workspace = tmp_path / "scan"
    reports = workspace / "reports"
    reports.mkdir(parents=True)
    for path in (
        workspace / "selection_contact_sheet.jpg",
        reports / "final_overview.html",
        reports / "alignment_components.csv",
        workspace / "underwater_model.obj",
        workspace / "underwater_model_metric.obj",
        reports / "metric_scale.json",
        workspace / "manifest.json",
    ):
        path.write_text("test\n", encoding="utf-8")

    window = RealityScanReconstructionWindow()
    try:
        window._handle_log_line(f"Output workspace: {workspace}")

        assert window._output_labels["workspace"].text() == str(workspace)
        assert window._output_buttons["metric_model"].isEnabled()
        assert window.view_metric_model_btn.isEnabled()
        assert window._output_labels["components"].text() == str(reports / "alignment_components.csv")
        assert window._output_labels["metric_scale"].text() == str(reports / "metric_scale.json")
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()


def test_realityscan_model_viewer_tab_is_selected_before_loading(tmp_path: Path):
    app = _app()
    model = tmp_path / "underwater_model_metric.obj"
    model.write_text("v 0 0 0\n", encoding="utf-8")

    window = RealityScanReconstructionWindow()
    calls: list[bool] = []

    def fake_start_viewer() -> None:
        calls.append(window.tabs.currentWidget() is window.model_viewer_panel)

    window.model_viewer_panel._start_viewer = fake_start_viewer
    try:
        window._output_paths["metric_model"] = model
        window._launch_model_viewer()
        app.processEvents()

        assert calls == [True]
        assert window.tabs.currentWidget() is window.model_viewer_panel
        assert window.model_viewer_panel.model_edit.text() == str(model)
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()
