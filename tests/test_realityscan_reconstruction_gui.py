import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication

from gui.realityscan_reconstruction_window import RealityScanReconstructionWindow
from gui.realityscan_model_viewer_window import RealityScanModelViewerPanel
from gui.style import apply_modern_style


def _app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
        apply_modern_style(app)
    return app


def _pipeline_root(tmp_path: Path) -> Path:
    root = tmp_path / "TritonPilot"
    tools = root / "tools"
    tools.mkdir(parents=True)
    (tools / "realityscan_underwater_pipeline.py").write_text("# test pipeline\n", encoding="utf-8")
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
            "tools.realityscan_underwater_pipeline",
        ]
        assert command[4] == str(session.resolve())
        assert command[command.index("--output") + 1] == str(output)
        assert command[command.index("--stereo-calibration") + 1] == str(calibration.resolve())
        assert command[command.index("--reconstruction-preset") + 1] == "max-detail"
        assert "--metric-scale-from-stereo" in command
        assert "--metric-scale-required" in command
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()


def test_realityscan_gui_alignment_only_enables_tournament(tmp_path: Path):
    app = _app()
    window = RealityScanReconstructionWindow()
    try:
        window.alignment_only_check.setChecked(True)
        command = window.build_command(preview=True)

        assert "--alignment-only" in command
        assert command[command.index("--alignment-tournament") + 1] == "standard"
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
        assert window._output_labels["metric_scale"].text() == str(reports / "metric_scale.json")
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()
