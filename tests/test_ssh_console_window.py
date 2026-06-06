import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication

from triton_analysis.apps.main_triton_analysis import build_parser
from triton_analysis.gui.ssh_console_window import (
    SshConsolePage,
    SshPreset,
    default_analysis_ssh_presets,
)


def _app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_ssh_console_applies_presets_and_cleans_output():
    app = _app()
    page = SshConsolePage(presets=[SshPreset("Pilot", "10.77.0.1", "analysis-user")])
    try:
        app.processEvents()

        assert page.host_edit.text() == "10.77.0.1"
        assert page.user_edit.text() == "analysis-user"
        assert page.port_spin.value() == 22
        assert page._clean_output("\x1b[31mred\x1b[0m\r\nnext\rline") == "red\nnext\nline"
    finally:
        page.close()
        page.deleteLater()
        app.processEvents()


def test_ssh_console_requires_host_and_user():
    app = _app()
    page = SshConsolePage()
    try:
        page.connect_to_host()

        assert "required" in page.status_label.text()
        assert page.connect_btn.isEnabled() is True
        assert page.send_btn.isEnabled() is False
    finally:
        page.close()
        page.deleteLater()
        app.processEvents()


def test_default_analysis_ssh_presets_include_pilot_link_and_routed_rov():
    presets = default_analysis_ssh_presets(local_user="analysis-user")

    assert [(preset.name, preset.host, preset.username) for preset in presets[:2]] == [
        ("Pilot Link", "10.77.0.1", "analysis-user"),
        ("ROV if Routed", "192.168.1.4", "triton"),
    ]


def test_unified_app_parser_accepts_ssh_initial_tab():
    args = build_parser().parse_args(["--tab", "ssh"])

    assert args.tab == "ssh"
