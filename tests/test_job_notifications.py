"""Tests for cross-tab job badges, toasts, and the Activity panel."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import QApplication

from triton_analysis.gui.job_notifications import JobStatusTabBar
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
def _isolate_settings():
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


def _make_window(tmp_path):
    from triton_analysis.gui.triton_analysis_window import TritonAnalysisWindow

    window = TritonAnalysisWindow(pilot_transfer_auto_sync=False, workspace_root=tmp_path / "ws")
    window.show()
    _app().processEvents()
    return window


def _tone_for(window, key: str) -> str:
    tab_bar = window.tabs.tabBar()
    assert isinstance(tab_bar, JobStatusTabBar)
    index = window.tabs.indexOf(window._windows[key])
    return tab_bar.index_tone(index)


def test_running_job_badges_its_tab(tmp_path):
    app = _app()
    window = _make_window(tmp_path)
    try:
        # Default tab is coral-reconstruction, so crab-counter is not current.
        job = window._job_center.start("crab-counter", "Crab Counter")
        app.processEvents()
        assert _tone_for(window, "crab-counter") == "running"
        assert window._activity_button.text() == "Activity (1)"

        job.succeed("7 crabs")
        app.processEvents()
        assert _tone_for(window, "crab-counter") == "ok"
        assert window._activity_button.text() == "Activity"
    finally:
        window.close()
        app.processEvents()


def test_finish_on_other_tab_shows_toast_and_click_focuses(tmp_path):
    app = _app()
    window = _make_window(tmp_path)
    try:
        job = window._job_center.start("stereo-calibration", "Stereo Calibration")
        job.fail("RMS too high")
        app.processEvents()

        assert len(window._toast_stack._toasts) == 1
        assert _tone_for(window, "stereo-calibration") == "alert"

        toast = window._toast_stack._toasts[0]
        toast.clicked.emit(toast.key)
        app.processEvents()
        # Clicking the toast jumps to the originating tab.
        assert window.tabs.currentIndex() == window.tabs.indexOf(window._windows["stereo-calibration"])
    finally:
        window.close()
        app.processEvents()


def test_viewing_tab_clears_its_finished_badge(tmp_path):
    app = _app()
    window = _make_window(tmp_path)
    try:
        job = window._job_center.start("crab-dataset", "Crab Dataset")
        job.succeed("done")
        app.processEvents()
        assert _tone_for(window, "crab-dataset") == "ok"

        window.focus_tab("crab-dataset")
        app.processEvents()
        assert _tone_for(window, "crab-dataset") == ""
    finally:
        window.close()
        app.processEvents()


def test_finish_while_actively_watching_is_silent(tmp_path, monkeypatch):
    app = _app()
    window = _make_window(tmp_path)
    try:
        monkeypatch.setattr(window, "isActiveWindow", lambda: True)
        window.focus_tab("crab-counter")
        app.processEvents()

        job = window._job_center.start("crab-counter", "Crab Counter")
        job.succeed("3 crabs")
        app.processEvents()

        assert len(window._toast_stack._toasts) == 0
        assert job.acknowledged is True
        assert _tone_for(window, "crab-counter") == ""
    finally:
        window.close()
        app.processEvents()


def test_activity_button_counts_active_jobs(tmp_path):
    app = _app()
    window = _make_window(tmp_path)
    try:
        a = window._job_center.start("crab-counter", "Crab Counter")
        window._job_center.start("coral-reconstruction", "Reconstruction")
        app.processEvents()
        assert window._activity_button.text() == "Activity (2)"

        a.succeed("done")
        app.processEvents()
        assert window._activity_button.text() == "Activity (1)"
    finally:
        window.close()
        app.processEvents()


def test_activity_panel_lists_running_and_recent_jobs(tmp_path):
    app = _app()
    window = _make_window(tmp_path)
    try:
        window._job_center.start("crab-counter", "Crab Counter")
        done = window._job_center.start("stereo-calibration", "Stereo Calibration")
        done.succeed("ok")
        app.processEvents()

        window._refresh_activity_panel()
        rows = window._activity_panel._rows_layout.count()
        assert rows == 2
    finally:
        window.close()
        app.processEvents()


def test_crab_counter_failure_reaches_main_window_badge(tmp_path):
    app = _app()
    window = _make_window(tmp_path)
    try:
        crab = window._windows["crab-counter"]
        crab._begin_job("Crab Counter")
        app.processEvents()
        assert _tone_for(window, "crab-counter") == "running"

        crab._finish_analysis({"ok": False, "error": "boom"})
        app.processEvents()
        assert _tone_for(window, "crab-counter") == "alert"
        jobs = window._job_center.jobs_for("crab-counter")
        assert jobs and jobs[-1].detail == "boom"
    finally:
        window.close()
        app.processEvents()


def test_reconstruction_success_reaches_main_window_badge(tmp_path):
    app = _app()
    from PyQt6.QtCore import QProcess

    window = _make_window(tmp_path)
    try:
        recon = window._windows["coral-reconstruction"]
        recon._begin_job("Coral Reconstruction")
        app.processEvents()

        recon._on_process_finished(0, QProcess.ExitStatus.NormalExit)
        app.processEvents()

        jobs = window._job_center.jobs_for("coral-reconstruction")
        assert jobs and jobs[-1].state.value == "ok"
        # Reconstruction is the default current tab, so finishing it badges only
        # when the window is not active (offscreen tests are not active).
        assert _tone_for(window, "coral-reconstruction") in {"ok", ""}
    finally:
        window.close()
        app.processEvents()


def test_sound_toggle_persists_to_settings(tmp_path):
    app = _app()
    window = _make_window(tmp_path)
    try:
        assert window._notify_sound_enabled is False
        window._set_notify_sound_enabled(True)
        assert window._notify_sound_enabled is True
        assert QSettings("TritonAnalysis", "UnifiedApp").value("notifications/sound") == "1"
    finally:
        window.close()
        app.processEvents()
