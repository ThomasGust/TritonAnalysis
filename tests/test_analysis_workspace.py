from pathlib import Path
from datetime import datetime

from triton_analysis.workspace import (
    DEFAULT_WORKSPACE_NAME,
    ENV_WORKSPACE_ROOT,
    REPO_ROOT,
    default_workspace_root,
    fresh_output_subdir,
    latest_pilot_run_dir,
    latest_pilot_stereo_sessions_dir,
    recent_pilot_run_dirs,
    set_active_workspace_root,
    workspace_label,
    workspace_paths,
)


def test_default_workspace_root_is_repo_local(monkeypatch):
    monkeypatch.delenv(ENV_WORKSPACE_ROOT, raising=False)
    set_active_workspace_root(None)
    try:
        assert default_workspace_root() == REPO_ROOT / DEFAULT_WORKSPACE_NAME
    finally:
        set_active_workspace_root(None)


def test_workspace_paths_are_stable_under_configured_root(tmp_path: Path):
    workspace = workspace_paths(tmp_path / "analysis-root", create=True)

    assert workspace.pilot_incoming == tmp_path / "analysis-root" / "incoming" / "pilot"
    assert workspace.realityscan_results == tmp_path / "analysis-root" / "results" / "realityscan"
    assert workspace.reports == tmp_path / "analysis-root" / "reports"
    assert workspace.pilot_incoming.exists()
    assert workspace.results.exists()
    assert workspace.realityscan_results.exists()
    assert workspace.coral_results.exists()
    assert workspace.color_correction_results.exists()


def test_workspace_label_hides_machine_specific_root(tmp_path: Path):
    root = tmp_path / "machine-specific"
    path = root / "incoming" / "pilot" / "run_01" / "frame.jpg"

    assert workspace_label(path, root=root) == str(Path("Workspace") / "incoming" / "pilot" / "run_01" / "frame.jpg")


def test_fresh_output_subdir_uses_timestamp_and_avoids_existing_folder(tmp_path: Path):
    parent = tmp_path / "results"
    when = datetime(2026, 5, 27, 10, 30, 45)

    first = fresh_output_subdir(parent, "My Scan!", when=when)
    first.mkdir(parents=True)
    second = fresh_output_subdir(parent, "My Scan!", when=when)

    assert first == parent / "My_Scan_20260527_103045"
    assert second == parent / "My_Scan_20260527_103045_02"


def test_latest_pilot_run_helpers_prefer_new_timestamped_runs(tmp_path: Path):
    workspace = workspace_paths(tmp_path / "analysis-root", create=True)
    older = workspace.pilot_incoming / "20260605-120000"
    newer = workspace.pilot_incoming / "20260605-130000"
    legacy = workspace.pilot_incoming / "run_01"
    root_stereo = workspace.pilot_incoming / "stereo_sessions"
    for folder in (older, newer, legacy, root_stereo):
        folder.mkdir(parents=True)
    (older / "stereo_sessions").mkdir()
    (newer / "stereo_sessions").mkdir()

    assert recent_pilot_run_dirs(workspace.root)[:3] == [newer, older, legacy]
    assert latest_pilot_run_dir(workspace.root) == newer
    assert latest_pilot_stereo_sessions_dir(workspace.root) == newer / "stereo_sessions"


def test_latest_pilot_run_helpers_fall_back_to_inbox_for_flat_files(tmp_path: Path):
    workspace = workspace_paths(tmp_path / "analysis-root", create=True)
    (workspace.pilot_incoming / "frame.png").write_text("image", encoding="utf-8")

    assert recent_pilot_run_dirs(workspace.root) == []
    assert latest_pilot_run_dir(workspace.root) == workspace.pilot_incoming
    assert latest_pilot_stereo_sessions_dir(workspace.root) == workspace.pilot_incoming
