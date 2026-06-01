from pathlib import Path
from datetime import datetime

from triton_analysis.workspace import (
    DEFAULT_WORKSPACE_NAME,
    ENV_WORKSPACE_ROOT,
    REPO_ROOT,
    default_workspace_root,
    fresh_output_subdir,
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
    assert workspace.crab_results == tmp_path / "analysis-root" / "results" / "crab_detection"
    assert workspace.reports == tmp_path / "analysis-root" / "reports"
    assert workspace.pilot_incoming.exists()
    assert workspace.results.exists()
    assert workspace.realityscan_results.exists()
    assert workspace.crab_results.exists()
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
