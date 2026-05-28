from pathlib import Path

from analysis_workspace import workspace_label, workspace_paths


def test_workspace_paths_are_stable_under_configured_root(tmp_path: Path):
    workspace = workspace_paths(tmp_path / "analysis-root", create=True)

    assert workspace.pilot_incoming == tmp_path / "analysis-root" / "incoming" / "pilot"
    assert workspace.realityscan_results == tmp_path / "analysis-root" / "results" / "realityscan"
    assert workspace.crab_results == tmp_path / "analysis-root" / "results" / "crab_detection"
    assert workspace.reports == tmp_path / "analysis-root" / "reports"
    assert workspace.pilot_incoming.exists()


def test_workspace_label_hides_machine_specific_root(tmp_path: Path):
    root = tmp_path / "machine-specific"
    path = root / "incoming" / "pilot" / "run_01" / "frame.jpg"

    assert workspace_label(path, root=root) == str(Path("Workspace") / "incoming" / "pilot" / "run_01" / "frame.jpg")
