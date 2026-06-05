"""Portable workspace folders for TritonAnalysis inputs and outputs."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


ENV_WORKSPACE_ROOT = "TRITON_ANALYSIS_WORKSPACE"
DEFAULT_WORKSPACE_NAME = "Workspace"
REPO_ROOT = Path(__file__).resolve().parents[1]
_ACTIVE_WORKSPACE_ROOT: Path | None = None
_PILOT_RUN_NAME_RE = re.compile(r"^\d{8}-\d{6}(?:-\d{3})?(?:-\d{2})?$")


def set_active_workspace_root(root: str | Path | None) -> None:
    """Set the process-local workspace root used by applets."""
    global _ACTIVE_WORKSPACE_ROOT
    _ACTIVE_WORKSPACE_ROOT = Path(root).expanduser() if root else None


def default_workspace_root() -> Path:
    """Return the default workspace root without creating it."""
    env_root = os.environ.get(ENV_WORKSPACE_ROOT, "").strip()
    if env_root:
        return Path(env_root).expanduser()
    if _ACTIVE_WORKSPACE_ROOT is not None:
        return _ACTIVE_WORKSPACE_ROOT
    return REPO_ROOT / DEFAULT_WORKSPACE_NAME


@dataclass(frozen=True)
class AnalysisWorkspace:
    """Stable logical folders below one machine-specific root."""

    root: Path

    @property
    def incoming(self) -> Path:
        return self.root / "incoming"

    @property
    def pilot_incoming(self) -> Path:
        return self.incoming / "pilot"

    @property
    def sources(self) -> Path:
        return self.root / "sources"

    @property
    def results(self) -> Path:
        return self.root / "results"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def exports(self) -> Path:
        return self.root / "exports"

    @property
    def calibrations(self) -> Path:
        return self.root / "calibrations"

    @property
    def scratch(self) -> Path:
        return self.root / "scratch"

    @property
    def realityscan_results(self) -> Path:
        return self.results / "realityscan"

    @property
    def crab_results(self) -> Path:
        return self.results / "crab_detection"

    @property
    def coral_results(self) -> Path:
        return self.results / "coral_garden"

    @property
    def color_correction_results(self) -> Path:
        return self.results / "color_correction"

    def ensure(self) -> "AnalysisWorkspace":
        for folder in (
            self.incoming,
            self.pilot_incoming,
            self.sources,
            self.results,
            self.realityscan_results,
            self.crab_results,
            self.coral_results,
            self.color_correction_results,
            self.reports,
            self.exports,
            self.calibrations,
            self.scratch,
        ):
            folder.mkdir(parents=True, exist_ok=True)
        return self

    def label_for(self, path: str | Path) -> str:
        """Return a stable workspace-relative label when possible."""
        candidate = Path(path).expanduser()
        try:
            rel = candidate.resolve().relative_to(self.root.expanduser().resolve())
        except (OSError, ValueError):
            return str(candidate)
        return str(Path("Workspace") / rel)


def workspace_paths(root: str | Path | None = None, *, create: bool = False) -> AnalysisWorkspace:
    """Build workspace folder paths for *root* or the configured default."""
    workspace = AnalysisWorkspace(Path(root).expanduser() if root else default_workspace_root())
    return workspace.ensure() if create else workspace


def _pilot_run_sort_key(path: Path) -> tuple[int, str | int, str]:
    name = Path(path).name
    if _PILOT_RUN_NAME_RE.match(name):
        return (2, name, name.lower())
    try:
        mtime_ns = int(Path(path).stat().st_mtime_ns)
    except OSError:
        mtime_ns = 0
    return (1, mtime_ns, name.lower())


def recent_pilot_run_dirs(root: str | Path | None = None, *, create: bool = False) -> list[Path]:
    """Return synced TritonPilot run folders newest first."""
    incoming = workspace_paths(root, create=create).pilot_incoming
    try:
        children = list(incoming.iterdir())
    except OSError:
        return []
    run_dirs = [
        child
        for child in children
        if child.is_dir() and child.name.lower() != "stereo_sessions"
    ]
    return sorted(run_dirs, key=_pilot_run_sort_key, reverse=True)


def latest_pilot_run_dir(root: str | Path | None = None, *, create: bool = False) -> Path:
    """Return the newest synced Pilot run folder, or the Pilot inbox for legacy flat files."""
    workspace = workspace_paths(root, create=create)
    runs = recent_pilot_run_dirs(root, create=False)
    return runs[0] if runs else workspace.pilot_incoming


def latest_pilot_stereo_sessions_dir(root: str | Path | None = None, *, create: bool = False) -> Path:
    """Return the stereo_sessions folder inside the newest run that has one."""
    workspace = workspace_paths(root, create=create)
    for run_dir in recent_pilot_run_dirs(root, create=False):
        stereo_sessions = run_dir / "stereo_sessions"
        if stereo_sessions.exists():
            return stereo_sessions
    legacy = workspace.pilot_incoming / "stereo_sessions"
    if legacy.exists():
        return legacy
    return latest_pilot_run_dir(root, create=False)


def safe_output_slug(text: str, *, fallback: str = "run") -> str:
    """Return a filesystem-friendly slug for generated output folders."""
    chars: list[str] = []
    for char in str(text or ""):
        if char.isalnum() or char in ("-", "_"):
            chars.append(char)
        else:
            chars.append("_")
    slug = "".join(chars).strip("_")
    return slug or fallback


def fresh_output_subdir(
    parent: str | Path,
    label: str,
    *,
    create: bool = False,
    when: datetime | None = None,
) -> Path:
    """Return a timestamped subfolder that does not already contain outputs."""
    root = Path(parent).expanduser()
    stamp = (when or datetime.now()).strftime("%Y%m%d_%H%M%S")
    stem = f"{safe_output_slug(label)}_{stamp}"
    candidate = root / stem
    if not candidate.exists():
        if create:
            candidate.mkdir(parents=True, exist_ok=False)
        return candidate
    for suffix in range(2, 1000):
        candidate = root / f"{stem}_{suffix:02d}"
        if not candidate.exists():
            if create:
                candidate.mkdir(parents=True, exist_ok=False)
            return candidate
    raise RuntimeError(f"Could not find an unused output folder under {root}")


def workspace_label(path: str | Path, root: str | Path | None = None) -> str:
    """Return a portable display label for a path."""
    return workspace_paths(root).label_for(path)
