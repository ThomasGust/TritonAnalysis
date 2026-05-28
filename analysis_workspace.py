"""Portable workspace folders for TritonAnalysis inputs and outputs."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


ENV_WORKSPACE_ROOT = "TRITON_ANALYSIS_WORKSPACE"
DEFAULT_WORKSPACE_NAME = "TritonAnalysisWorkspace"


def default_workspace_root() -> Path:
    """Return the per-machine workspace root without creating it."""
    env_root = os.environ.get(ENV_WORKSPACE_ROOT, "").strip()
    if env_root:
        return Path(env_root).expanduser()
    documents = Path.home() / "Documents"
    return documents / DEFAULT_WORKSPACE_NAME


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


def workspace_label(path: str | Path, root: str | Path | None = None) -> str:
    """Return a portable display label for a path."""
    return workspace_paths(root).label_for(path)
