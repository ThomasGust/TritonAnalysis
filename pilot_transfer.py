"""Pull saved TritonPilot media into the TritonAnalysis inbox."""

from __future__ import annotations

import json
import os
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import quote

from analysis_workspace import workspace_paths


DEFAULT_PILOT_TRANSFER_URL = os.environ.get("TRITON_PILOT_TRANSFER_URL", "http://10.77.0.1:8765")
DEFAULT_INBOX = Path(os.environ.get("TRITON_ANALYSIS_INBOX", str(workspace_paths().pilot_incoming)))


@dataclass(frozen=True)
class PilotTransferFile:
    """One file advertised by the TritonPilot transfer server."""

    path: str
    size: int
    mtime_ns: int


@dataclass
class PilotTransferSummary:
    """Result of a transfer sync."""

    base_url: str
    destination: Path
    scanned: int = 0
    copied: int = 0
    skipped: int = 0
    bytes_copied: int = 0
    dry_run: bool = False
    copied_paths: list[Path] | None = None
    skipped_paths: list[Path] | None = None

    def __post_init__(self) -> None:
        self.destination = Path(self.destination)
        if self.copied_paths is None:
            self.copied_paths = []
        if self.skipped_paths is None:
            self.skipped_paths = []


def _safe_relative_path(raw_path: str) -> PurePosixPath:
    rel = PurePosixPath(str(raw_path or "").replace("\\", "/"))
    if not rel.parts or rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"unsafe transfer path: {raw_path!r}")
    return rel


def _local_path(destination: Path, rel_path: str) -> Path:
    rel = _safe_relative_path(rel_path)
    destination = Path(destination).expanduser().resolve()
    candidate = destination.joinpath(*rel.parts).resolve()
    try:
        candidate.relative_to(destination)
    except ValueError as exc:
        raise ValueError(f"transfer path escapes destination: {rel_path!r}") from exc
    return candidate


def _file_url(base_url: str, rel_path: str) -> str:
    rel = _safe_relative_path(rel_path)
    return f"{base_url.rstrip('/')}/files/" + "/".join(quote(part) for part in rel.parts)


def fetch_pilot_index(base_url: str = DEFAULT_PILOT_TRANSFER_URL, *, timeout: float = 5.0) -> list[PilotTransferFile]:
    """Fetch the file index from TritonPilot."""
    url = f"{base_url.rstrip('/')}/index.json"
    with urllib.request.urlopen(url, timeout=float(timeout)) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if payload.get("type") != "triton-analysis-transfer-index":
        raise RuntimeError("Pilot transfer endpoint did not return a Triton index")
    files = []
    for item in payload.get("files", []):
        files.append(
            PilotTransferFile(
                path=str(item["path"]),
                size=int(item.get("size", 0)),
                mtime_ns=int(item.get("mtime_ns", 0)),
            )
        )
    files.sort(key=lambda item: item.path.lower())
    return files


def file_is_current(path: Path, source: PilotTransferFile) -> bool:
    """Return whether *path* already matches the advertised source file."""
    try:
        stat = path.stat()
    except OSError:
        return False
    return int(stat.st_size) == int(source.size) and int(stat.st_mtime_ns) == int(source.mtime_ns)


def sync_from_pilot(
    base_url: str = DEFAULT_PILOT_TRANSFER_URL,
    destination: str | Path = DEFAULT_INBOX,
    *,
    dry_run: bool = False,
    overwrite: bool = False,
    timeout: float = 10.0,
) -> PilotTransferSummary:
    """Copy new or changed files from TritonPilot into *destination*."""
    destination = Path(destination).expanduser()
    files = fetch_pilot_index(base_url, timeout=timeout)
    summary = PilotTransferSummary(
        base_url=str(base_url).rstrip("/"),
        destination=destination,
        scanned=len(files),
        dry_run=bool(dry_run),
    )

    for source in files:
        target = _local_path(destination, source.path)
        if target.exists() and not overwrite and file_is_current(target, source):
            summary.skipped += 1
            summary.skipped_paths.append(target)
            continue
        if dry_run:
            summary.copied += 1
            summary.bytes_copied += int(source.size)
            summary.copied_paths.append(target)
            continue

        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_name(target.name + ".part")
        request_url = _file_url(base_url, source.path)
        try:
            with urllib.request.urlopen(request_url, timeout=float(timeout)) as response, temp.open("wb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
            if temp.stat().st_size != int(source.size):
                raise RuntimeError(f"Downloaded size mismatch for {source.path}")
            os.replace(temp, target)
        except Exception:
            try:
                temp.unlink()
            except OSError:
                pass
            raise
        if source.mtime_ns:
            os.utime(target, ns=(int(time.time_ns()), int(source.mtime_ns)))
        summary.copied += 1
        summary.bytes_copied += int(source.size)
        summary.copied_paths.append(target)

    return summary
