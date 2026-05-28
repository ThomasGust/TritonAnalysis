"""Qt worker for automatic TritonPilot media transfer."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

from PyQt6.QtCore import QObject, pyqtSignal

from pilot_transfer import PilotTransferSummary, sync_from_pilot


SyncFn = Callable[..., PilotTransferSummary]


class PilotTransferSyncWorker(QObject):
    """Run one Pilot transfer sync outside the UI thread."""

    finished = pyqtSignal(object)

    def __init__(
        self,
        *,
        base_url: str,
        destination: str | Path,
        timeout: float = 2.0,
        overwrite: bool = False,
        sync_fn: SyncFn = sync_from_pilot,
    ):
        super().__init__()
        self.base_url = str(base_url).rstrip("/")
        self.destination = Path(destination).expanduser()
        self.timeout = float(timeout)
        self.overwrite = bool(overwrite)
        self.sync_fn = sync_fn

    def run(self) -> None:
        started = time.time()
        try:
            summary = self.sync_fn(
                self.base_url,
                self.destination,
                overwrite=self.overwrite,
                timeout=self.timeout,
            )
            self.finished.emit(
                {
                    "ok": True,
                    "summary": summary,
                    "started": started,
                    "finished": time.time(),
                }
            )
        except Exception as exc:
            self.finished.emit(
                {
                    "ok": False,
                    "error": str(exc),
                    "started": started,
                    "finished": time.time(),
                }
            )
