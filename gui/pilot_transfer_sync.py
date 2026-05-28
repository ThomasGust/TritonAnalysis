"""Qt worker for automatic TritonPilot media transfer."""

from __future__ import annotations

import inspect
import time
from pathlib import Path
from typing import Callable

from PyQt6.QtCore import QObject, pyqtSignal

from pilot_transfer import PilotTransferSummary, sync_from_pilot


SyncFn = Callable[..., PilotTransferSummary]


class PilotTransferSyncWorker(QObject):
    """Run one Pilot transfer sync outside the UI thread."""

    progress = pyqtSignal(object)
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

    def _sync_accepts_progress(self) -> bool:
        try:
            signature = inspect.signature(self.sync_fn)
        except (TypeError, ValueError):
            return True
        return "progress_callback" in signature.parameters or any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()
        )

    def _emit_progress(self, payload: object) -> None:
        self.progress.emit(payload if isinstance(payload, dict) else {"event": "progress", "payload": payload})

    def run(self) -> None:
        started = time.time()
        try:
            self.progress.emit(
                {
                    "event": "sync_start",
                    "base_url": self.base_url,
                    "destination": str(self.destination),
                    "time": started,
                }
            )
            kwargs = {
                "overwrite": self.overwrite,
                "timeout": self.timeout,
            }
            if self._sync_accepts_progress():
                kwargs["progress_callback"] = self._emit_progress
            summary = self.sync_fn(
                self.base_url,
                self.destination,
                **kwargs,
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
            self.progress.emit(
                {
                    "event": "sync_error",
                    "error": str(exc),
                    "base_url": self.base_url,
                    "destination": str(self.destination),
                    "time": time.time(),
                }
            )
            self.finished.emit(
                {
                    "ok": False,
                    "error": str(exc),
                    "started": started,
                    "finished": time.time(),
                }
            )
