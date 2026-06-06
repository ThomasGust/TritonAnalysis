"""Qt worker for automatic TritonPilot media transfer."""

from __future__ import annotations

import inspect
import time
from pathlib import Path
from typing import Callable

from PyQt6.QtCore import QObject, pyqtSignal

from triton_analysis.sync.pilot_transfer import PilotTransferEvent, PilotTransferSummary, sync_from_pilot, wait_for_pilot_change


SyncFn = Callable[..., PilotTransferSummary]
EventFn = Callable[..., PilotTransferEvent]


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
        watch_for_changes: bool = False,
        since_event_id: int = 0,
        event_timeout: float = 20.0,
        event_fn: EventFn = wait_for_pilot_change,
    ):
        super().__init__()
        self.base_url = str(base_url).rstrip("/")
        self.destination = Path(destination).expanduser()
        self.timeout = float(timeout)
        self.overwrite = bool(overwrite)
        self.sync_fn = sync_fn
        self.watch_for_changes = bool(watch_for_changes)
        self.since_event_id = int(since_event_id or 0)
        self.event_timeout = float(event_timeout)
        self.event_fn = event_fn

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

    def _call_event_fn(self) -> PilotTransferEvent:
        kwargs = {
            "since_event_id": self.since_event_id,
            "timeout": self.event_timeout,
        }
        try:
            signature = inspect.signature(self.event_fn)
        except (TypeError, ValueError):
            kwargs["request_timeout"] = self.event_timeout + max(2.0, self.timeout)
        else:
            if "request_timeout" in signature.parameters or any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()
            ):
                kwargs["request_timeout"] = self.event_timeout + max(2.0, self.timeout)
        return self.event_fn(self.base_url, **kwargs)

    def run(self) -> None:
        started = time.time()
        try:
            event_result: PilotTransferEvent | None = None
            if self.watch_for_changes:
                self.progress.emit(
                    {
                        "event": "watch_start",
                        "base_url": self.base_url,
                        "since_event_id": self.since_event_id,
                        "timeout_s": self.event_timeout,
                        "time": started,
                    }
                )
                try:
                    event_result = self._call_event_fn()
                except Exception as exc:
                    self.progress.emit(
                        {
                            "event": "watch_error",
                            "error": str(exc),
                            "base_url": self.base_url,
                            "time": time.time(),
                        }
                    )
                else:
                    self.progress.emit(
                        {
                            "event": "watch_done",
                            "changed": bool(event_result.changed),
                            "event_id": int(event_result.event_id),
                            "file_count": int(event_result.file_count),
                            "total_bytes": int(event_result.total_bytes),
                            "time": time.time(),
                        }
                    )
                    if not event_result.changed:
                        self.finished.emit(
                            {
                                "ok": True,
                                "no_change": True,
                                "event": event_result,
                                "event_id": int(event_result.event_id),
                                "started": started,
                                "finished": time.time(),
                            }
                        )
                        return

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
                    "event": event_result,
                    "event_id": int(event_result.event_id) if event_result is not None else 0,
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
