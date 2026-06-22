"""Central registry that tracks long-running analysis jobs across every tab.

Each analysis window reports the lifecycle of its background work to a shared
:class:`JobCenter`.  The unified window listens to the center's signals and
renders the aggregate (per-tab status dots, finish toasts, and an Activity
panel) so an operator never loses track of a process that finished on a tab
they are not currently looking at.

The module deliberately depends only on ``PyQt6.QtCore`` so it stays trivial to
unit test headlessly and so standalone applets can import it without pulling in
widget code.  When a window runs on its own (or inside a test) it is simply
constructed without a center and every report becomes a no-op.

All reporting is expected to happen on the GUI thread.  The analysis windows
already marshal their worker ``QThread``/``QProcess`` callbacks onto the GUI
thread before they touch any widget, and the same call sites report here, so no
extra locking is required.
"""

from __future__ import annotations

import time
from enum import Enum

from PyQt6.QtCore import QObject, pyqtSignal


class JobState(Enum):
    """Lifecycle state of a tracked job.

    The values map onto the shared ``tone`` vocabulary used by the applet
    stylesheet (see ``gui/style.py``) so badges and toasts can reuse the same
    colors as the in-tab status labels.
    """

    RUNNING = "running"
    SUCCESS = "ok"
    WARNING = "warn"
    FAILED = "alert"

    @property
    def is_terminal(self) -> bool:
        return self is not JobState.RUNNING


# Visual priority when a single tab needs to collapse to one badge tone.
_STATE_PRIORITY = {
    JobState.RUNNING: 3,
    JobState.FAILED: 2,
    JobState.WARNING: 1,
    JobState.SUCCESS: 0,
}


def state_priority(state: JobState) -> int:
    """Return the badge priority for *state* (higher wins)."""
    return _STATE_PRIORITY.get(state, 0)


class Job:
    """A single tracked unit of background work.

    Windows obtain a :class:`Job` from :meth:`JobCenter.start` and then call
    :meth:`progress`, :meth:`succeed`, :meth:`warn`, or :meth:`fail` from the
    same call sites where they already update their in-tab status.  Mutating a
    job notifies its owning center, which re-emits the appropriate signal.
    """

    __slots__ = (
        "job_id",
        "key",
        "title",
        "state",
        "detail",
        "percent",
        "started_at",
        "finished_at",
        "acknowledged",
        "_center",
    )

    def __init__(self, center: "JobCenter", job_id: int, key: str, title: str):
        self._center = center
        self.job_id = job_id
        self.key = str(key or "")
        self.title = str(title or "Job")
        self.state = JobState.RUNNING
        self.detail = ""
        self.percent: int | None = None
        self.started_at = time.monotonic()
        self.finished_at: float | None = None
        # Whether the operator has already seen the terminal result (used to
        # decide whether a finished tab still deserves a badge).
        self.acknowledged = False

    @property
    def is_running(self) -> bool:
        return self.state is JobState.RUNNING

    @property
    def is_finished(self) -> bool:
        return self.state.is_terminal

    def elapsed(self) -> float:
        """Seconds the job has been (or was) running."""
        end = self.finished_at if self.finished_at is not None else time.monotonic()
        return max(0.0, end - self.started_at)

    def progress(self, detail: str = "", percent: int | float | None = None) -> None:
        """Report incremental progress while the job is still running."""
        if self.state is not JobState.RUNNING:
            return
        if detail:
            self.detail = str(detail)
        if percent is not None:
            self.percent = max(0, min(100, int(percent)))
        self._center._emit_updated(self)

    def succeed(self, detail: str = "") -> None:
        self._finish(JobState.SUCCESS, detail, percent=100)

    def warn(self, detail: str = "") -> None:
        self._finish(JobState.WARNING, detail)

    def fail(self, detail: str = "") -> None:
        self._finish(JobState.FAILED, detail)

    def finish(self, *, ok: bool, detail: str = "") -> None:
        """Convenience for call sites that already carry an ``ok`` flag."""
        if ok:
            self.succeed(detail)
        else:
            self.fail(detail)

    def _finish(self, state: JobState, detail: str, *, percent: int | None = None) -> None:
        if self.state.is_terminal:
            return
        self.state = state
        if detail:
            self.detail = str(detail)
        if percent is not None:
            self.percent = percent
        self.finished_at = time.monotonic()
        self._center._emit_finished(self)


class JobCenter(QObject):
    """Owns all :class:`Job` records and broadcasts lifecycle changes.

    Signals carry the affected :class:`Job` so listeners can render without
    re-querying.  ``jobsChanged`` is a coarse "something changed, re-render"
    pulse for views (such as the Activity panel) that prefer to rebuild wholesale.
    """

    jobStarted = pyqtSignal(object)
    jobUpdated = pyqtSignal(object)
    jobFinished = pyqtSignal(object)
    jobsChanged = pyqtSignal()

    def __init__(self, parent: QObject | None = None, *, history_limit: int = 60):
        super().__init__(parent)
        self._jobs: list[Job] = []
        self._next_id = 1
        self._history_limit = max(1, int(history_limit))

    def start(self, key: str, title: str) -> Job:
        """Register and return a new running job for tab *key*."""
        job = Job(self, self._next_id, key, title)
        self._next_id += 1
        self._jobs.append(job)
        self._trim_history()
        self.jobStarted.emit(job)
        self.jobsChanged.emit()
        return job

    def jobs(self) -> list[Job]:
        """All tracked jobs, oldest first."""
        return list(self._jobs)

    def active_jobs(self) -> list[Job]:
        return [job for job in self._jobs if job.is_running]

    def recent_jobs(self, limit: int | None = None) -> list[Job]:
        """Jobs ordered most-recent first, running jobs always first."""
        ordered = sorted(
            self._jobs,
            key=lambda job: (
                0 if job.is_running else 1,
                -(job.finished_at or job.started_at),
            ),
        )
        if limit is not None:
            return ordered[: max(0, int(limit))]
        return ordered

    def jobs_for(self, key: str) -> list[Job]:
        normalized = str(key or "")
        return [job for job in self._jobs if job.key == normalized]

    def active_count(self) -> int:
        return sum(1 for job in self._jobs if job.is_running)

    def acknowledge_key(self, key: str) -> bool:
        """Mark every finished job on *key* as seen.

        Returns ``True`` when at least one job changed, so the caller can decide
        whether to refresh badges.
        """
        normalized = str(key or "")
        changed = False
        for job in self._jobs:
            if job.key == normalized and job.is_finished and not job.acknowledged:
                job.acknowledged = True
                changed = True
        if changed:
            self.jobsChanged.emit()
        return changed

    def clear_finished(self) -> None:
        """Drop all finished jobs (running jobs are kept)."""
        kept = [job for job in self._jobs if job.is_running]
        if len(kept) != len(self._jobs):
            self._jobs = kept
            self.jobsChanged.emit()

    def _emit_updated(self, job: Job) -> None:
        self.jobUpdated.emit(job)
        self.jobsChanged.emit()

    def _emit_finished(self, job: Job) -> None:
        self.jobFinished.emit(job)
        self._trim_history()
        self.jobsChanged.emit()

    def _trim_history(self) -> None:
        finished = [job for job in self._jobs if job.is_finished]
        overflow = len(finished) - self._history_limit
        if overflow <= 0:
            return
        # Drop the oldest finished jobs while preserving running ones.
        to_drop = set(id(job) for job in finished[:overflow])
        self._jobs = [job for job in self._jobs if id(job) not in to_drop]


class JobReporter:
    """Mixin giving a window a single "current job" with guarded helpers.

    Windows set ``self._job_center`` (possibly ``None``) and ``self._job_key``
    during construction, then call :meth:`_begin_job` / :meth:`_report_progress`
    / :meth:`_finish_job` from their existing status call sites.  Every helper is
    a no-op when no center is attached, so standalone applets and tests are
    unaffected.
    """

    _job_center: JobCenter | None = None
    _job_key: str = ""

    def attach_job_center(self, center: JobCenter | None, key: str) -> None:
        self._job_center = center
        self._job_key = str(key or "")
        self._active_job = None

    def _begin_job(self, title: str) -> Job | None:
        center = getattr(self, "_job_center", None)
        if center is None:
            return None
        job = center.start(getattr(self, "_job_key", ""), title)
        self._active_job = job
        return job

    def _report_progress(self, detail: str = "", percent: int | float | None = None) -> None:
        job = getattr(self, "_active_job", None)
        if job is not None and job.is_running:
            job.progress(detail, percent)

    def _finish_job(self, *, ok: bool, detail: str = "") -> None:
        job = getattr(self, "_active_job", None)
        if job is not None:
            job.finish(ok=ok, detail=detail)
        self._active_job = None

    def _fail_job(self, detail: str = "") -> None:
        self._finish_job(ok=False, detail=detail)
