"""Unit tests for the cross-tab job tracking model."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication

from triton_analysis.gui.job_center import Job, JobCenter, JobState, state_priority


def _app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_start_registers_running_job_and_emits():
    _app()
    center = JobCenter()
    started: list[Job] = []
    changed: list[int] = []
    center.jobStarted.connect(started.append)
    center.jobsChanged.connect(lambda: changed.append(1))

    job = center.start("crab-counter", "Crab Counter")

    assert job.is_running
    assert job.state is JobState.RUNNING
    assert started == [job]
    assert changed == [1]
    assert center.active_count() == 1
    assert center.jobs_for("crab-counter") == [job]


def test_progress_updates_and_is_ignored_after_finish():
    _app()
    center = JobCenter()
    updates: list[Job] = []
    finished: list[Job] = []
    center.jobUpdated.connect(updates.append)
    center.jobFinished.connect(finished.append)

    job = center.start("coral-reconstruction", "Reconstruction")
    job.progress("Aligning images", 40)

    assert job.detail == "Aligning images"
    assert job.percent == 40
    assert updates == [job]

    job.succeed("done")
    assert job.state is JobState.SUCCESS
    assert job.percent == 100
    assert finished == [job]

    # Progress after a terminal state must not resurrect or mutate the job.
    job.progress("late", 10)
    assert job.state is JobState.SUCCESS
    assert job.detail == "done"
    assert updates == [job]  # no second update emitted


def test_percent_is_clamped():
    _app()
    center = JobCenter()
    job = center.start("stereo-calibration", "Calibration")
    job.progress("", 540)
    assert job.percent == 100
    job.progress("", -5)
    assert job.percent == 0


def test_fail_marks_failed_state():
    _app()
    center = JobCenter()
    job = center.start("crab-counter", "Crab Counter")
    job.fail("OpenAI timeout")
    assert job.state is JobState.FAILED
    assert job.is_finished
    assert job.detail == "OpenAI timeout"


def test_finish_helper_routes_on_ok_flag():
    _app()
    center = JobCenter()
    ok_job = center.start("a", "A")
    ok_job.finish(ok=True, detail="great")
    fail_job = center.start("b", "B")
    fail_job.finish(ok=False, detail="bad")
    assert ok_job.state is JobState.SUCCESS
    assert fail_job.state is JobState.FAILED


def test_recent_jobs_puts_running_first():
    _app()
    center = JobCenter()
    first = center.start("a", "A")
    first.succeed()
    second = center.start("b", "B")  # still running
    third = center.start("c", "C")
    third.fail()

    recent = center.recent_jobs()
    # The single running job is always surfaced first.
    assert recent[0] is second
    assert set(recent) == {first, second, third}
    assert center.recent_jobs(limit=1) == [second]


def test_acknowledge_key_only_affects_finished_jobs():
    _app()
    center = JobCenter()
    running = center.start("crab-counter", "Crab Counter")
    done = center.start("crab-counter", "Crab Counter 2")
    done.succeed("7")

    assert center.acknowledge_key("crab-counter") is True
    assert done.acknowledged is True
    assert running.acknowledged is False
    # Nothing new to acknowledge the second time.
    assert center.acknowledge_key("crab-counter") is False


def test_clear_finished_keeps_running_jobs():
    _app()
    center = JobCenter()
    running = center.start("a", "A")
    done = center.start("b", "B")
    done.succeed()

    center.clear_finished()

    assert center.jobs() == [running]


def test_history_is_trimmed_but_running_jobs_survive():
    _app()
    center = JobCenter(history_limit=3)
    running = center.start("live", "Live")
    finished = []
    for index in range(5):
        job = center.start("batch", f"Job {index}")
        job.succeed()
        finished.append(job)

    jobs = center.jobs()
    assert running in jobs
    # Only the newest 3 finished jobs are retained.
    retained_finished = [job for job in jobs if job.is_finished]
    assert len(retained_finished) == 3
    assert retained_finished == finished[-3:]


def test_state_priority_orders_running_above_terminal():
    assert state_priority(JobState.RUNNING) > state_priority(JobState.FAILED)
    assert state_priority(JobState.FAILED) > state_priority(JobState.WARNING)
    assert state_priority(JobState.WARNING) > state_priority(JobState.SUCCESS)
