"""Widgets that render :mod:`job_center` activity in the unified window.

Three surfaces share one :class:`~triton_analysis.gui.job_center.JobCenter`:

* :class:`JobStatusTabBar` paints a small status dot on each tab so the whole
  tab strip is glanceable.
* :class:`JobToast` / :class:`ToastStack` pop a transient banner when a job
  finishes on a tab the operator is not currently watching.
* :class:`ActivityPanel` is a popup that lists every running and recent job.

The widgets are intentionally decoupled from :class:`JobCenter`: the unified
window owns the center and drives these views, which keeps the render logic
testable and lets standalone applets skip the whole layer.
"""

from __future__ import annotations

from PyQt6.QtCore import QPoint, QRect, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QPainter
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QTabBar,
    QVBoxLayout,
    QWidget,
)

from triton_analysis.gui.job_center import Job, JobState

# tone -> dot color, shared with the stylesheet's ``tone`` palette.
_TONE_COLORS = {
    "running": QColor(0x6A, 0xA6, 0xFF),
    "ok": QColor(0x9B, 0xE7, 0xB0),
    "warn": QColor(0xF4, 0xCF, 0x7A),
    "alert": QColor(0xFF, 0xAA, 0xA5),
}

_STATE_GLYPH = {
    JobState.RUNNING: "●",   # ●
    JobState.SUCCESS: "✓",   # ✓
    JobState.WARNING: "⚠",   # ⚠
    JobState.FAILED: "✗",    # ✗
}


def tone_color(tone: str) -> QColor:
    return _TONE_COLORS.get(tone, QColor(0x9A, 0x9A, 0xB0))


def state_glyph(state: JobState) -> str:
    return _STATE_GLYPH.get(state, "●")


class JobStatusTabBar(QTabBar):
    """Tab bar that paints a per-tab status dot in the top-right corner.

    The unified window owns the index -> tone mapping and pushes it here; the
    bar only renders.  Storing the mapping (rather than baking it into the paint
    call) keeps it assertable from tests without rendering.
    """

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._tone_by_index: dict[int, str] = {}

    def set_index_tones(self, tones: dict[int, str]) -> None:
        cleaned = {int(i): str(tone) for i, tone in tones.items() if tone}
        if cleaned != self._tone_by_index:
            self._tone_by_index = cleaned
            self.update()

    def index_tone(self, index: int) -> str:
        return self._tone_by_index.get(int(index), "")

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().paintEvent(event)
        if not self._tone_by_index:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        radius = 4
        for index, tone in self._tone_by_index.items():
            if index < 0 or index >= self.count():
                continue
            if not self.isTabVisible(index):
                continue
            rect = self.tabRect(index)
            if rect.isNull():
                continue
            center = QPoint(rect.right() - radius - 4, rect.top() + radius + 5)
            color = tone_color(tone)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(color)
            painter.drawEllipse(center, radius, radius)
        painter.end()


class JobToast(QFrame):
    """A single transient finish banner; clicking it focuses the owning tab."""

    clicked = pyqtSignal(str)  # emits the job's tab key
    dismissed = pyqtSignal(object)  # emits self

    def __init__(self, job: Job, *, timeout_ms: int = 7000, parent: QWidget | None = None):
        super().__init__(parent)
        self._key = job.key
        self.setObjectName("jobToast")
        self.setProperty("tone", job.state.value)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.setStyleSheet(
            """
            QFrame#jobToast {
                background: #23232c;
                border: 1px solid #3a3a48;
                border-left: 4px solid #6aa6ff;
                border-radius: 8px;
            }
            QFrame#jobToast[tone="ok"] { border-left-color: #9be7b0; }
            QFrame#jobToast[tone="warn"] { border-left-color: #f4cf7a; }
            QFrame#jobToast[tone="alert"] { border-left-color: #ffaaa5; }
            QLabel#jobToastTitle { font-weight: 700; color: #ffffff; }
            QLabel#jobToastDetail { color: #c7cad7; }
            QPushButton#jobToastClose {
                border: none; background: transparent; color: #9a9ab0;
                font-weight: 700; padding: 0 4px;
            }
            QPushButton#jobToastClose:hover { color: #ffffff; }
            """
        )

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 8, 6, 8)
        layout.setSpacing(8)

        text_col = QVBoxLayout()
        text_col.setContentsMargins(0, 0, 0, 0)
        text_col.setSpacing(1)
        glyph = state_glyph(job.state)
        self.title_label = QLabel(f"{glyph}  {job.title}")
        self.title_label.setObjectName("jobToastTitle")
        detail = job.detail or self._default_detail(job.state)
        self.detail_label = QLabel(detail)
        self.detail_label.setObjectName("jobToastDetail")
        self.detail_label.setWordWrap(True)
        text_col.addWidget(self.title_label)
        text_col.addWidget(self.detail_label)
        layout.addLayout(text_col, 1)

        close_btn = QPushButton("✕")
        close_btn.setObjectName("jobToastClose")
        close_btn.setFixedWidth(20)
        close_btn.setCursor(Qt.CursorShape.ArrowCursor)
        close_btn.clicked.connect(self._dismiss)
        layout.addWidget(close_btn, 0, Qt.AlignmentFlag.AlignTop)

        self.setMaximumWidth(360)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._dismiss)
        if timeout_ms > 0:
            self._timer.start(int(timeout_ms))

    @staticmethod
    def _default_detail(state: JobState) -> str:
        if state is JobState.SUCCESS:
            return "Finished. Click to open."
        if state is JobState.FAILED:
            return "Failed. Click to open."
        if state is JobState.WARNING:
            return "Finished with warnings. Click to open."
        return "Click to open."

    @property
    def key(self) -> str:
        return self._key

    def mousePressEvent(self, event) -> None:  # noqa: N802 (Qt override)
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self._key)
            self._dismiss()
            return
        super().mousePressEvent(event)

    def _dismiss(self) -> None:
        self._timer.stop()
        self.dismissed.emit(self)
        self.hide()
        self.deleteLater()


class ToastStack(QWidget):
    """Bottom-right overlay that stacks active :class:`JobToast` banners.

    Reparented onto the main window; it repositions itself against the parent's
    bottom-right corner whenever a toast is shown or the parent resizes.
    """

    toastClicked = pyqtSignal(str)

    def __init__(self, parent: QWidget, *, max_visible: int = 4):
        super().__init__(parent)
        self._max_visible = max_visible
        self._toasts: list[JobToast] = []
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addStretch(1)
        self._layout = layout
        self.setFixedWidth(372)
        self.hide()

    def show_job(self, job: Job, *, timeout_ms: int = 7000) -> JobToast:
        toast = JobToast(job, timeout_ms=timeout_ms, parent=self)
        toast.clicked.connect(self.toastClicked)
        toast.dismissed.connect(self._remove_toast)
        self._toasts.append(toast)
        self._layout.addWidget(toast)
        while len(self._toasts) > self._max_visible:
            oldest = self._toasts[0]
            oldest._dismiss()
        self.show()
        self.raise_()
        self.reposition()
        return toast

    def _remove_toast(self, toast: JobToast) -> None:
        if toast in self._toasts:
            self._toasts.remove(toast)
        if not self._toasts:
            self.hide()
        else:
            self.reposition()

    def reposition(self) -> None:
        parent = self.parentWidget()
        if parent is None:
            return
        self.adjustSize()
        margin = 16
        x = parent.width() - self.width() - margin
        y = parent.height() - self.height() - margin
        self.move(max(margin, x), max(margin, y))


class ActivityPanel(QFrame):
    """Popup listing running and recently finished jobs."""

    jobActivated = pyqtSignal(str)  # tab key
    clearRequested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent, Qt.WindowType.Popup)
        self.setObjectName("activityPanel")
        self.setStyleSheet(
            """
            QFrame#activityPanel {
                background: #1d1d24;
                border: 1px solid #3a3a48;
                border-radius: 10px;
            }
            QLabel#activityHeader { font-weight: 700; color: #ebebeb; }
            QLabel#activityEmpty { color: #9a9ab0; padding: 12px 4px; }
            QPushButton#activityClear {
                border: none; background: transparent; color: #9aa6c8;
            }
            QPushButton#activityClear:hover { color: #ffffff; }
            """
        )
        self.setMinimumWidth(340)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 10, 12, 12)
        outer.setSpacing(8)

        header_row = QHBoxLayout()
        header = QLabel("Activity")
        header.setObjectName("activityHeader")
        header_row.addWidget(header, 1)
        self.clear_btn = QPushButton("Clear finished")
        self.clear_btn.setObjectName("activityClear")
        self.clear_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.clear_btn.clicked.connect(self.clearRequested)
        header_row.addWidget(self.clear_btn, 0)
        outer.addLayout(header_row)

        self._rows_host = QWidget()
        self._rows_layout = QVBoxLayout(self._rows_host)
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.setSpacing(6)
        outer.addWidget(self._rows_host)

        self._empty_label = QLabel("No analysis jobs yet.")
        self._empty_label.setObjectName("activityEmpty")
        outer.addWidget(self._empty_label)

    def set_jobs(self, jobs: list[Job]) -> None:
        while self._rows_layout.count():
            item = self._rows_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._empty_label.setVisible(not jobs)
        self.clear_btn.setVisible(any(job.is_finished for job in jobs))
        for job in jobs:
            self._rows_layout.addWidget(self._build_row(job))
        self.adjustSize()

    def _build_row(self, job: Job) -> QWidget:
        row = QFrame()
        row.setCursor(Qt.CursorShape.PointingHandCursor)
        layout = QHBoxLayout(row)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(8)

        dot = QLabel(state_glyph(job.state))
        dot.setStyleSheet(f"color: {tone_color(job.state.value).name()}; font-weight: 700;")
        layout.addWidget(dot, 0)

        text_col = QVBoxLayout()
        text_col.setContentsMargins(0, 0, 0, 0)
        text_col.setSpacing(0)
        title = QLabel(job.title)
        title.setStyleSheet("color: #ebebeb;")
        detail_bits = []
        if job.is_running:
            if job.percent is not None:
                detail_bits.append(f"{job.percent}%")
            detail_bits.append(self._format_elapsed(job.elapsed()))
        else:
            detail_bits.append(self._format_elapsed(job.elapsed()))
        if job.detail:
            detail_bits.append(job.detail)
        detail = QLabel("  ·  ".join(detail_bits))
        detail.setStyleSheet("color: #9a9ab0;")
        detail.setWordWrap(True)
        text_col.addWidget(title)
        text_col.addWidget(detail)
        layout.addLayout(text_col, 1)

        # Capture the key so the row activates the right tab when clicked.
        key = job.key
        row.mousePressEvent = lambda _event, k=key: self.jobActivated.emit(k)  # type: ignore[assignment]
        return row

    @staticmethod
    def _format_elapsed(seconds: float) -> str:
        total = int(seconds)
        minutes, secs = divmod(total, 60)
        if minutes:
            return f"{minutes}:{secs:02d}"
        return f"{secs}s"

    def popup_at(self, global_pos: QPoint) -> None:
        self.adjustSize()
        self.move(global_pos)
        self.show()
        self.raise_()
