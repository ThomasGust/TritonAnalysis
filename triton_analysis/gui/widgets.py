"""Small shared input widgets tuned for fast, error-resistant data entry.

These are used on the competition data-entry tabs (eDNA counts, iceberg sheet
values) where an operator reads numbers off a physical spreadsheet under time
pressure.  Two behaviours matter there:

* Fields start **blank** instead of showing ``0`` so a real entry is never
  mistaken for a pre-filled value, and typing ``4`` into an empty box yields
  ``4`` rather than ``40``.
* The box **selects its contents on focus** so a click-and-type replaces the
  value, and it **ignores scroll-wheel** ticks so brushing the mouse never
  silently changes a recorded number.
"""

from __future__ import annotations

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QDoubleSpinBox, QSpinBox

# A single space renders as an empty-looking field while still being a valid
# special-value string (Qt treats "" inconsistently across styles).
_BLANK_TEXT = " "


class BlankZeroSpinBox(QSpinBox):
    """Integer spin box that shows blank at its minimum and replaces on focus."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setSpecialValueText(_BLANK_TEXT)

    def focusInEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().focusInEvent(event)
        # Defer: the base class repositions the cursor during focus-in, so
        # selecting now would be undone.
        QTimer.singleShot(0, self.selectAll)

    def wheelEvent(self, event) -> None:  # noqa: N802 (Qt override)
        event.ignore()


class BlankZeroDoubleSpinBox(QDoubleSpinBox):
    """Float spin box that shows blank at its minimum and replaces on focus."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setSpecialValueText(_BLANK_TEXT)

    def focusInEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().focusInEvent(event)
        QTimer.singleShot(0, self.selectAll)

    def wheelEvent(self, event) -> None:  # noqa: N802 (Qt override)
        event.ignore()
