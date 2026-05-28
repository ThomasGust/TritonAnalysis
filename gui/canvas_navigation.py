"""Shared zoom/pan helpers for image measurement canvases."""

from __future__ import annotations

from PyQt6.QtCore import QRect, QRectF

LEFT_DRAG_PAN_THRESHOLD_PX = 6.0


def clamp_pan_to_edge_margin(pan, contents: QRect | QRectF, target: QRectF) -> None:
    """Clamp pan while allowing zoomed image edges to move inward for clicking."""
    if target.width() <= contents.width():
        pan[0] = 0.0
    else:
        margin = _edge_margin(float(contents.width()))
        min_pan = float(contents.right()) - margin - float(target.right())
        max_pan = float(contents.left()) + margin - float(target.left())
        pan[0] = _clamp(float(pan[0]), min_pan, max_pan)

    if target.height() <= contents.height():
        pan[1] = 0.0
    else:
        margin = _edge_margin(float(contents.height()))
        min_pan = float(contents.bottom()) - margin - float(target.bottom())
        max_pan = float(contents.top()) + margin - float(target.top())
        pan[1] = _clamp(float(pan[1]), min_pan, max_pan)


def _edge_margin(view_size: float) -> float:
    return min(view_size * 0.45, max(48.0, view_size * 0.35))


def moved_past_pan_threshold(start: tuple[float, float] | None, x: float, y: float) -> bool:
    if start is None:
        return False
    dx = float(x) - float(start[0])
    dy = float(y) - float(start[1])
    return (dx * dx + dy * dy) >= LEFT_DRAG_PAN_THRESHOLD_PX * LEFT_DRAG_PAN_THRESHOLD_PX


def _clamp(value: float, lower: float, upper: float) -> float:
    if lower > upper:
        return (lower + upper) * 0.5
    return max(lower, min(upper, value))
