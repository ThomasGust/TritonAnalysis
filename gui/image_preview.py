"""Shared Qt image preview helpers."""

from __future__ import annotations

import cv2
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget


def frame_to_pixmap(frame: np.ndarray | None) -> QPixmap:
    """Convert a BGR frame or ``None`` placeholder into a Qt pixmap."""
    if frame is None:
        return QPixmap()

    if frame.ndim == 2:
        rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    height, width, _ = rgb.shape
    image = QImage(rgb.data, width, height, rgb.strides[0], QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(image)


class ImagePreviewPanel(QWidget):
    """Labeled image preview panel."""

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self._pixmap = QPixmap()
        self._placeholder_text = "No image"

        self.title_label = QLabel(title)
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.image_label = QLabel(self._placeholder_text)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.image_label.setStyleSheet("background-color: #111; border: 1px solid #444;")
        self.image_label.setMinimumSize(220, 220)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.title_label)
        layout.addWidget(self.image_label, 1)

    def set_frame(self, frame: np.ndarray | None, *, placeholder_text: str = "No image") -> None:
        self._placeholder_text = placeholder_text
        self._pixmap = frame_to_pixmap(frame)
        self._update_pixmap()

    def clear(self, placeholder_text: str = "No image") -> None:
        self.set_frame(None, placeholder_text=placeholder_text)

    def _update_pixmap(self) -> None:
        if self._pixmap.isNull():
            self.image_label.clear()
            self.image_label.setText(self._placeholder_text)
            return

        self.image_label.setText("")
        self.image_label.setPixmap(
            self._pixmap.scaled(
                max(200, self.image_label.width() - 12),
                max(200, self.image_label.height() - 12),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_pixmap()
