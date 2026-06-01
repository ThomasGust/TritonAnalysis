"""File dialog helpers with image thumbnails for the active folder."""

from __future__ import annotations

from pathlib import Path

import json

from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtGui import QIcon, QImageReader, QPixmap
from PyQt6.QtWidgets import (
    QDialog,
    QFileDialog as QtFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QSizePolicy,
    QStyle,
    QToolButton,
    QVBoxLayout,
)


MAX_THUMBNAILS = 48
THUMBNAIL_SIZE = QSize(96, 96)
GRID_SIZE = QSize(128, 132)
IMAGE_SUFFIXES = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


class DirectoryThumbnailPreview(QFrame):
    """Side panel showing image thumbnails from one directory."""

    parentRequested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._directory: Path | None = None
        self.setMinimumWidth(280)
        self.setMaximumWidth(380)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)

        self.title_label = QLabel("Folder thumbnails")
        self.title_label.setStyleSheet("font-weight: 700;")
        self.parent_button = QToolButton()
        self.parent_button.setText("Up")
        self.parent_button.setToolTip("Move to the parent folder")
        self.parent_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowUp))
        self.parent_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.parent_button.clicked.connect(self.parentRequested.emit)
        self.path_label = QLabel("")
        self.path_label.setWordWrap(True)

        self.thumbnail_list = QListWidget()
        self.thumbnail_list.setViewMode(QListWidget.ViewMode.IconMode)
        self.thumbnail_list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.thumbnail_list.setMovement(QListWidget.Movement.Static)
        self.thumbnail_list.setSelectionMode(QListWidget.SelectionMode.NoSelection)
        self.thumbnail_list.setIconSize(THUMBNAIL_SIZE)
        self.thumbnail_list.setGridSize(GRID_SIZE)
        self.thumbnail_list.setSpacing(6)
        self.thumbnail_list.setUniformItemSizes(True)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 0, 0, 0)
        layout.setSpacing(6)
        header = QHBoxLayout()
        header.addWidget(self.title_label, 1)
        header.addWidget(self.parent_button)
        layout.addLayout(header)
        layout.addWidget(self.path_label)
        layout.addWidget(self.thumbnail_list, 1)
        layout.addWidget(self.status_label)

    def set_directory(self, value: str | Path) -> None:
        directory = Path(value).expanduser()
        if directory.is_file():
            directory = directory.parent
        if not directory.is_dir():
            return

        try:
            directory = directory.resolve()
        except OSError:
            pass
        if directory == self._directory:
            return

        self._directory = directory
        self.path_label.setText(directory.name or str(directory))
        self.thumbnail_list.clear()

        try:
            images, source_label = _thumbnail_images_for_directory(directory)
        except OSError as exc:
            self.status_label.setText(f"Could not read folder: {exc}")
            return

        for image_path in images[:MAX_THUMBNAILS]:
            item = QListWidgetItem(QIcon(_thumbnail_pixmap(image_path)), _display_name(directory, image_path))
            item.setToolTip(str(image_path))
            self.thumbnail_list.addItem(item)

        if not images:
            self.status_label.setText("No image files in this folder.")
        elif len(images) > MAX_THUMBNAILS:
            self.status_label.setText(f"Showing {MAX_THUMBNAILS} of {len(images)} {source_label}.")
        else:
            self.status_label.setText(f"{len(images)} {source_label}.")


def _thumbnail_images_for_directory(directory: Path) -> tuple[list[Path], str]:
    direct = _direct_images(directory)
    if direct:
        return direct, _image_count_label(len(direct))

    manifest_images = _manifest_images(directory)
    if manifest_images:
        return manifest_images, "session image" + ("" if len(manifest_images) == 1 else "s")

    child_images = _child_directory_images(directory)
    if child_images:
        return child_images, "subfolder image" + ("" if len(child_images) == 1 else "s")

    return [], "images"


def _direct_images(directory: Path) -> list[Path]:
    return [
        child
        for child in sorted(directory.iterdir(), key=lambda path: path.name.lower())
        if child.is_file() and child.suffix.lower() in IMAGE_SUFFIXES
    ]


def _manifest_images(directory: Path) -> list[Path]:
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        return []
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    images: list[Path] = []
    seen: set[Path] = set()
    for frame in manifest.get("frames") or []:
        if not isinstance(frame, dict):
            continue
        for key in ("left_path", "right_path"):
            value = frame.get(key)
            if not value:
                continue
            path = (directory / str(value)).resolve()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES and path not in seen:
                seen.add(path)
                images.append(path)
        for side in ("left", "right"):
            side_value = frame.get(side)
            value = side_value.get("path") if isinstance(side_value, dict) else None
            if not value:
                continue
            path = (directory / str(value)).resolve()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES and path not in seen:
                seen.add(path)
                images.append(path)
    return images


def _child_directory_images(directory: Path) -> list[Path]:
    images: list[Path] = []
    for child_dir in sorted((child for child in directory.iterdir() if child.is_dir()), key=lambda path: path.name.lower()):
        for image_path in _direct_images(child_dir):
            images.append(image_path)
    return images


def _image_count_label(count: int) -> str:
    return "image" if count == 1 else "images"


def _display_name(directory: Path, path: Path) -> str:
    try:
        return str(path.relative_to(directory))
    except ValueError:
        return path.name


def _thumbnail_pixmap(path: Path) -> QPixmap:
    reader = QImageReader(str(path))
    reader.setAutoTransform(True)
    size = reader.size()
    if size.isValid():
        size.scale(THUMBNAIL_SIZE, Qt.AspectRatioMode.KeepAspectRatio)
        reader.setScaledSize(size)
    image = reader.read()
    if image.isNull():
        return QPixmap()
    return QPixmap.fromImage(image)


class ThumbnailFileDialog(QtFileDialog):
    """QFileDialog with a side panel that previews the active directory."""

    def __init__(self, parent=None, caption: str = "", directory: str = "", filter: str = ""):
        super().__init__(parent, caption, directory, filter)
        self.setOption(QtFileDialog.Option.DontUseNativeDialog, True)
        self.setMinimumSize(980, 640)
        self.resize(1180, 760)
        self._thumbnail_preview = DirectoryThumbnailPreview(self)
        self._thumbnail_preview.parentRequested.connect(self._go_to_parent_directory)
        self._attach_thumbnail_preview()
        self.currentChanged.connect(self._preview_path)
        self.directoryEntered.connect(self._preview_path)
        self._preview_path(directory or self.directory().absolutePath())

    def _attach_thumbnail_preview(self) -> None:
        layout = self.layout()
        if isinstance(layout, QGridLayout):
            layout.addWidget(
                self._thumbnail_preview,
                0,
                layout.columnCount(),
                max(1, layout.rowCount()),
                1,
            )
        elif layout is not None:
            layout.addWidget(self._thumbnail_preview)

    def _preview_path(self, value: str) -> None:
        path = Path(value).expanduser() if value else Path(self.directory().absolutePath())
        if not path.exists():
            path = Path(self.directory().absolutePath())
        self._thumbnail_preview.set_directory(path)

    def _go_to_parent_directory(self) -> None:
        directory = Path(self.directory().absolutePath())
        try:
            current = directory.resolve()
        except OSError:
            current = directory
        parent = current.parent
        if parent == current:
            return
        self.setDirectory(str(parent))
        self._preview_path(str(parent))

    @staticmethod
    def getExistingDirectory(
        parent=None,
        caption: str = "",
        directory: str = "",
        options: QtFileDialog.Option = QtFileDialog.Option.ShowDirsOnly,
    ) -> str:
        dialog = ThumbnailFileDialog(parent, caption, directory)
        dialog.setFileMode(QtFileDialog.FileMode.Directory)
        dialog.setOptions(options | QtFileDialog.Option.DontUseNativeDialog)
        return _selected_path(dialog)

    @staticmethod
    def getOpenFileName(
        parent=None,
        caption: str = "",
        directory: str = "",
        filter: str = "",
        selectedFilter: str = "",
        options: QtFileDialog.Option = QtFileDialog.Option(0),
    ) -> tuple[str, str]:
        dialog = ThumbnailFileDialog(parent, caption, directory, filter)
        dialog.setAcceptMode(QtFileDialog.AcceptMode.AcceptOpen)
        dialog.setFileMode(QtFileDialog.FileMode.ExistingFile)
        dialog.setOptions(options | QtFileDialog.Option.DontUseNativeDialog)
        if selectedFilter:
            dialog.selectNameFilter(selectedFilter)
        return _selected_file_and_filter(dialog)

    @staticmethod
    def getOpenFileNames(
        parent=None,
        caption: str = "",
        directory: str = "",
        filter: str = "",
        selectedFilter: str = "",
        options: QtFileDialog.Option = QtFileDialog.Option(0),
    ) -> tuple[list[str], str]:
        dialog = ThumbnailFileDialog(parent, caption, directory, filter)
        dialog.setAcceptMode(QtFileDialog.AcceptMode.AcceptOpen)
        dialog.setFileMode(QtFileDialog.FileMode.ExistingFiles)
        dialog.setOptions(options | QtFileDialog.Option.DontUseNativeDialog)
        if selectedFilter:
            dialog.selectNameFilter(selectedFilter)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return [], dialog.selectedNameFilter()
        return dialog.selectedFiles(), dialog.selectedNameFilter()

    @staticmethod
    def getSaveFileName(
        parent=None,
        caption: str = "",
        directory: str = "",
        filter: str = "",
        selectedFilter: str = "",
        options: QtFileDialog.Option = QtFileDialog.Option(0),
    ) -> tuple[str, str]:
        dialog = ThumbnailFileDialog(parent, caption, directory, filter)
        dialog.setAcceptMode(QtFileDialog.AcceptMode.AcceptSave)
        dialog.setFileMode(QtFileDialog.FileMode.AnyFile)
        dialog.setOptions(options | QtFileDialog.Option.DontUseNativeDialog)
        if selectedFilter:
            dialog.selectNameFilter(selectedFilter)
        return _selected_file_and_filter(dialog)


def _selected_path(dialog: ThumbnailFileDialog) -> str:
    if dialog.exec() != QDialog.DialogCode.Accepted:
        return ""
    selected = dialog.selectedFiles()
    return selected[0] if selected else ""


def _selected_file_and_filter(dialog: ThumbnailFileDialog) -> tuple[str, str]:
    if dialog.exec() != QDialog.DialogCode.Accepted:
        return "", dialog.selectedNameFilter()
    selected = dialog.selectedFiles()
    return (selected[0] if selected else ""), dialog.selectedNameFilter()
