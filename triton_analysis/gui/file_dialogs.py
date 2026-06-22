"""File dialog helpers with image thumbnails for the active folder."""

from __future__ import annotations

import json
import os
from fnmatch import fnmatch
from pathlib import Path

from PyQt6.QtCore import (
    QDir,
    QItemSelectionModel,
    QModelIndex,
    QObject,
    QRunnable,
    QSettings,
    QSize,
    QThreadPool,
    QTimer,
    Qt,
    pyqtSignal,
)
from PyQt6.QtGui import QColor, QIcon, QImage, QImageReader, QPixmap, QStandardItem, QStandardItemModel
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog as QtFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QSizePolicy,
    QStyle,
    QToolButton,
    QTreeView,
    QVBoxLayout,
)


MAX_THUMBNAILS = 12
MAX_CHILD_PREVIEW_DIRS = 12
MAX_PREVIEW_DIRECTORY_ENTRIES = 300
DIRECTORY_LOAD_BATCH_SIZE = 40
THUMBNAIL_SIZE = QSize(96, 96)
GRID_SIZE = QSize(128, 132)
IMAGE_SUFFIXES = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
PATH_ROLE = int(Qt.ItemDataRole.UserRole) + 1
IS_DIR_ROLE = int(Qt.ItemDataRole.UserRole) + 2
RECENT_DIRS_LIMIT = 8


class _ThumbnailSignals(QObject):
    """Carries a decoded thumbnail back to the GUI thread.

    Kept unparented and shared across decode tasks so it outlives the preview
    widget if a worker is still running when the dialog closes; Qt drops the
    queued ``done`` delivery once the connected preview is destroyed.
    """

    done = pyqtSignal(int, str, object)  # generation, path, QImage


class _ThumbnailDecodeTask(QRunnable):
    """Decode one scaled thumbnail off the GUI thread.

    A *generation* token lets the preview discard work for folders the operator
    has already navigated away from, so rapid browsing never piles up stale
    decodes on the thread pool.
    """

    def __init__(self, generation: int, path: Path, size: QSize, signals: _ThumbnailSignals, is_current):
        super().__init__()
        self._generation = generation
        self._path = path
        self._size = size
        self._signals = signals
        self._is_current = is_current
        self.setAutoDelete(True)

    def run(self) -> None:
        if not self._is_current(self._generation):
            return
        image = _thumbnail_image(self._path, self._size)
        if image.isNull() or not self._is_current(self._generation):
            return
        self._signals.done.emit(self._generation, str(self._path), image)


class DirectoryThumbnailPreview(QFrame):
    """Side panel showing image thumbnails from one directory."""

    parentRequested = pyqtSignal()
    openRequested = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._directory: Path | None = None
        self._image_cache: dict[Path, tuple[list[Path], str]] = {}
        self._icon_cache: dict[Path, QIcon] = {}
        self._thumb_items: dict[str, QListWidgetItem] = {}
        self._thumb_generation = 0
        # A per-widget pool (not the global one) so its destructor blocks until
        # in-flight decodes finish, preventing a worker from emitting into a
        # half-destroyed preview when the dialog closes mid-load.  It is created
        # before the signals object so that, during teardown, child destruction
        # drains the pool's tasks before the signals object they emit on is gone.
        self._thumb_pool = QThreadPool(self)
        self._thumb_pool.setMaxThreadCount(4)
        # Parented to self so it is destroyed deterministically with the preview
        # (while the QApplication is still alive) rather than lingering until the
        # interpreter-shutdown GC, which races QApplication teardown.
        self._thumb_signals = _ThumbnailSignals(self)
        self._thumb_signals.done.connect(self._on_thumbnail_decoded)
        self._placeholder_icon = _placeholder_thumbnail_icon()
        self.setMinimumWidth(280)
        self.setMaximumWidth(380)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)

        self.title_label = QLabel("Folder thumbnails")
        self.title_label.setStyleSheet("font-weight: 700;")
        self.parent_button = QToolButton()
        self.parent_button.setToolTip("Move to the parent folder")
        self.parent_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowUp))
        self.parent_button.clicked.connect(self.parentRequested.emit)
        self.open_button = QToolButton()
        self.open_button.setToolTip("Enter the previewed folder")
        self.open_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon))
        self.open_button.clicked.connect(self._request_open_directory)
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
        header.addWidget(self.open_button)
        header.addWidget(self.parent_button)
        layout.addLayout(header)
        layout.addWidget(self.path_label)
        layout.addWidget(self.thumbnail_list, 1)
        layout.addWidget(self.status_label)

    def _request_open_directory(self) -> None:
        if self._directory is not None:
            self.openRequested.emit(self._directory)

    def _begin_new_generation(self) -> int:
        """Invalidate in-flight decodes and reset per-folder thumbnail state."""
        self._thumb_generation += 1
        self._thumb_items = {}
        self.thumbnail_list.clear()
        return self._thumb_generation

    def show_pending_directory(self, value: str | Path) -> None:
        self._begin_new_generation()
        directory = Path(value).expanduser()
        if directory.is_file():
            directory = directory.parent
        try:
            directory = directory.resolve()
        except OSError:
            pass

        self._directory = None
        self.path_label.setText(directory.name or str(directory))
        self.status_label.setText("Loading thumbnails...")

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

        generation = self._begin_new_generation()
        self._directory = directory
        self.path_label.setText(directory.name or str(directory))

        cached = self._image_cache.get(directory)
        if cached is None:
            try:
                images, source_label = _thumbnail_images_for_directory(directory)
            except OSError as exc:
                self.status_label.setText(f"Could not read folder: {exc}")
                return
            self._image_cache[directory] = (images, source_label)
        else:
            images, source_label = cached

        if not images:
            self.status_label.setText("No image files in this folder.")
            return

        # Lay out every tile immediately with a placeholder (or cached icon) so
        # the grid appears instantly; real thumbnails stream in from the pool.
        for image_path in images[:MAX_THUMBNAILS]:
            cached_icon = self._icon_cache.get(image_path)
            item = QListWidgetItem(cached_icon or self._placeholder_icon, _display_name(directory, image_path))
            item.setToolTip(str(image_path))
            self.thumbnail_list.addItem(item)
            self._thumb_items[str(image_path)] = item
            if cached_icon is None:
                self._dispatch_thumbnail_decode(generation, image_path)

        total = len(images)
        if total > MAX_THUMBNAILS:
            self.status_label.setText(f"Showing first {MAX_THUMBNAILS} {source_label}.")
        else:
            self.status_label.setText(f"{total} {source_label}.")

    def _dispatch_thumbnail_decode(self, generation: int, image_path: Path) -> None:
        task = _ThumbnailDecodeTask(
            generation,
            image_path,
            THUMBNAIL_SIZE,
            self._thumb_signals,
            self._is_thumbnail_generation_current,
        )
        self._thumb_pool.start(task)

    def _is_thumbnail_generation_current(self, generation: int) -> bool:
        return generation == self._thumb_generation

    def _on_thumbnail_decoded(self, generation: int, path_str: str, image: object) -> None:
        if generation != self._thumb_generation:
            return
        if not isinstance(image, QImage) or image.isNull():
            return
        icon = QIcon(QPixmap.fromImage(image))
        self._icon_cache[Path(path_str)] = icon
        item = self._thumb_items.get(path_str)
        if item is not None:
            item.setIcon(icon)


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


def _direct_images(directory: Path, limit: int = MAX_THUMBNAILS + 1) -> list[Path]:
    images: list[Path] = []
    for index, child in enumerate(directory.iterdir(), start=1):
        if index > MAX_PREVIEW_DIRECTORY_ENTRIES:
            break
        if child.is_file() and child.suffix.lower() in IMAGE_SUFFIXES:
            images.append(child)
            if len(images) >= limit:
                break
    return sorted(images, key=lambda path: path.name.lower())


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
    for frame in (manifest.get("frames") or [])[: MAX_THUMBNAILS + 1]:
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
                if len(images) > MAX_THUMBNAILS:
                    return images
        for side in ("left", "right"):
            side_value = frame.get(side)
            value = side_value.get("path") if isinstance(side_value, dict) else None
            if not value:
                continue
            path = (directory / str(value)).resolve()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES and path not in seen:
                seen.add(path)
                images.append(path)
                if len(images) > MAX_THUMBNAILS:
                    return images
    return images


def _child_directory_images(directory: Path) -> list[Path]:
    images: list[Path] = []
    for child_dir in _preview_child_directories(directory):
        for image_path in _direct_images(child_dir, limit=MAX_THUMBNAILS + 1 - len(images)):
            images.append(image_path)
            if len(images) > MAX_THUMBNAILS:
                return images
    return images


def _preview_child_directories(directory: Path) -> list[Path]:
    children: list[Path] = []
    seen: set[Path] = set()
    for name in ("left", "right", "images", "frames"):
        child = directory / name
        if child.is_dir():
            children.append(child)
            seen.add(child)

    for index, child in enumerate(directory.iterdir(), start=1):
        if len(children) >= MAX_CHILD_PREVIEW_DIRS:
            break
        if index > MAX_PREVIEW_DIRECTORY_ENTRIES:
            break
        if child in seen or not child.is_dir():
            continue
        children.append(child)
        seen.add(child)
    return children


def _image_count_label(count: int) -> str:
    return "image" if count == 1 else "images"


def _display_name(directory: Path, path: Path) -> str:
    try:
        return str(path.relative_to(directory))
    except ValueError:
        return path.name


def _thumbnail_image(path: Path, size: QSize = THUMBNAIL_SIZE) -> QImage:
    """Decode an image scaled to fit *size*. Safe to call off the GUI thread."""
    reader = QImageReader(str(path))
    reader.setAutoTransform(True)
    source_size = reader.size()
    if source_size.isValid():
        source_size.scale(size, Qt.AspectRatioMode.KeepAspectRatio)
        reader.setScaledSize(source_size)
    return reader.read()


def _thumbnail_pixmap(path: Path) -> QPixmap:
    image = _thumbnail_image(path)
    if image.isNull():
        return QPixmap()
    return QPixmap.fromImage(image)


def _placeholder_thumbnail_icon() -> QIcon:
    pixmap = QPixmap(THUMBNAIL_SIZE)
    pixmap.fill(QColor(0x24, 0x24, 0x2C))
    return QIcon(pixmap)


def _resolve_path(path: Path) -> Path:
    try:
        return path.expanduser().resolve()
    except OSError:
        return path.expanduser()


def _initial_directory_and_file(value: str | Path) -> tuple[Path, str]:
    text = str(value or "").strip()
    if not text:
        return Path.cwd(), ""

    path = Path(text).expanduser()
    if path.exists():
        if path.is_file():
            return path.parent, path.name
        return path, ""
    if path.suffix:
        return path.parent if str(path.parent) else Path.cwd(), path.name
    return path, ""


def _filter_patterns(filter_label: str) -> list[str]:
    label = str(filter_label or "").strip()
    start = label.rfind("(")
    end = label.rfind(")")
    if start < 0 or end <= start:
        return ["*"]
    patterns = [part.strip() for part in label[start + 1 : end].split() if part.strip()]
    return patterns or ["*"]


def _filter_entries(filter_text: str) -> list[tuple[str, list[str]]]:
    entries: list[tuple[str, list[str]]] = []
    for raw_entry in str(filter_text or "").split(";;"):
        label = raw_entry.strip()
        if label:
            entries.append((label, _filter_patterns(label)))
    return entries or [("All files (*)", ["*"])]


def _default_suffix_for_filter(filter_label: str) -> str:
    for pattern in _filter_patterns(filter_label):
        if not pattern.startswith("*."):
            continue
        suffix = pattern[1:]
        if any(char in suffix for char in "*?[]"):
            continue
        return suffix
    return ""


class ThumbnailFileDialog(QDialog):
    """Fast Qt file dialog with deterministic folder preview behavior."""

    FileMode = QtFileDialog.FileMode
    AcceptMode = QtFileDialog.AcceptMode
    Option = QtFileDialog.Option
    DialogLabel = QtFileDialog.DialogLabel

    def __init__(self, parent=None, caption: str = "", directory: str = "", filter: str = ""):
        super().__init__(parent)
        self.setWindowTitle(caption or "Select file")
        self.setMinimumSize(980, 640)
        self.resize(1180, 760)

        start_dir, start_file = _initial_directory_and_file(directory)
        self._current_dir = _resolve_path(start_dir)
        self._file_mode = QtFileDialog.FileMode.ExistingFile
        self._accept_mode = QtFileDialog.AcceptMode.AcceptOpen
        self._options = QtFileDialog.Option(0)
        self._selected_files: list[str] = []
        self._selected_directory: Path | None = None
        self._name_filters = _filter_entries(filter)
        self._active_patterns = self._name_filters[0][1]
        self._selected_name_filter = self._name_filters[0][0]
        self._pending_preview_path: Path | None = None
        self._displayed_items: dict[str, QStandardItem] = {}
        # Sorted (folders-first) scan of the current folder, plus the queue the
        # batch timer renders from after the quick filter is applied.
        self._all_entries: list[tuple[str, Path, bool]] = []
        self._render_queue: list[tuple[str, Path, bool]] = []
        self._render_index = 0
        self._filter_text = ""
        self._directory_initialized = False
        self._dialog_settings = QSettings("TritonAnalysis", "FileDialog")

        # Decoding now runs off-thread, so the preview can react quickly without
        # stalling rapid keyboard/mouse browsing.
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(120)
        self._preview_timer.timeout.connect(self._apply_pending_preview)

        self._entry_model = QStandardItemModel(self)
        self._folder_icon = self.style().standardIcon(QStyle.StandardPixmap.SP_DirIcon)
        self._file_icon = self.style().standardIcon(QStyle.StandardPixmap.SP_FileIcon)
        self._directory_load_timer = QTimer(self)
        self._directory_load_timer.setInterval(1)
        self._directory_load_timer.timeout.connect(self._load_next_directory_batch)

        self._thumbnail_preview = DirectoryThumbnailPreview(self)
        self._thumbnail_preview.parentRequested.connect(self._go_to_parent_directory)
        self._thumbnail_preview.openRequested.connect(self._open_directory)

        self._build_ui()
        self._apply_file_mode()
        self._populate_filter_combo()
        self.setDirectory(str(self._current_dir))
        if start_file:
            self.selectFile(start_file)

    def _build_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(8)

        nav_layout = QHBoxLayout()
        nav_layout.setSpacing(6)
        self.up_button = QToolButton()
        self.up_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowUp))
        self.up_button.setToolTip("Move to the parent folder")
        self.up_button.clicked.connect(self._go_to_parent_directory)
        self.refresh_button = QToolButton()
        self.refresh_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_BrowserReload))
        self.refresh_button.setToolTip("Refresh this folder")
        self.refresh_button.clicked.connect(self._refresh_directory)
        self.path_edit = QLineEdit()
        self.path_edit.returnPressed.connect(self._open_path_from_edit)
        nav_layout.addWidget(self.up_button)
        nav_layout.addWidget(self.refresh_button)
        nav_layout.addWidget(self.path_edit, 1)
        main_layout.addLayout(nav_layout)

        body_layout = QHBoxLayout()
        body_layout.setSpacing(10)

        self.places_list = self._build_places_sidebar()
        body_layout.addWidget(self.places_list)

        center_column = QVBoxLayout()
        center_column.setContentsMargins(0, 0, 0, 0)
        center_column.setSpacing(6)
        self.quick_filter_edit = QLineEdit()
        self.quick_filter_edit.setClearButtonEnabled(True)
        self.quick_filter_edit.setPlaceholderText("Filter this folder...")
        self.quick_filter_edit.textChanged.connect(self._on_quick_filter_changed)
        center_column.addWidget(self.quick_filter_edit)

        self.file_view = QTreeView()
        self._file_views = [self.file_view]
        self.file_view.setModel(self._entry_model)
        self.file_view.setRootIsDecorated(False)
        self.file_view.setItemsExpandable(False)
        self.file_view.setAlternatingRowColors(True)
        self.file_view.setSortingEnabled(False)
        self.file_view.setUniformRowHeights(True)
        self.file_view.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.file_view.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.file_view.header().hide()
        self.file_view.clicked.connect(self._select_index)
        self.file_view.doubleClicked.connect(self._open_or_accept_index)
        self.file_view.activated.connect(self._open_or_accept_index)
        center_column.addWidget(self.file_view, 1)
        body_layout.addLayout(center_column, 1)

        body_layout.addWidget(self._thumbnail_preview)
        main_layout.addLayout(body_layout, 1)

        form_layout = QGridLayout()
        form_layout.setHorizontalSpacing(8)
        form_layout.setVerticalSpacing(6)
        self.file_name_label = QLabel("File name")
        self.file_name_edit = QLineEdit()
        self.file_name_edit.returnPressed.connect(self.accept)
        self.filter_label = QLabel("File type")
        self.filter_combo = QComboBox()
        self.filter_combo.currentIndexChanged.connect(self._filter_changed)
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        form_layout.addWidget(self.file_name_label, 0, 0)
        form_layout.addWidget(self.file_name_edit, 0, 1)
        form_layout.addWidget(self.filter_label, 1, 0)
        form_layout.addWidget(self.filter_combo, 1, 1)
        form_layout.addWidget(self.status_label, 2, 0, 1, 2)
        main_layout.addLayout(form_layout)

        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        main_layout.addWidget(self.button_box)

        selection_model = self.file_view.selectionModel()
        if selection_model is not None:
            selection_model.selectionChanged.connect(lambda _selected, _deselected: self._selection_changed())
            selection_model.currentChanged.connect(lambda current, _previous: self._select_index(current))

    def _populate_filter_combo(self) -> None:
        self.filter_combo.blockSignals(True)
        self.filter_combo.clear()
        for label, _patterns in self._name_filters:
            self.filter_combo.addItem(label)
        selected_index = next(
            (i for i, item in enumerate(self._name_filters) if item[0] == self._selected_name_filter),
            0,
        )
        self.filter_combo.setCurrentIndex(selected_index)
        self.filter_combo.blockSignals(False)
        self._apply_filter_patterns()

    def _apply_filter_patterns(self) -> None:
        if self._file_mode == QtFileDialog.FileMode.Directory:
            self._active_patterns = ["*"]
            if self._directory_initialized:
                self._restart_directory_listing()
            return
        index = self.filter_combo.currentIndex()
        if 0 <= index < len(self._name_filters):
            self._selected_name_filter = self._name_filters[index][0]
            self._active_patterns = self._name_filters[index][1]
            if self._directory_initialized:
                self._restart_directory_listing()

    def _filter_changed(self, index: int) -> None:
        if 0 <= index < len(self._name_filters):
            self._selected_name_filter = self._name_filters[index][0]
        self._apply_filter_patterns()

    @staticmethod
    def _path_key(path: Path) -> str:
        return str(_resolve_path(path)).lower()

    def _show_files(self) -> bool:
        return self._file_mode != QtFileDialog.FileMode.Directory or not (
            self._options & QtFileDialog.Option.ShowDirsOnly
        )

    def _file_name_matches_filter(self, name: str) -> bool:
        patterns = [pattern.lower() for pattern in self._active_patterns] or ["*"]
        if "*" in patterns or "*.*" in patterns:
            return True
        lowered = name.lower()
        return any(fnmatch(lowered, pattern) for pattern in patterns)

    def _restart_directory_listing(self) -> None:
        if not hasattr(self, "file_view"):
            return
        if not self._scan_directory():
            return
        self._render_entries()

    def _scan_directory(self) -> bool:
        """Read the current folder once and sort folders-first, alphabetical."""
        self._directory_load_timer.stop()
        entries: list[tuple[str, Path, bool]] = []
        try:
            with os.scandir(self._current_dir) as iterator:
                for entry in iterator:
                    visible, is_dir = self._entry_is_visible(entry)
                    if visible:
                        entries.append((entry.name, Path(entry.path), is_dir))
        except OSError as exc:
            self._all_entries = []
            self._entry_model.clear()
            self._displayed_items.clear()
            self.status_label.setText(f"Could not read folder: {exc}")
            return False
        entries.sort(key=lambda record: (not record[2], record[0].lower()))
        self._all_entries = entries
        return True

    def _render_entries(self) -> None:
        """Rebuild the visible rows from the scan, applying the quick filter."""
        self._directory_load_timer.stop()
        self._entry_model.clear()
        self._displayed_items.clear()
        self._selected_files = []
        self._selected_directory = None

        needle = self._filter_text.strip().lower()
        if needle:
            self._render_queue = [record for record in self._all_entries if needle in record[0].lower()]
        else:
            self._render_queue = list(self._all_entries)
        self._render_index = 0

        if not self.status_label.text().startswith("Choose a valid"):
            if needle and self._all_entries:
                self.status_label.setText(f"{len(self._render_queue)} of {len(self._all_entries)} shown.")
            else:
                self.status_label.setText("")
        if self._render_queue:
            self._directory_load_timer.start()

    def _entry_is_visible(self, entry) -> tuple[bool, bool]:
        name = str(getattr(entry, "name", "") or "")
        if not name or name in {".", ".."} or name.startswith("."):
            return False, False
        try:
            is_dir = bool(entry.is_dir(follow_symlinks=False))
        except OSError:
            return False, False
        if is_dir:
            return True, True
        if not self._show_files():
            return False, False
        try:
            is_file = bool(entry.is_file(follow_symlinks=False))
        except OSError:
            return False, False
        if not is_file or not self._file_name_matches_filter(name):
            return False, False
        return True, False

    def _append_directory_entry(self, name: str, path: Path, *, is_dir: bool) -> None:
        item = QStandardItem(self._folder_icon if is_dir else self._file_icon, name)
        item.setEditable(False)
        item.setData(str(path), PATH_ROLE)
        item.setData(bool(is_dir), IS_DIR_ROLE)
        self._entry_model.appendRow(item)
        self._displayed_items[self._path_key(path)] = item

    def _load_next_directory_batch(self) -> None:
        added = 0
        while added < DIRECTORY_LOAD_BATCH_SIZE and self._render_index < len(self._render_queue):
            name, path, is_dir = self._render_queue[self._render_index]
            self._render_index += 1
            self._append_directory_entry(name, path, is_dir=is_dir)
            added += 1
        if self._render_index >= len(self._render_queue):
            self._directory_load_timer.stop()

    def _on_quick_filter_changed(self, text: str) -> None:
        self._filter_text = str(text or "")
        if self._directory_initialized:
            self._render_entries()

    def _source_index(self, index: QModelIndex) -> QModelIndex:
        if not index.isValid():
            return QModelIndex()
        return index

    def _path_from_index(self, index: QModelIndex) -> Path | None:
        source_index = self._source_index(index)
        if not source_index.isValid():
            return None
        path_text = source_index.data(PATH_ROLE)
        return Path(path_text) if path_text else None

    def _index_for_path(self, path: str | Path) -> QModelIndex:
        key = self._path_key(Path(path))
        item = self._displayed_items.get(key)
        return item.index() if item is not None else QModelIndex()

    def _select_index(self, index: QModelIndex, *, defer_preview: bool = True) -> None:
        path = self._path_from_index(index)
        if path is not None:
            self._select_path(path, defer_preview=defer_preview)

    def _select_path(self, path: Path, *, defer_preview: bool = True) -> None:
        path = _resolve_path(path)
        if path.is_dir():
            self._selected_directory = path
            if self._file_mode == QtFileDialog.FileMode.Directory:
                self.file_name_edit.setText(path.name or str(path))
        else:
            self._selected_directory = None
            self.file_name_edit.setText(path.name)

        if defer_preview:
            self._queue_preview_path(path)
        else:
            self._preview_path(path)

    def _selection_changed(self) -> None:
        paths = self._selected_view_paths()
        if not paths:
            return

        first = paths[0]
        if self._file_mode == QtFileDialog.FileMode.ExistingFiles:
            files = [path for path in paths if path.is_file()]
            self._selected_files = [str(path) for path in files]
            self.file_name_edit.setText("; ".join(path.name for path in files))
        elif first.is_dir():
            self._selected_directory = first
            if self._file_mode == QtFileDialog.FileMode.Directory:
                self.file_name_edit.setText(first.name or str(first))
        else:
            self._selected_files = [str(first)]
            self._selected_directory = None
            self.file_name_edit.setText(first.name)
        self._queue_preview_path(first)

    def _selected_view_paths(self) -> list[Path]:
        selection_model = self.file_view.selectionModel()
        if selection_model is None:
            return []
        paths: list[Path] = []
        seen: set[str] = set()
        for proxy_index in selection_model.selectedRows(0):
            path = self._path_from_index(proxy_index)
            if path is None:
                continue
            path = _resolve_path(path)
            key = str(path).lower()
            if key in seen:
                continue
            seen.add(key)
            paths.append(path)
        return paths

    def _preview_path(self, value: str | Path) -> None:
        self._pending_preview_path = None
        self._preview_timer.stop()
        path = Path(value).expanduser() if value else self._current_dir
        if not path.exists():
            path = self._current_dir
        self._thumbnail_preview.set_directory(path)

    def _queue_preview_path(self, value: str | Path) -> None:
        path = Path(value).expanduser() if value else self._current_dir
        if not path.exists():
            path = self._current_dir
        self._pending_preview_path = path
        self._thumbnail_preview.show_pending_directory(path)
        self._preview_timer.start()

    def _apply_pending_preview(self) -> None:
        path = self._pending_preview_path
        self._pending_preview_path = None
        if path is not None:
            self._thumbnail_preview.set_directory(path)

    def _open_or_accept_index(self, index: QModelIndex) -> None:
        path = self._path_from_index(index)
        if path is None:
            return
        if path.is_dir():
            self._open_directory(path)
            return
        if self._file_mode != QtFileDialog.FileMode.Directory:
            self._select_path(path, defer_preview=False)
            self.accept()

    def _open_directory_index(self, index: QModelIndex) -> bool:
        path = self._path_from_index(index)
        if path is None or not path.is_dir():
            return False
        self._open_directory(path)
        return True

    def _open_directory(self, value: str | Path) -> None:
        directory = Path(value).expanduser()
        if directory.is_file():
            directory = directory.parent
        if not directory.is_dir():
            return
        self.setDirectory(str(directory))
        self.selectFile("")

    def _go_to_parent_directory(self) -> None:
        current = _resolve_path(self._current_dir)
        parent = current.parent
        if parent != current:
            self._open_directory(parent)

    def _refresh_directory(self) -> None:
        self._restart_directory_listing()
        self._queue_preview_path(self._current_dir)

    def _open_path_from_edit(self) -> None:
        path = Path(self.path_edit.text().strip()).expanduser()
        if path.is_dir():
            self._open_directory(path)
        elif path.is_file():
            self._open_directory(path.parent)
            self.selectFile(path.name)

    def _typed_path(self) -> Path | None:
        text = self.file_name_edit.text().strip()
        if not text:
            return None
        path = Path(text.strip('"')).expanduser()
        if not path.is_absolute():
            path = self._current_dir / path
        return path

    def _accepted_paths(self) -> list[Path]:
        if self._file_mode == QtFileDialog.FileMode.Directory:
            typed = self._typed_path()
            candidate = typed if typed is not None else self._selected_directory or self._current_dir
            if candidate.is_dir():
                return [_resolve_path(candidate)]
            return []

        if self._file_mode == QtFileDialog.FileMode.ExistingFiles:
            paths = [path for path in self._selected_view_paths() if path.is_file()]
            if paths:
                return [_resolve_path(path) for path in paths]
            typed = self._typed_path()
            return [_resolve_path(typed)] if typed is not None and typed.is_file() else []

        typed = self._typed_path()
        selected = self._selected_view_paths()
        candidate = typed or (selected[0] if selected else None)
        if candidate is None:
            return []

        if self._accept_mode == QtFileDialog.AcceptMode.AcceptSave:
            suffix = _default_suffix_for_filter(self._selected_name_filter)
            if suffix and not candidate.suffix:
                candidate = candidate.with_suffix(suffix)
            return [_resolve_path(candidate)]

        return [_resolve_path(candidate)] if candidate.is_file() else []

    def _apply_file_mode(self) -> None:
        if not hasattr(self, "file_view"):
            return
        if self._file_mode == QtFileDialog.FileMode.ExistingFiles:
            self.file_view.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        else:
            self.file_view.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)

        is_directory_mode = self._file_mode == QtFileDialog.FileMode.Directory
        self.file_name_label.setText("Folder name" if is_directory_mode else "File name")
        self.filter_combo.setEnabled(not is_directory_mode)
        self._apply_filter_patterns()
        self._update_accept_button()

    def _update_accept_button(self) -> None:
        if not hasattr(self, "button_box"):
            return
        button = self.button_box.button(QDialogButtonBox.StandardButton.Ok)
        if button is None:
            return
        if self._file_mode == QtFileDialog.FileMode.Directory:
            button.setText("Select Folder")
        elif self._accept_mode == QtFileDialog.AcceptMode.AcceptSave:
            button.setText("Save")
        else:
            button.setText("Open")

    def setDirectory(self, directory: str | Path) -> None:
        path = Path(directory).expanduser()
        if path.is_file():
            path = path.parent
        if not path.is_dir():
            path = Path.cwd()
        self._current_dir = _resolve_path(path)
        self.path_edit.setText(str(self._current_dir))
        self._directory_initialized = True
        self._reset_quick_filter()
        self.file_view.setRootIndex(QModelIndex())
        self.file_view.setColumnWidth(0, 320)
        self._restart_directory_listing()
        self._record_recent_directory(self._current_dir)
        self._refresh_places()
        self._queue_preview_path(self._current_dir)

    def _reset_quick_filter(self) -> None:
        if not hasattr(self, "quick_filter_edit"):
            return
        if self.quick_filter_edit.text():
            blocked = self.quick_filter_edit.blockSignals(True)
            self.quick_filter_edit.clear()
            self.quick_filter_edit.blockSignals(blocked)
        self._filter_text = ""

    # ------------------------------------------------------------------
    # Places / recent sidebar
    # ------------------------------------------------------------------
    def _build_places_sidebar(self) -> QListWidget:
        places = QListWidget()
        places.setObjectName("placesSidebar")
        places.setFixedWidth(184)
        places.setFrameShape(QFrame.Shape.NoFrame)
        places.setUniformItemSizes(False)
        places.itemActivated.connect(self._on_place_activated)
        places.itemClicked.connect(self._on_place_activated)
        return places

    def _add_place_header(self, text: str) -> None:
        item = QListWidgetItem(text)
        item.setFlags(Qt.ItemFlag.NoItemFlags)
        font = item.font()
        font.setBold(True)
        item.setFont(font)
        item.setForeground(QColor(0x9A, 0x9A, 0xB0))
        self.places_list.addItem(item)

    def _add_place(self, label: str, path: Path, icon_pixmap: QStyle.StandardPixmap) -> None:
        if not path.is_dir():
            return
        item = QListWidgetItem(self.style().standardIcon(icon_pixmap), label)
        item.setData(Qt.ItemDataRole.UserRole, str(path))
        item.setToolTip(str(path))
        self.places_list.addItem(item)

    def _refresh_places(self) -> None:
        if not hasattr(self, "places_list"):
            return
        self.places_list.clear()
        self._add_place_header("Quick access")
        self._add_place("Home", Path.home(), QStyle.StandardPixmap.SP_DirHomeIcon)

        for label, path in self._workspace_places():
            self._add_place(label, path, QStyle.StandardPixmap.SP_DirIcon)

        drives = [Path(info.absoluteFilePath()) for info in QDir.drives()]
        if drives:
            self._add_place_header("Drives")
            for drive in drives:
                self._add_place(str(drive), drive, QStyle.StandardPixmap.SP_DriveHDIcon)

        recent = [Path(text) for text in self._recent_directories()]
        recent = [path for path in recent if path != self._current_dir and path.is_dir()]
        if recent:
            self._add_place_header("Recent")
            for path in recent:
                self._add_place(path.name or str(path), path, QStyle.StandardPixmap.SP_DirOpenIcon)

    @staticmethod
    def _workspace_places() -> list[tuple[str, Path]]:
        """Workspace shortcuts (incoming/pilot, results, ...) when discoverable."""
        try:
            from triton_analysis.workspace import workspace_paths

            workspace = workspace_paths(create=False)
        except Exception:
            return []
        candidates = [
            ("Pilot incoming", getattr(workspace, "pilot_incoming", None)),
            ("Results", getattr(workspace, "results", None)),
            ("Reports", getattr(workspace, "reports", None)),
            ("Calibrations", getattr(workspace, "calibrations", None)),
        ]
        return [(label, Path(path)) for label, path in candidates if path is not None and Path(path).is_dir()]

    def _on_place_activated(self, item: QListWidgetItem) -> None:
        path_text = item.data(Qt.ItemDataRole.UserRole)
        if path_text:
            self._open_directory(str(path_text))

    def _recent_directories(self) -> list[str]:
        raw = self._dialog_settings.value("recent_dirs", [])
        if isinstance(raw, str):
            raw = [raw]
        return [str(entry) for entry in (raw or []) if str(entry).strip()]

    def _record_recent_directory(self, path: Path) -> None:
        text = str(path)
        recent = [entry for entry in self._recent_directories() if entry.lower() != text.lower()]
        recent.insert(0, text)
        self._dialog_settings.setValue("recent_dirs", recent[:RECENT_DIRS_LIMIT])

    def directory(self) -> QDir:
        return QDir(str(self._current_dir))

    def setFileMode(self, mode: QtFileDialog.FileMode) -> None:
        self._file_mode = mode
        self._apply_file_mode()

    def fileMode(self) -> QtFileDialog.FileMode:
        return self._file_mode

    def setAcceptMode(self, mode: QtFileDialog.AcceptMode) -> None:
        self._accept_mode = mode
        self._update_accept_button()

    def setOptions(self, options: QtFileDialog.Option) -> None:
        self._options = options
        self._apply_file_mode()

    def setLabelText(self, label: QtFileDialog.DialogLabel, text: str) -> None:
        if label == QtFileDialog.DialogLabel.Accept:
            button = self.button_box.button(QDialogButtonBox.StandardButton.Ok)
            if button is not None:
                button.setText(text)

    def selectNameFilter(self, selected_filter: str) -> None:
        for index, (label, _patterns) in enumerate(self._name_filters):
            if label == selected_filter:
                self.filter_combo.setCurrentIndex(index)
                self._selected_name_filter = label
                self._apply_filter_patterns()
                return

    def selectedNameFilter(self) -> str:
        return self._selected_name_filter

    def selectFile(self, value: str | Path) -> None:
        text = str(value or "").strip()
        self.file_name_edit.setText(text)
        if not text:
            self._selected_files = []
            self._selected_directory = None
            return

        path = Path(text).expanduser()
        if not path.is_absolute():
            path = self._current_dir / path
        if path.exists():
            path = _resolve_path(path)
            if path.is_dir():
                self._selected_directory = path
            elif path.is_file():
                self._selected_files = [str(path)]
            index = self._index_for_path(path)
            if index.isValid():
                selection = self.file_view.selectionModel()
                if selection is not None:
                    selection.select(
                        index,
                        QItemSelectionModel.SelectionFlag.ClearAndSelect | QItemSelectionModel.SelectionFlag.Rows,
                    )
                    self.file_view.setCurrentIndex(index)
                    self.file_view.scrollTo(index)

    def selectedFiles(self) -> list[str]:
        if self._selected_files:
            return list(self._selected_files)
        accepted = self._accepted_paths()
        return [str(path) for path in accepted]

    def _wire_file_views(self) -> None:
        return

    def accept(self) -> None:
        paths = self._accepted_paths()
        if not paths:
            self.status_label.setText(
                "Choose a valid folder."
                if self._file_mode == QtFileDialog.FileMode.Directory
                else "Choose a valid file."
            )
            return
        self.status_label.setText("")
        self._selected_files = [str(path) for path in paths]
        if self._file_mode == QtFileDialog.FileMode.Directory:
            self._selected_directory = paths[0]
        super().accept()

    @staticmethod
    def getExistingDirectory(
        parent=None,
        caption: str = "",
        directory: str = "",
        options: QtFileDialog.Option = QtFileDialog.Option.ShowDirsOnly,
    ) -> str:
        dialog = ThumbnailFileDialog(parent, caption, directory)
        dialog.setFileMode(QtFileDialog.FileMode.Directory)
        dialog.setOptions(options)
        dialog.setLabelText(QtFileDialog.DialogLabel.Accept, "Select Folder")
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
        dialog.setOptions(options)
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
        dialog.setOptions(options)
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
        dialog.setOptions(options)
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
