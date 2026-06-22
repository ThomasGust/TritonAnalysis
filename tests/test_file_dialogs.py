import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtGui import QColor, QImage
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QFileDialog

from triton_analysis.gui.file_dialogs import DirectoryThumbnailPreview, ThumbnailFileDialog


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def _write_test_image(path):
    image = QImage(32, 24, QImage.Format.Format_RGB32)
    image.fill(QColor("green"))
    assert image.save(str(path))


def _dialog_index_for_path(dialog: ThumbnailFileDialog, path):
    for _attempt in range(100):
        _app().processEvents()
        index = dialog._index_for_path(path)
        if index.isValid():
            return index
        QTest.qWait(5)
    pytest.fail(f"Could not find dialog index for {path}")


def _wait_for_preview(preview: DirectoryThumbnailPreview, count: int):
    for _attempt in range(100):
        _app().processEvents()
        if preview.thumbnail_list.count() == count and not preview.status_label.text().startswith("Loading"):
            return
        QTest.qWait(5)
    pytest.fail(f"Preview did not finish loading {count} thumbnails")


def test_directory_thumbnail_preview_shows_images_in_selected_folder(tmp_path):
    app = _app()
    timestamp_dir = tmp_path / "20260531-120000"
    timestamp_dir.mkdir()
    image_path = timestamp_dir / "frame_0001.png"
    _write_test_image(image_path)

    preview = DirectoryThumbnailPreview()
    try:
        preview.set_directory(timestamp_dir)
        _wait_for_preview(preview, 1)

        assert preview.thumbnail_list.count() == 1
        assert preview.thumbnail_list.item(0).text() == "frame_0001.png"
        assert preview.status_label.text() == "1 image."
    finally:
        preview.deleteLater()
        app.processEvents()


def test_directory_thumbnail_preview_shows_stereo_session_subfolder_images(tmp_path):
    app = _app()
    session_dir = tmp_path / "20260531-120000_stereo"
    left_dir = session_dir / "left"
    right_dir = session_dir / "right"
    left_dir.mkdir(parents=True)
    right_dir.mkdir()
    for path in (left_dir / "pair_000001_left.png", right_dir / "pair_000001_right.png"):
        _write_test_image(path)
    (session_dir / "manifest.json").write_text(
        """
        {
          "frames": [
            {
              "left_path": "left/pair_000001_left.png",
              "right_path": "right/pair_000001_right.png"
            }
          ]
        }
        """,
        encoding="utf-8",
    )

    preview = DirectoryThumbnailPreview()
    try:
        preview.set_directory(session_dir)
        _wait_for_preview(preview, 2)

        assert preview.thumbnail_list.count() == 2
        assert preview.thumbnail_list.item(0).text() == "left\\pair_000001_left.png"
        assert "session images" in preview.status_label.text()
    finally:
        preview.deleteLater()
        app.processEvents()


def test_thumbnail_file_dialog_attaches_preview_panel(tmp_path):
    app = _app()
    dialog = ThumbnailFileDialog(None, "Open media", str(tmp_path))
    try:
        panels = dialog.findChildren(DirectoryThumbnailPreview)

        assert len(panels) == 1
        assert panels[0].isVisibleTo(dialog) or not dialog.isVisible()
        assert dialog.minimumWidth() >= 980
        assert dialog.minimumHeight() >= 640
    finally:
        dialog.deleteLater()
        app.processEvents()


def test_thumbnail_file_dialog_moves_to_parent_folder(tmp_path):
    app = _app()
    child = tmp_path / "timestamped-session"
    child.mkdir()
    dialog = ThumbnailFileDialog(None, "Open media", str(child))
    try:
        dialog._go_to_parent_directory()

        assert dialog.directory().absolutePath().replace("/", "\\").endswith(str(tmp_path))
    finally:
        dialog.deleteLater()
        app.processEvents()


def test_thumbnail_file_dialog_previews_highlighted_child_folder_without_entering(tmp_path):
    app = _app()
    child = tmp_path / "timestamped-session"
    child.mkdir()
    _write_test_image(child / "frame_0001.png")
    dialog = ThumbnailFileDialog(None, "Open media", str(tmp_path))
    try:
        dialog._select_index(_dialog_index_for_path(dialog, child), defer_preview=False)

        panels = dialog.findChildren(DirectoryThumbnailPreview)
        _wait_for_preview(panels[0], 1)
        assert panels[0].path_label.text() == "timestamped-session"
        assert panels[0].thumbnail_list.count() == 1
        assert panels[0].thumbnail_list.item(0).text() == "frame_0001.png"
        assert Path(dialog.directory().absolutePath()).resolve() == tmp_path.resolve()
    finally:
        dialog.deleteLater()
        app.processEvents()


def test_thumbnail_file_dialog_single_click_selects_directory_without_entering(tmp_path):
    app = _app()
    child = tmp_path / "timestamped-session"
    child.mkdir()
    dialog = ThumbnailFileDialog(None, "Choose folder", str(tmp_path))
    try:
        dialog.setFileMode(QFileDialog.FileMode.Directory)
        dialog._wire_file_views()
        dialog._select_index(_dialog_index_for_path(dialog, child), defer_preview=False)

        assert dialog._selected_directory == child.resolve()
        assert Path(dialog.directory().absolutePath()).resolve() == tmp_path.resolve()
    finally:
        dialog.deleteLater()
        app.processEvents()


def test_thumbnail_file_dialog_single_and_double_click_folder_flow(tmp_path):
    app = _app()
    child = tmp_path / "timestamped-session"
    child.mkdir()
    dialog = ThumbnailFileDialog(None, "Choose folder", str(tmp_path))
    try:
        dialog.setFileMode(QFileDialog.FileMode.Directory)
        dialog._wire_file_views()
        index = _dialog_index_for_path(dialog, child)

        dialog.file_view.clicked.emit(index)
        app.processEvents()

        assert dialog._selected_directory == child.resolve()
        assert Path(dialog.directory().absolutePath()).resolve() == tmp_path.resolve()

        dialog.file_view.doubleClicked.emit(index)
        app.processEvents()

        assert Path(dialog.directory().absolutePath()).resolve() == child.resolve()
    finally:
        dialog.deleteLater()
        app.processEvents()


def test_thumbnail_file_dialog_opens_highlighted_child_folder(tmp_path):
    app = _app()
    child = tmp_path / "timestamped-session"
    child.mkdir()
    dialog = ThumbnailFileDialog(None, "Open media", str(tmp_path))
    try:
        assert dialog._open_directory_index(_dialog_index_for_path(dialog, child))

        assert Path(dialog.directory().absolutePath()).resolve() == child.resolve()
    finally:
        dialog.deleteLater()
        app.processEvents()


def _wait_for_rows(dialog: ThumbnailFileDialog, count: int):
    for _attempt in range(100):
        _app().processEvents()
        if dialog._entry_model.rowCount() >= count:
            return
        QTest.qWait(5)
    pytest.fail(f"Directory listing did not reach {count} rows")


def _rows(dialog: ThumbnailFileDialog) -> list[str]:
    return [dialog._entry_model.item(i).text() for i in range(dialog._entry_model.rowCount())]


def test_directory_listing_sorts_folders_first_then_alphabetical(tmp_path):
    app = _app()
    for name in ("zebra", "alpha", "Mango"):
        (tmp_path / name).mkdir()
    for name in ("readme.txt", "banana.txt", "Apple.txt"):
        (tmp_path / name).write_text("x", encoding="utf-8")
    dialog = ThumbnailFileDialog(None, "Open", str(tmp_path), "All files (*)")
    try:
        _wait_for_rows(dialog, 6)
        assert _rows(dialog) == ["alpha", "Mango", "zebra", "Apple.txt", "banana.txt", "readme.txt"]
    finally:
        dialog.deleteLater()
        app.processEvents()


def test_quick_filter_narrows_listing_without_rescanning(tmp_path):
    app = _app()
    (tmp_path / "Mango").mkdir()
    (tmp_path / "alpha").mkdir()
    for name in ("banana.txt", "Apple.txt"):
        (tmp_path / name).write_text("x", encoding="utf-8")
    dialog = ThumbnailFileDialog(None, "Open", str(tmp_path), "All files (*)")
    try:
        _wait_for_rows(dialog, 4)

        dialog.quick_filter_edit.setText("an")
        for _attempt in range(100):
            app.processEvents()
            rows = _rows(dialog)
            if rows and all("an" in row.lower() for row in rows):
                break
            QTest.qWait(5)
        assert set(_rows(dialog)) == {"Mango", "banana.txt"}
        assert "of 4 shown" in dialog.status_label.text()

        dialog.quick_filter_edit.clear()
        _wait_for_rows(dialog, 4)
        assert len(_rows(dialog)) == 4
    finally:
        dialog.deleteLater()
        app.processEvents()


def test_thumbnail_preview_renders_grid_before_decode_finishes(tmp_path):
    app = _app()
    image_path = tmp_path / "frame_0001.png"
    _write_test_image(image_path)

    preview = DirectoryThumbnailPreview()
    try:
        preview.set_directory(tmp_path)
        # The tile and final status appear immediately, before the off-thread
        # decode has had a chance to run.
        assert preview.thumbnail_list.count() == 1
        assert not preview.status_label.text().startswith("Loading")

        for _attempt in range(100):
            app.processEvents()
            if image_path.resolve() in preview._icon_cache:
                break
            QTest.qWait(5)
        assert image_path.resolve() in preview._icon_cache
    finally:
        preview.deleteLater()
        app.processEvents()


def test_places_sidebar_lists_home_and_navigates(tmp_path):
    app = _app()
    child = tmp_path / "destination"
    child.mkdir()
    dialog = ThumbnailFileDialog(None, "Open", str(tmp_path))
    try:
        labels = [dialog.places_list.item(i).text() for i in range(dialog.places_list.count())]
        assert "Quick access" in labels
        assert "Home" in labels

        home_item = next(
            dialog.places_list.item(i)
            for i in range(dialog.places_list.count())
            if dialog.places_list.item(i).text() == "Home"
        )
        dialog._on_place_activated(home_item)
        assert Path(dialog.directory().absolutePath()).resolve() == Path.home().resolve()
    finally:
        dialog.deleteLater()
        app.processEvents()
