import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtGui import QColor, QImage
from PyQt6.QtWidgets import QApplication

from triton_analysis.gui.file_dialogs import DirectoryThumbnailPreview, ThumbnailFileDialog


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_directory_thumbnail_preview_shows_images_in_selected_folder(tmp_path):
    app = _app()
    timestamp_dir = tmp_path / "20260531-120000"
    timestamp_dir.mkdir()
    image_path = timestamp_dir / "frame_0001.png"
    image = QImage(32, 24, QImage.Format.Format_RGB32)
    image.fill(QColor("red"))
    assert image.save(str(image_path))

    preview = DirectoryThumbnailPreview()
    try:
        preview.set_directory(timestamp_dir)

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
        image = QImage(32, 24, QImage.Format.Format_RGB32)
        image.fill(QColor("blue"))
        assert image.save(str(path))
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
