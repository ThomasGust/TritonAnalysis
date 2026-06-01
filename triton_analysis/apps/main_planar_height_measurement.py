"""Entry point for the standalone planar height measurement GUI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from PyQt6.QtWidgets import QApplication

from triton_analysis.gui.planar_height_measurement_window import PlanarHeightMeasurementWindow
from triton_analysis.gui.style import apply_modern_style


def build_parser() -> argparse.ArgumentParser:
    """Build the planar height measurement command-line parser."""
    parser = argparse.ArgumentParser(
        description="Standalone GUI for planar prop height measurement.",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="Optional image or video file to load when the app starts.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Launch the planar height measurement applet."""
    parser = build_parser()
    args = parser.parse_args(argv)

    app = QApplication(sys.argv)
    apply_modern_style(app)

    window = PlanarHeightMeasurementWindow(media_paths=args.paths)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
