"""Entry point for the standalone iceberg tracking GUI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from PyQt6.QtWidgets import QApplication

from gui.iceberg_tracking_window import IcebergTrackingWindow
from gui.style import apply_modern_style


def build_parser() -> argparse.ArgumentParser:
    """Build the iceberg tracking command-line parser."""
    return argparse.ArgumentParser(
        description="Standalone GUI for MATE ROV iceberg tracking threat assessment.",
    )


def main(argv: list[str] | None = None) -> int:
    """Launch the iceberg tracking applet."""
    parser = build_parser()
    parser.parse_args(argv)

    app = QApplication(sys.argv)
    apply_modern_style(app)

    window = IcebergTrackingWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
