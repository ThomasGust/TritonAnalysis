"""Entry point for the RealityScan metric model viewer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from PyQt6.QtWidgets import QApplication

from gui.realityscan_model_viewer_window import RealityScanModelViewerWindow
from gui.style import apply_modern_style


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Open a RealityScan OBJ model in a Three.js measurement viewport.",
    )
    parser.add_argument("model", nargs="?", help="OBJ model path, usually underwater_model_metric.obj.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    app = QApplication(sys.argv)
    apply_modern_style(app)

    window = RealityScanModelViewerWindow(model_path=args.model)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
