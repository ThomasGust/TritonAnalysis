"""Entry point for the OpenAI-assisted crab counter GUI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from PyQt6.QtWidgets import QApplication

from triton_analysis.gui.crab_counter_window import CrabCounterWindow
from triton_analysis.gui.style import apply_modern_style


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GUI for counting European green crabs in saved board images.")
    parser.add_argument("image", nargs="?", default=None, help="Optional crab-board image to load at startup.")
    parser.add_argument("--workspace", default=None, help="TritonAnalysis workspace root.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    app = QApplication(sys.argv)
    apply_modern_style(app)

    window = CrabCounterWindow(image_path=args.image, workspace_root=args.workspace)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
