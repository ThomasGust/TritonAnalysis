"""Entry point for the RealityScan stereo reconstruction GUI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from PyQt6.QtWidgets import QApplication

from triton_analysis.gui.realityscan_reconstruction_window import RealityScanReconstructionWindow
from triton_analysis.gui.style import apply_modern_style


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Standalone GUI wrapper for Triton stereo RealityScan reconstruction.",
    )
    parser.add_argument("session", nargs="?", help="Stereo session folder or manifest.json.")
    parser.add_argument("--calibration", help="Stereo calibration JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    app = QApplication(sys.argv)
    apply_modern_style(app)

    window = RealityScanReconstructionWindow(
        session_path=args.session,
        calibration_path=args.calibration,
    )
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
