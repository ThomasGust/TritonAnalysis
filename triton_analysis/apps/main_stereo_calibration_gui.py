"""Entry point for the standalone stereo calibration GUI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from PyQt6.QtWidgets import QApplication

from triton_analysis.gui.stereo_calibration_window import StereoCalibrationWindow
from triton_analysis.gui.style import apply_modern_style


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone GUI for stereo calibration sessions.")
    parser.add_argument("manifest", nargs="?", help="Optional TritonPilot stereo manifest.json to load.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    app = QApplication(sys.argv)
    apply_modern_style(app)

    window = StereoCalibrationWindow(manifest_path=args.manifest)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
