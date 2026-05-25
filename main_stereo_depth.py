"""Entry point for the standalone stereo depth GUI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from PyQt6.QtWidgets import QApplication

from gui.stereo_depth_window import StereoDepthWindow
from gui.style import apply_modern_style


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone GUI for stereo depth and 3D length checks.")
    parser.add_argument("manifest", nargs="?", help="Optional TritonPilot stereo manifest.json or session folder.")
    parser.add_argument("--calibration", default="", help="Optional stereo_calibration.json artifact.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    app = QApplication(sys.argv)
    apply_modern_style(app)

    window = StereoDepthWindow(
        manifest_path=args.manifest,
        calibration_path=args.calibration or None,
    )
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())

