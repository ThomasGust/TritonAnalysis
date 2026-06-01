"""Entry point for the standalone stereo segment measurement GUI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from PyQt6.QtWidgets import QApplication

from triton_analysis.gui.stereo_segment_measurement_window import StereoSegmentMeasurementWindow
from triton_analysis.gui.style import apply_modern_style
from triton_analysis.stereo.segment_measurement import STEREO_SEGMENT_PRESETS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone GUI for stereo segment measurement.")
    parser.add_argument("manifest", nargs="?", help="Optional TritonPilot stereo manifest.json or session folder.")
    parser.add_argument("--calibration", default="", help="Optional stereo_calibration.json artifact.")
    parser.add_argument(
        "--preset",
        default="generic",
        choices=[preset.key for preset in STEREO_SEGMENT_PRESETS],
        help="Measurement preset to select on startup.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    app = QApplication(sys.argv)
    apply_modern_style(app)

    window = StereoSegmentMeasurementWindow(
        manifest_path=args.manifest,
        calibration_path=args.calibration or None,
        preset_key=args.preset,
    )
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
