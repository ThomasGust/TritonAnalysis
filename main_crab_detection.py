"""Entry point for the standalone crab detection GUI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from PyQt6.QtWidgets import QApplication

from gui.crab_detection_window import CrabDetectionWindow
from gui.style import apply_modern_style
from tools.crab_yolo_predict import DEFAULT_CONFIDENCE


def build_parser() -> argparse.ArgumentParser:
    """Build the crab detection command-line parser."""
    parser = argparse.ArgumentParser(
        description="Standalone GUI for reference-board European green crab detection.",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="Image files or folders to load when the app starts.",
    )
    parser.add_argument(
        "--reference-image",
        default=None,
        help="Optional crab-board reference image. Defaults to TRITON_CRAB_REFERENCE_IMAGE or the known TritonPilot recording.",
    )
    parser.add_argument(
        "--detector",
        choices=("auto", "yolo", "board"),
        default="auto",
        help="Detector mode. Auto uses the newest trained YOLO model when available, otherwise board projection.",
    )
    parser.add_argument(
        "--yolo-model",
        default=None,
        help="Optional YOLO .pt weights for European green crab detection.",
    )
    parser.add_argument(
        "--yolo-conf",
        type=float,
        default=DEFAULT_CONFIDENCE,
        help="YOLO confidence threshold.",
    )
    parser.add_argument(
        "--no-force-square",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--unwrap-size",
        nargs=2,
        type=int,
        metavar=("WIDTH", "HEIGHT"),
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Launch the crab detection applet."""
    parser = build_parser()
    args = parser.parse_args(argv)

    app = QApplication(sys.argv)
    apply_modern_style(app)

    window = CrabDetectionWindow(
        image_paths=args.paths,
        reference_image=args.reference_image,
        detector_mode=args.detector,
        yolo_model=args.yolo_model,
        yolo_confidence=args.yolo_conf,
    )
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
