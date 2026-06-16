"""Entry point for the unified TritonAnalysis competition GUI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from PyQt6.QtWidgets import QApplication

from triton_analysis.gui.style import apply_modern_style
from triton_analysis.gui.triton_analysis_window import TritonAnalysisWindow


TAB_CHOICES = (
    "coral-reconstruction",
    "stereo-iceberg-length",
    "iceberg-tracking",
    "edna-analysis",
    "crab-counter",
    "crab-dataset",
    "stereo-calibration",
    "backup-coral-measurement",
    "backup-iceberg-measurement",
    "ssh",
    "iceberg-measurement",
    "edna",
    "crab",
    "crab-count",
    "terminal",
)


def build_parser() -> argparse.ArgumentParser:
    """Build the unified analysis app command-line parser."""
    parser = argparse.ArgumentParser(
        description="Unified TritonAnalysis GUI for competition-day analysis tasks.",
    )
    parser.add_argument(
        "--backup-coral",
        nargs="*",
        default=None,
        metavar="PATH",
        help="Optional image or video file to preload in the Backup Coral Measurement tab.",
    )
    parser.add_argument(
        "--backup-iceberg",
        nargs="*",
        default=None,
        metavar="PATH",
        help="Optional image or video file to preload in the Backup Iceberg Measurement tab.",
    )
    parser.add_argument(
        "--stereo-manifest",
        default="",
        help="Optional TritonPilot stereo manifest.json or session folder for stereo tabs.",
    )
    parser.add_argument(
        "--calibration",
        default="",
        help="Optional stereo_calibration.json artifact for stereo measurement and reconstruction.",
    )
    parser.add_argument(
        "--reconstruction-session",
        default="",
        help="Optional stereo session or manifest for the Coral Reconstruction tab.",
    )
    parser.add_argument(
        "--edna-sample",
        action="store_true",
        help="Load the sample eDNA counts at startup.",
    )
    parser.add_argument(
        "--tab",
        choices=TAB_CHOICES,
        default="coral-reconstruction",
        help="Initial tab to show.",
    )
    parser.add_argument(
        "--pilot-transfer-url",
        default=None,
        help="TritonPilot transfer URL for automatic media sync.",
    )
    parser.add_argument(
        "--pilot-transfer-output",
        default=None,
        help="Folder where automatic TritonPilot media sync writes files.",
    )
    parser.add_argument(
        "--workspace",
        default=None,
        help="Portable TritonAnalysis workspace root for incoming files, reports, and results.",
    )
    parser.add_argument(
        "--no-pilot-auto-sync",
        action="store_true",
        help="Open the unified app with automatic TritonPilot media sync disabled.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Launch the unified analysis app."""
    args = build_parser().parse_args(argv)

    app = QApplication(sys.argv)
    apply_modern_style(app)

    window = TritonAnalysisWindow(
        backup_coral_paths=args.backup_coral,
        backup_iceberg_paths=args.backup_iceberg,
        stereo_manifest_path=args.stereo_manifest or None,
        stereo_calibration_path=args.calibration or None,
        reconstruction_session_path=args.reconstruction_session or None,
        use_sample_edna=args.edna_sample,
        initial_tab=args.tab,
        pilot_transfer_url=args.pilot_transfer_url,
        pilot_transfer_output=args.pilot_transfer_output,
        pilot_transfer_auto_sync=False if args.no_pilot_auto_sync else None,
        workspace_root=args.workspace,
    )
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
