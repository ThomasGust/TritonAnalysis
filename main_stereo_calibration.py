"""Command-line stereo calibration from TritonPilot stereo capture sessions."""

from __future__ import annotations

import argparse
from pathlib import Path

from stereo_calibration import (
    CharucoBoardSpec,
    CheckerboardSpec,
    calibrate_stereo_from_observations,
    collect_charuco_observations,
    collect_checkerboard_observations,
    manifest_image_pairs,
    write_calibration_artifact,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calibrate a stereo rig from a TritonPilot stereo manifest.")
    parser.add_argument("manifest", help="Path to TritonPilot stereo manifest.json.")
    parser.add_argument("--output", default="", help="Output calibration JSON path.")
    parser.add_argument("--rig-id", default="", help="Rig id to store in the artifact.")
    parser.add_argument("--pair-name", default="", help="Stereo pair name to store in the artifact.")
    parser.add_argument("--min-pairs", type=int, default=8, help="Minimum accepted stereo pairs required.")

    board = parser.add_mutually_exclusive_group(required=True)
    board.add_argument("--checkerboard", action="store_true", help="Use checkerboard inner-corner detection.")
    board.add_argument("--charuco", action="store_true", help="Use ChArUco board detection.")

    parser.add_argument("--columns", type=int, default=9, help="Checkerboard inner columns.")
    parser.add_argument("--rows", type=int, default=6, help="Checkerboard inner rows.")
    parser.add_argument("--square-size", type=float, required=True, help="Square size in board units.")
    parser.add_argument("--units", default="cm", help="Board units, for artifact metadata.")
    parser.add_argument("--squares-x", type=int, default=7, help="ChArUco board squares in X.")
    parser.add_argument("--squares-y", type=int, default=5, help="ChArUco board squares in Y.")
    parser.add_argument("--marker-size", type=float, default=0.0, help="ChArUco marker size in board units.")
    parser.add_argument("--dictionary", default="DICT_4X4_50", help="OpenCV aruco predefined dictionary name.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    manifest_path = Path(args.manifest)
    image_pairs = manifest_image_pairs(manifest_path)
    if not image_pairs:
        raise SystemExit(f"No image pairs found in manifest: {manifest_path}")

    if args.checkerboard:
        board = CheckerboardSpec(
            columns=args.columns,
            rows=args.rows,
            square_size=args.square_size,
            units=args.units,
        )
        observations = collect_checkerboard_observations(image_pairs, board, min_pairs=args.min_pairs)
    else:
        marker_size = args.marker_size if args.marker_size > 0 else args.square_size * 0.7
        board = CharucoBoardSpec(
            squares_x=args.squares_x,
            squares_y=args.squares_y,
            square_size=args.square_size,
            marker_size=marker_size,
            dictionary=args.dictionary,
            units=args.units,
        )
        observations = collect_charuco_observations(image_pairs, board, min_pairs=args.min_pairs)

    artifact = calibrate_stereo_from_observations(
        observations,
        rig_id=args.rig_id or "stereo_rig",
        pair_name=args.pair_name or "stereo_pair",
        board_spec=board,
    )
    out_path = Path(args.output) if args.output else manifest_path.parent / "stereo_calibration.json"
    write_calibration_artifact(artifact, out_path)
    print(f"Accepted pairs: {artifact['observation_count']}")
    print(f"Stereo RMS: {artifact['rms']['stereo']:.4f}")
    print(f"Baseline ({args.units}): {artifact['stereo']['baseline']:.4f}")
    print(f"Calibration: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
