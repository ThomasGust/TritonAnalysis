"""Command-line stereo calibration from TritonPilot stereo capture sessions."""

from __future__ import annotations

import argparse
from pathlib import Path

from stereo_calibration import (
    DEFAULT_CHARUCO_DICTIONARY,
    CharucoBoardSpec,
    CheckerboardSpec,
    calibrate_stereo_from_observations,
    collect_charuco_observations,
    collect_checkerboard_observations,
    load_manifest_collection,
    write_calibration_artifact,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calibrate a stereo rig from a TritonPilot stereo manifest.")
    parser.add_argument("sources", nargs="+", help="TritonPilot stereo manifest file(s) or session folder(s).")
    parser.add_argument("--output", default="", help="Output calibration JSON path.")
    parser.add_argument("--rig-id", default="", help="Rig id to store in the artifact.")
    parser.add_argument("--pair-name", default="", help="Stereo pair name to store in the artifact.")
    parser.add_argument("--min-pairs", type=int, default=8, help="Minimum accepted stereo pairs required.")

    board = parser.add_mutually_exclusive_group(required=True)
    board.add_argument("--checkerboard", action="store_true", help="Use checkerboard inner-corner detection.")
    board.add_argument("--charuco", action="store_true", help="Use ChArUco board detection.")

    parser.add_argument("--columns", type=int, default=9, help="Checkerboard inner columns.")
    parser.add_argument("--rows", type=int, default=6, help="Checkerboard inner rows.")
    parser.add_argument("--square-size", type=float, default=30.0, help="Square size in board units.")
    parser.add_argument("--units", default="mm", help="Board units, for artifact metadata.")
    parser.add_argument("--squares-x", type=int, default=24, help="ChArUco board squares in X / columns.")
    parser.add_argument("--squares-y", type=int, default=17, help="ChArUco board squares in Y / rows.")
    parser.add_argument("--marker-size", type=float, default=22.0, help="ChArUco marker size in board units.")
    parser.add_argument("--dictionary", default=DEFAULT_CHARUCO_DICTIONARY, help="OpenCV aruco predefined dictionary name.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    manifest, image_pairs = load_manifest_collection([Path(source) for source in args.sources])
    if not image_pairs:
        raise SystemExit("No image pairs found in stereo manifest source(s)")

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
        rig_id=args.rig_id or str((manifest.get("pair") or {}).get("rig_id") or "stereo_rig"),
        pair_name=args.pair_name or str((manifest.get("pair") or {}).get("name") or "stereo_pair"),
        board_spec=board,
    )
    first_source = Path(args.sources[0])
    default_parent = first_source if first_source.is_dir() else first_source.parent
    out_path = Path(args.output) if args.output else default_parent / "stereo_calibration.json"
    write_calibration_artifact(artifact, out_path)
    quality = artifact.get("quality") or {}
    epipolar = quality.get("epipolar") or {}
    left_coverage = quality.get("left_coverage") or {}
    right_coverage = quality.get("right_coverage") or {}
    print(f"Accepted pairs: {artifact['observation_count']}")
    print(f"Stereo RMS: {artifact['rms']['stereo']:.4f} px")
    print(f"Epipolar RMS: {float(epipolar.get('rms_px') or 0.0):.4f} px")
    print(
        "Coverage: left {left:.0%}, right {right:.0%}".format(
            left=float(left_coverage.get("area_fraction") or 0.0),
            right=float(right_coverage.get("area_fraction") or 0.0),
        )
    )
    print(f"Baseline ({args.units}): {artifact['stereo']['baseline']:.4f}")
    for warning in quality.get("warnings") or []:
        print(f"Warning: {warning}")
    print(f"Calibration: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
