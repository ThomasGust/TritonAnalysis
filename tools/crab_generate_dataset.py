"""Generate a synthetic YOLO dataset for the crab detection task."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from triton_analysis.crab.synthetic import (
    CRAB_CLASS_NAMES,
    SyntheticDatasetConfig,
    discover_background_media,
    generate_synthetic_dataset,
)
from triton_analysis.workspace import fresh_output_subdir, workspace_paths


def _default_downloads_reference_paths() -> dict[str, Path]:
    downloads = Path.home() / "Downloads"
    candidates = {
        "european_green_crab": (
            downloads / "European Green Crab Image (1).jpg",
            downloads / "European Green Crab Image.jpg",
            downloads / "crab" / "European Green Crab Image.jpg",
        ),
        "native_rock_crab": (
            downloads / "Native Rock Crab (1).jpg",
            downloads / "Native Rock Crab.jpg",
            downloads / "crab" / "Native Rock Crab.jpg",
        ),
        "jonah_crab": (
            downloads / "Jonah crab 2 (1).png",
            downloads / "Jonah crab 2.png",
            downloads / "crab" / "Jonah crab 2.png",
        ),
    }
    defaults: dict[str, Path] = {}
    for class_name, paths in candidates.items():
        for path in paths:
            if path.exists():
                defaults[class_name] = path
                break
    return defaults


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate synthetic YOLO training data for the MATE ROV crab task.",
    )
    parser.add_argument("--workspace", default=None, help="TritonAnalysis workspace root.")
    parser.add_argument("--output", default=None, help="Dataset output folder. Defaults to Workspace/datasets/crab_synth_TIMESTAMP.")
    parser.add_argument("--count", type=int, default=2000, help="Number of synthetic images to generate.")
    parser.add_argument("--width", type=int, default=1280, help="Output image width.")
    parser.add_argument("--height", type=int, default=720, help="Output image height.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for repeatable generation.")
    parser.add_argument("--val-fraction", type=float, default=0.2, help="Approximate validation split fraction.")
    parser.add_argument("--min-crabs", type=int, default=4, help="Minimum crabs per non-empty image.")
    parser.add_argument("--max-crabs", type=int, default=11, help="Maximum crabs per non-empty image.")
    parser.add_argument("--min-crab-scale", type=float, default=0.10, help="Minimum normal crab long-edge size as a fraction of board size.")
    parser.add_argument("--max-crab-scale", type=float, default=0.42, help="Maximum normal crab long-edge size as a fraction of board size.")
    parser.add_argument("--large-crab-fraction", type=float, default=0.32, help="Fraction of placed crabs sampled from the larger printed-crab size range.")
    parser.add_argument("--large-crab-min-scale", type=float, default=0.34, help="Minimum long-edge size for occasional large crabs as a fraction of board size.")
    parser.add_argument("--large-crab-max-scale", type=float, default=0.52, help="Maximum long-edge size for occasional large crabs as a fraction of board size.")
    parser.add_argument("--sparse-layout-fraction", type=float, default=0.25, help="Fraction of boards using the sparse crab-count/coverage profile.")
    parser.add_argument("--full-layout-fraction", type=float, default=0.30, help="Fraction of boards using the fuller crab-count/coverage profile.")
    parser.add_argument("--even-placement-fraction", type=float, default=0.55, help="Fraction of boards using an evenly distributed placement profile.")
    parser.add_argument("--even-placement-jitter", type=float, default=0.36, help="Jitter applied to even placement cells; larger values are less grid-like.")
    parser.add_argument("--max-crab-overlap", type=float, default=0.025, help="Maximum allowed IoU between crab boxes on the synthetic board.")
    parser.add_argument("--crab-spacing", type=float, default=0.012, help="Minimum extra spacing around crab boxes as a fraction of board size.")
    parser.add_argument("--min-crab-box-long-edge", type=int, default=40, help="Minimum final rendered crab box long edge in pixels; scenes below this are resampled.")
    parser.add_argument("--min-board-scale", type=float, default=0.58, help="Minimum board long-edge size relative to the shorter image dimension.")
    parser.add_argument("--max-board-scale", type=float, default=1.18, help="Maximum board long-edge size relative to the shorter image dimension.")
    parser.add_argument("--max-board-roll", type=float, default=50.0, help="Maximum in-plane board rotation in degrees.")
    parser.add_argument("--min-board-tilt", type=float, default=0.0, help="Minimum perspective tilt strength for the board plane.")
    parser.add_argument("--max-board-tilt", type=float, default=0.58, help="Maximum perspective tilt strength for the board plane.")
    parser.add_argument("--board-center-jitter", type=float, default=0.16, help="How far the board center may wander from image center, as a frame fraction.")
    parser.add_argument("--board-corner-jitter", type=float, default=0.06, help="Random corner perturbation relative to board size.")
    parser.add_argument("--board-min-visible-fraction", type=float, default=0.92, help="Minimum fraction of the board polygon that must remain in frame.")
    parser.add_argument("--board-min-frame-area-fraction", type=float, default=0.08, help="Minimum visible board area as a fraction of the full frame.")
    parser.add_argument("--green-positive-fraction", type=float, default=0.7, help="Fraction of non-empty images that contain at least one green crab.")
    parser.add_argument("--empty-fraction", type=float, default=0.0, help="Fraction of images with a board but no crabs.")
    parser.add_argument("--background-dir", action="append", default=None, help="Image/video folder to sample as real pool backgrounds. May be repeated.")
    parser.add_argument("--stereo-background-dir", action="append", default=None, help="Stereo image folder to prioritize as a background source. May be repeated.")
    parser.add_argument("--no-stereo-backgrounds", action="store_true", help="Do not automatically prioritize Workspace incoming stereo_sessions folders.")
    parser.add_argument("--no-backgrounds", action="store_true", help="Use only synthetic water-like backgrounds.")
    parser.add_argument("--background-limit", type=int, default=500, help="Maximum discovered background files to consider.")
    parser.add_argument("--camera-blur-fraction", type=float, default=0.12, help="Fraction of images receiving camera blur.")
    parser.add_argument("--max-camera-blur-sigma", type=float, default=0.45, help="Maximum Gaussian blur sigma for camera blur.")
    parser.add_argument("--jpeg-artifact-fraction", type=float, default=0.15, help="Fraction of images receiving JPEG recompression artifacts.")
    parser.add_argument("--min-jpeg-quality", type=int, default=88, help="Minimum JPEG quality when artifacts are applied.")
    parser.add_argument("--green-template", default=None, help="European green crab reference image.")
    parser.add_argument("--rock-template", default=None, help="Native rock crab reference image.")
    parser.add_argument("--jonah-template", default=None, help="Jonah crab reference image.")
    parser.add_argument("--preview-count", type=int, default=12, help="Number of generated images to place in preview.jpg.")
    return parser


def _stereo_background_dirs(workspace_root: Path) -> list[Path]:
    pilot_incoming = workspace_root / "incoming" / "pilot"
    if not pilot_incoming.exists():
        return []
    dirs = [path for path in pilot_incoming.rglob("stereo_sessions") if path.is_dir()]
    return sorted(dirs, key=lambda path: path.stat().st_mtime_ns if path.exists() else 0, reverse=True)


def _reference_paths_from_args(args: argparse.Namespace) -> dict[str, Path]:
    refs = _default_downloads_reference_paths()
    overrides = {
        "european_green_crab": args.green_template,
        "native_rock_crab": args.rock_template,
        "jonah_crab": args.jonah_template,
    }
    for class_name, value in overrides.items():
        if value:
            refs[class_name] = Path(value).expanduser()
    missing = [name for name in CRAB_CLASS_NAMES if name not in refs]
    if missing:
        raise SystemExit(
            "Missing crab reference image(s): "
            + ", ".join(missing)
            + ". Pass --green-template, --rock-template, and --jonah-template."
        )
    return refs


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    workspace = workspace_paths(args.workspace, create=True)
    output = Path(args.output).expanduser() if args.output else fresh_output_subdir(workspace.root / "datasets", "crab_synth", create=True)

    background_dirs = []
    if not args.no_backgrounds:
        if not args.no_stereo_backgrounds:
            background_dirs.extend(Path(value).expanduser() for value in (args.stereo_background_dir or []))
            background_dirs.extend(_stereo_background_dirs(workspace.root))
        if args.background_dir:
            background_dirs.extend(Path(value).expanduser() for value in args.background_dir)
        background_dirs.append(workspace.incoming)
    backgrounds = discover_background_media(background_dirs, limit=max(0, args.background_limit)) if background_dirs else []

    config = SyntheticDatasetConfig(
        output_dir=output,
        reference_paths=_reference_paths_from_args(args),
        background_paths=backgrounds,
        image_count=max(0, args.count),
        image_size=(max(64, args.width), max(64, args.height)),
        val_fraction=max(0.0, min(1.0, args.val_fraction)),
        seed=args.seed,
        min_crabs=max(0, args.min_crabs),
        max_crabs=max(args.min_crabs, args.max_crabs),
        crab_long_edge_range=(
            max(0.01, args.min_crab_scale),
            max(args.min_crab_scale, args.max_crab_scale),
        ),
        large_crab_fraction=max(0.0, min(1.0, args.large_crab_fraction)),
        large_crab_long_edge_range=(
            max(0.01, args.large_crab_min_scale),
            max(args.large_crab_min_scale, args.large_crab_max_scale),
        ),
        sparse_layout_fraction=max(0.0, min(1.0, args.sparse_layout_fraction)),
        full_layout_fraction=max(0.0, min(1.0, args.full_layout_fraction)),
        even_placement_fraction=max(0.0, min(1.0, args.even_placement_fraction)),
        even_placement_jitter=max(0.0, min(0.8, args.even_placement_jitter)),
        max_crab_iou=max(0.0, min(1.0, args.max_crab_overlap)),
        crab_spacing_fraction=max(0.0, min(0.2, args.crab_spacing)),
        min_crab_box_long_edge_px=max(0, args.min_crab_box_long_edge),
        board_long_edge_range=(
            max(0.05, args.min_board_scale),
            max(args.min_board_scale, args.max_board_scale),
        ),
        board_roll_range_degrees=(-abs(args.max_board_roll), abs(args.max_board_roll)),
        board_tilt_range=(
            max(0.0, args.min_board_tilt),
            max(args.min_board_tilt, args.max_board_tilt),
        ),
        board_center_jitter=max(0.0, min(0.49, args.board_center_jitter)),
        board_corner_jitter=max(0.0, min(0.3, args.board_corner_jitter)),
        board_min_visible_fraction=max(0.0, min(1.0, args.board_min_visible_fraction)),
        board_min_frame_area_fraction=max(0.0, min(0.8, args.board_min_frame_area_fraction)),
        green_positive_fraction=max(0.0, min(1.0, args.green_positive_fraction)),
        empty_fraction=max(0.0, min(1.0, args.empty_fraction)),
        max_background_sources=max(0, args.background_limit),
        camera_blur_fraction=max(0.0, min(1.0, args.camera_blur_fraction)),
        camera_blur_sigma_range=(0.05, max(0.05, args.max_camera_blur_sigma)),
        jpeg_artifact_fraction=max(0.0, min(1.0, args.jpeg_artifact_fraction)),
        jpeg_quality_range=(max(1, min(100, args.min_jpeg_quality)), 100),
        preview_count=max(0, args.preview_count),
    )
    result = generate_synthetic_dataset(config)
    print(f"Dataset: {result.output_dir}")
    print(f"YOLO data: {result.data_yaml}")
    print(f"Images: train={result.train_images} val={result.val_images}")
    print("Objects: " + ", ".join(f"{name}={count}" for name, count in result.class_counts.items()))
    if result.preview_image:
        print(f"Preview: {result.preview_image}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
