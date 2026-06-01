"""Evaluate the crab YOLO detector across archived recording images."""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis_workspace import fresh_output_subdir, workspace_paths  # noqa: E402
from crab_detector import detect_european_green_crabs  # noqa: E402
from tools.crab_yolo_predict import (  # noqa: E402
    DEFAULT_BOARD_CROP_SCALES,
    DEFAULT_CONFIDENCE,
    _boxes_text,
    _crop_from_board_polygon,
    _deduplicate_predictions,
    _draw_predictions,
    _identity_source,
    _import_yolo,
    _inference_region_text,
    _parse_crop_scales,
    _predictions_from_result,
    resolve_model_path,
)
from tools.crab_yolo_train import choose_training_device  # noqa: E402


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
GENERATED_PARTS = {"realityscan", "tritonscan"}
IGNORE_NAME_PARTS = {
    "annotated",
    "mask",
    "overlay",
    "thumb",
    "thumbnail",
    "texture",
    "diffuse",
    "yolo",
}
DERIVED_NAME_PARTS = {"unwrapped"}


@dataclass(frozen=True)
class ArchiveImage:
    path: Path
    relative_path: str
    content_sha1: str


def _archive_default() -> Path:
    return (REPO_ROOT.parent / "TritonPilot" / "recordings_archive").resolve()


def _has_any_part(path: Path, values: set[str]) -> bool:
    normalized = {part.lower() for part in path.parts}
    return any(value.lower() in normalized for value in values)


def _is_ignored_name(path: Path, *, include_derived: bool) -> bool:
    stem = path.stem.lower()
    tokens = set(re.split(r"[^a-z0-9]+", stem))
    if tokens & IGNORE_NAME_PARTS:
        return True
    if not include_derived and tokens & DERIVED_NAME_PARTS:
        return True
    return False


def collect_archive_images(
    archive_root: Path,
    *,
    include_generated: bool,
    include_derived: bool,
    limit: int = 0,
    sample_per_folder: int = 0,
) -> list[ArchiveImage]:
    candidate_paths: list[Path] = []
    for path in sorted(archive_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        relative = path.relative_to(archive_root)
        if not include_generated and _has_any_part(relative, GENERATED_PARTS):
            continue
        if _is_ignored_name(path, include_derived=include_derived):
            continue
        candidate_paths.append(path)

    if sample_per_folder > 0:
        by_folder: dict[Path, list[Path]] = defaultdict(list)
        for path in candidate_paths:
            by_folder[path.parent].append(path)
        candidate_paths = []
        for paths in by_folder.values():
            ordered = sorted(paths)
            if len(ordered) <= sample_per_folder:
                candidate_paths.extend(ordered)
                continue
            if sample_per_folder == 1:
                candidate_paths.append(ordered[len(ordered) // 2])
                continue
            indexes = {
                round(index * (len(ordered) - 1) / (sample_per_folder - 1))
                for index in range(sample_per_folder)
            }
            candidate_paths.extend(ordered[index] for index in sorted(indexes))

    seen_hashes: set[str] = set()
    images: list[ArchiveImage] = []
    for path in sorted(candidate_paths):
        relative = path.relative_to(archive_root)
        digest = hashlib.sha1(path.read_bytes()).hexdigest()
        if digest in seen_hashes:
            continue
        seen_hashes.add(digest)
        images.append(ArchiveImage(path=path, relative_path=str(relative), content_sha1=digest))
        if limit > 0 and len(images) >= limit:
            break
    return images


def _safe_output_stem(index: int, image_path: Path) -> str:
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", image_path.stem).strip("._")
    if not safe_stem:
        safe_stem = "image"
    return f"{index:05d}_{safe_stem}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the production crab YOLO detector over useful archive images.")
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=_archive_default(),
        help="recordings_archive root to scan.",
    )
    parser.add_argument("--model", default=None, help="YOLO weights. Defaults to promoted production weights.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for CSVs and annotated images.")
    parser.add_argument("--device", default=None, help="Prediction device, for example cpu or 0.")
    parser.add_argument("--imgsz", type=int, default=640, help="Prediction image size.")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONFIDENCE, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.25, help="NMS IoU threshold.")
    parser.add_argument(
        "--board-crop-scales",
        default=",".join(f"{scale:.2f}" for scale in DEFAULT_BOARD_CROP_SCALES),
        help="Comma- or space-separated board crop scales for multi-scale inference.",
    )
    parser.add_argument("--reference-image", default=None, help="Optional crab-board reference image.")
    parser.add_argument(
        "--include-generated",
        action="store_true",
        help="Also scan RealityScan/TritonScan generated images.",
    )
    parser.add_argument(
        "--include-derived",
        action="store_true",
        help="Also scan derived images such as unwrapped board crops.",
    )
    parser.add_argument(
        "--include-no-board-yolo",
        action="store_true",
        help="Run YOLO on full images when the reference board is not matched.",
    )
    parser.add_argument(
        "--save-empty-annotations",
        action="store_true",
        help="Save overlays even when the model finds zero green crabs.",
    )
    parser.add_argument(
        "--sample-per-folder",
        type=int,
        default=0,
        help="Evaluate an evenly spaced sample from each image folder instead of every candidate.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional cap on scanned candidate images.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    archive_root = args.archive_root.expanduser().resolve()
    if not archive_root.exists():
        raise SystemExit(f"Archive root does not exist: {archive_root}")

    try:
        crop_scales = _parse_crop_scales(args.board_crop_scales)
    except argparse.ArgumentTypeError as exc:
        raise SystemExit(str(exc)) from exc

    output_dir = args.output_dir or fresh_output_subdir(
        workspace_paths(create=True).results / "crab_yolo",
        "archive_eval",
        create=True,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir = output_dir / "annotated"
    annotated_dir.mkdir(parents=True, exist_ok=True)
    inventory_path = output_dir / "inventory.csv"
    summary_path = output_dir / "summary.csv"

    images = collect_archive_images(
        archive_root,
        include_generated=args.include_generated,
        include_derived=args.include_derived,
        limit=args.limit,
        sample_per_folder=args.sample_per_folder,
    )

    model_path = resolve_model_path(args.model)
    device = choose_training_device(args.device)
    YOLO = _import_yolo()
    model = YOLO(str(model_path))

    print(f"Archive: {archive_root}")
    print(f"Model: {model_path}")
    print(f"Candidate images after filtering and exact dedupe: {len(images)}")
    print(f"Device: {device}")
    print(f"Output: {output_dir}")

    inventory_fields = [
        "index",
        "image",
        "relative_path",
        "sha1",
        "width",
        "height",
        "status",
        "board_matched",
        "board_inliers",
        "board_matches",
        "european_green_count",
        "boxes",
        "annotated_image",
        "inference_region",
    ]
    summary_fields = inventory_fields

    evaluated = 0
    board_matched = 0
    positives = 0
    with inventory_path.open("w", encoding="utf-8", newline="") as inventory_file, summary_path.open(
        "w", encoding="utf-8", newline=""
    ) as summary_file:
        inventory_writer = csv.DictWriter(inventory_file, fieldnames=inventory_fields)
        summary_writer = csv.DictWriter(summary_file, fieldnames=summary_fields)
        inventory_writer.writeheader()
        summary_writer.writeheader()

        for index, archive_image in enumerate(images, start=1):
            image = cv2.imread(str(archive_image.path), cv2.IMREAD_COLOR)
            row = {
                "index": index,
                "image": str(archive_image.path),
                "relative_path": archive_image.relative_path,
                "sha1": archive_image.content_sha1,
                "width": "",
                "height": "",
                "status": "ok",
                "board_matched": "0",
                "board_inliers": "",
                "board_matches": "",
                "european_green_count": "",
                "boxes": "",
                "annotated_image": "",
                "inference_region": "",
            }
            if image is None:
                row["status"] = "could not read image"
                inventory_writer.writerow(row)
                continue

            height, width = image.shape[:2]
            row["width"] = width
            row["height"] = height
            if min(width, height) < 256:
                row["status"] = "skipped small image"
                inventory_writer.writerow(row)
                continue

            board_result = detect_european_green_crabs(image, reference_image=args.reference_image)
            if board_result is None:
                if not args.include_no_board_yolo:
                    row["status"] = "skipped no board match"
                    inventory_writer.writerow(row)
                    continue
                sources = [_identity_source(image, "full image: board not matched")]
            else:
                row["board_matched"] = "1"
                row["board_inliers"] = board_result.inlier_count
                row["board_matches"] = board_result.match_count
                board_matched += 1
                sources = [
                    _crop_from_board_polygon(
                        image,
                        board_result.board_polygon,
                        image_size=args.imgsz,
                        crop_scale=crop_scale,
                    )
                    for crop_scale in crop_scales
                ]

            predictions = []
            for source in sources:
                results = model.predict(
                    source=source.image,
                    imgsz=args.imgsz,
                    conf=args.conf,
                    iou=args.iou,
                    device=device,
                    verbose=False,
                )
                if results:
                    predictions.extend(_predictions_from_result(results[0], source.inverse_matrix, image.shape))
            predictions = _deduplicate_predictions(predictions)
            count = len(predictions)
            evaluated += 1
            positives += int(count > 0)

            row["european_green_count"] = count
            row["boxes"] = _boxes_text(predictions)
            row["inference_region"] = _inference_region_text(sources)
            if count > 0 or args.save_empty_annotations:
                annotated_path = annotated_dir / f"{_safe_output_stem(index, archive_image.path)}_yolo_green_crabs.jpg"
                cv2.imwrite(str(annotated_path), _draw_predictions(image, predictions))
                row["annotated_image"] = str(annotated_path)

            inventory_writer.writerow(row)
            summary_writer.writerow(row)
            if evaluated % 25 == 0:
                print(
                    f"Evaluated {evaluated} images "
                    f"({board_matched} board-matched, {positives} with detections)..."
                )

    print(f"Board-matched/evaluated images: {evaluated}")
    print(f"Images with one or more detections: {positives}")
    print(f"Inventory: {inventory_path}")
    print(f"Summary: {summary_path}")
    print(f"Annotated positives: {annotated_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
