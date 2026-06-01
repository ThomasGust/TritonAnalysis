"""Batch helper for drawing European green crab boxes on saved images."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis_workspace import fresh_output_subdir, workspace_paths  # noqa: E402
from crab_detector import (  # noqa: E402
    detection_summary_text,
    detect_european_green_crabs,
    draw_european_green_crab_detections,
)


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def collect_image_paths(inputs: list[str | Path], *, recursive: bool = False) -> list[Path]:
    """Collect supported images from files/folders while preserving stable order."""
    paths: list[Path] = []
    seen: set[Path] = set()
    for raw_value in inputs:
        path = Path(raw_value).expanduser()
        if not path.exists():
            continue
        if path.is_dir():
            candidates = sorted(path.rglob("*") if recursive else path.iterdir())
        else:
            candidates = [path]
        for candidate in candidates:
            if not candidate.is_file() or candidate.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            resolved = candidate.resolve()
            if resolved not in seen:
                seen.add(resolved)
                paths.append(resolved)
    return paths


def build_parser() -> argparse.ArgumentParser:
    """Build the batch image detector parser."""
    parser = argparse.ArgumentParser(
        description="Draw European green crab boxes on saved crab-board images.",
    )
    parser.add_argument("paths", nargs="+", help="Image files or folders to process.")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recurse into subfolders. By default, folder inputs process only direct image files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for annotated images and summary.csv. Defaults to Workspace/results/crab_detection.",
    )
    parser.add_argument("--reference-image", default=None, help="Optional crab-board reference image.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run batch image detection."""
    args = build_parser().parse_args(argv)
    image_paths = collect_image_paths(args.paths, recursive=args.recursive)
    if not image_paths:
        print("No supported images found.")
        return 2

    output_dir = args.output_dir or fresh_output_subdir(
        workspace_paths(create=True).crab_results,
        "crab_images",
        create=True,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.csv"

    failures = 0
    with summary_path.open("w", newline="", encoding="utf-8") as summary_file:
        writer = csv.DictWriter(
            summary_file,
            fieldnames=[
                "image",
                "annotated_image",
                "european_green_count",
                "match_count",
                "inlier_count",
                "confidence",
                "boxes_xywh",
                "status",
            ],
        )
        writer.writeheader()
        for image_path in image_paths:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                failures += 1
                writer.writerow(
                    {
                        "image": str(image_path),
                        "annotated_image": "",
                        "european_green_count": "",
                        "match_count": "",
                        "inlier_count": "",
                        "confidence": "",
                        "boxes_xywh": "",
                        "status": "could not read image",
                    }
                )
                continue
            result = detect_european_green_crabs(image, reference_image=args.reference_image)
            annotated = draw_european_green_crab_detections(image, result)
            annotated_path = output_dir / f"{image_path.stem}_european_green_crabs.jpg"
            cv2.imwrite(str(annotated_path), annotated)
            if result is None:
                failures += 1
                writer.writerow(
                    {
                        "image": str(image_path),
                        "annotated_image": str(annotated_path),
                        "european_green_count": 0,
                        "match_count": 0,
                        "inlier_count": 0,
                        "confidence": "0.000",
                        "boxes_xywh": "",
                        "status": "board not matched",
                    }
                )
                print(f"{image_path.name}: {detection_summary_text(result)}")
                continue
            writer.writerow(
                {
                    "image": str(image_path),
                    "annotated_image": str(annotated_path),
                    "european_green_count": result.count,
                    "match_count": result.match_count,
                    "inlier_count": result.inlier_count,
                    "confidence": f"{result.confidence:.3f}",
                    "boxes_xywh": ";".join(
                        f"{detection.bbox[0]}:{detection.bbox[1]}:{detection.bbox[2]}:{detection.bbox[3]}"
                        for detection in result.detections
                    ),
                    "status": "ok",
                }
            )
            print(f"{image_path.name}: {detection_summary_text(result)}")

    print(f"Saved annotated images and summary to {output_dir}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
