"""Build OpenAI-stage reference data from saved crab-counter runs.

This turns past ``crab_counter`` run artifacts into in-domain reference data for all three
counter stages, written into the dedicated reference root (``data/crab/references`` by default)
that only the counter reads -- never the synthetic dataset generator:

* ``classification/<class>/`` -- one-crab crops bucketed by predicted species (atlas examples).
* ``detector/``               -- class-agnostic single-crab crops ("one box, one crab" unit).
* ``board/``                  -- board-footprint crops taken from the full-res source frames.

Crops are named with their confidence so the strongest examples sort first; curate by deleting
the weak ones. Re-running overwrites same-frame crops rather than multiplying them.

Examples
--------
    python tools/crab_extract_references.py
    python tools/crab_extract_references.py --min-confidence 0.6 --max-per-class 8 --clear
    python tools/crab_extract_references.py --from /path/to/results/crab_counter --accepted-only
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from triton_analysis.crab.synthetic import CRAB_CLASS_NAMES, IMAGE_EXTENSIONS
from triton_analysis.workspace import workspace_paths


CLASSIFICATION_DIR = "classification"
DETECTOR_DIR = "detector"
BOARD_DIR = "board"
DEFAULT_CROP_PAD_FRACTION = 0.08
DEFAULT_BOARD_MARGIN_FRACTION = 0.06


@dataclass
class ExtractionSummary:
    """Tally of what the extraction wrote."""

    runs: int = 0
    classification: dict[str, int] = field(default_factory=lambda: {name: 0 for name in CRAB_CLASS_NAMES})
    detector: int = 0
    board: int = 0
    skipped_low_confidence: int = 0

    def as_lines(self) -> list[str]:
        lines = [f"Processed {self.runs} run result file(s)."]
        for class_name in CRAB_CLASS_NAMES:
            lines.append(f"  classification/{class_name}: {self.classification[class_name]}")
        lines.append(f"  detector: {self.detector}")
        lines.append(f"  board: {self.board}")
        if self.skipped_low_confidence:
            lines.append(f"  skipped (below --min-confidence): {self.skipped_low_confidence}")
        return lines


def _read_image(path: Path) -> np.ndarray | None:
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise OSError(f"could not encode image: {path}")
    encoded.tofile(str(path))


def _frame_tag(image_path: Path) -> str:
    stem = image_path.stem
    for suffix in ("_auto_homography", "_manual_homography", "_manual_crop"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    cleaned = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in stem)
    return cleaned.strip("_") or "frame"


def _clamp_box(box: Sequence[float], width: int, height: int) -> tuple[int, int, int, int] | None:
    x0, y0, x1, y1 = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    ix0 = max(0, int(round(x0)))
    iy0 = max(0, int(round(y0)))
    ix1 = min(width, int(round(x1)))
    iy1 = min(height, int(round(y1)))
    if ix1 - ix0 < 4 or iy1 - iy0 < 4:
        return None
    return ix0, iy0, ix1, iy1


def _padded_crop(image: np.ndarray, box: Sequence[float], pad_fraction: float) -> np.ndarray | None:
    height, width = image.shape[:2]
    clamped = _clamp_box(box, width, height)
    if clamped is None:
        return None
    x0, y0, x1, y1 = clamped
    pad = int(round(max(x1 - x0, y1 - y0) * max(0.0, pad_fraction)))
    padded = _clamp_box((x0 - pad, y0 - pad, x1 + pad, y1 + pad), width, height)
    if padded is None:
        return None
    px0, py0, px1, py1 = padded
    crop = image[py0:py1, px0:px1].copy()
    return crop if crop.size else None


def _conf_tag(confidence: float) -> str:
    return f"conf{int(round(max(0.0, min(1.0, confidence)) * 100)):03d}"


def _iter_result_files(runs_root: Path) -> Iterable[Path]:
    if not runs_root.exists():
        return []
    seen: set[Path] = set()
    ordered: list[Path] = []
    for path in sorted(runs_root.rglob("*_crab_count.json")):
        # Skip ensemble/benchmark aggregates that just duplicate a per-image run.
        if path.name in {"ensemble_final_crab_count.json"}:
            continue
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        ordered.append(path)
    return ordered


def _find_preprocess_metadata(processed_image: Path) -> dict | None:
    folder = processed_image.parent
    for candidate in sorted(folder.glob("*_preprocess.json")):
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if Path(str(payload.get("processed_image", ""))).name == processed_image.name:
            return payload
        # Single-frame folders hold exactly one preprocess record.
        return payload
    return None


def _extract_board_crop(
    processed_image: Path,
    dest_root: Path,
    margin_fraction: float,
    summary: ExtractionSummary,
) -> None:
    metadata = _find_preprocess_metadata(processed_image)
    if not metadata:
        return
    points = metadata.get("ordered_points") or metadata.get("detected_points")
    source_image_path = metadata.get("source_image")
    if not points or len(points) != 4 or not source_image_path:
        return
    source_path = Path(str(source_image_path)).expanduser()
    source = _read_image(source_path)
    if source is None:
        return
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    box = (min(xs), min(ys), max(xs), max(ys))
    crop = _padded_crop(source, box, margin_fraction)
    if crop is None:
        return
    confidence = float(metadata.get("auto_board_confidence") or metadata.get("board_outline", {}).get("confidence") or 0.0)
    name = f"{_conf_tag(confidence)}_board_{_frame_tag(source_path)}.png"
    _write_image(dest_root / BOARD_DIR / name, crop)
    summary.board += 1


def extract_from_result_file(
    result_json: Path,
    dest_root: Path,
    *,
    min_confidence: float,
    accepted_only: bool,
    crop_pad_fraction: float,
    board_margin_fraction: float,
    summary: ExtractionSummary,
) -> None:
    try:
        payload = json.loads(result_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    processed_image = Path(str(payload.get("image_path", ""))).expanduser()
    image = _read_image(processed_image)
    if image is None:
        return
    summary.runs += 1
    frame_tag = _frame_tag(processed_image)

    for index, candidate in enumerate(payload.get("candidates", []), start=1):
        if not isinstance(candidate, dict):
            continue
        bbox = candidate.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        label = str(candidate.get("label") or "").strip()
        class_scores = candidate.get("class_scores") or {}
        species_conf = float(class_scores.get(label, candidate.get("confidence", 0.0)) or 0.0)
        accepted = bool(candidate.get("accepted_as_target", False))
        if species_conf < min_confidence:
            summary.skipped_low_confidence += 1
            continue
        crop = _padded_crop(image, bbox, crop_pad_fraction)
        if crop is None:
            continue
        # Confidence-first names so a plain listing (and the atlas/montage) surface the best examples.
        stem = f"{_conf_tag(species_conf)}_{frame_tag}_c{index:02d}"

        # Detector gallery: every single-crab box, class-agnostic (label kept in the name for review).
        detector_label = label or "crab"
        _write_image(dest_root / DETECTOR_DIR / f"{_conf_tag(species_conf)}_{detector_label}_{frame_tag}_c{index:02d}.png", crop)
        summary.detector += 1

        # Classification atlas: only real species buckets. With --accepted-only we keep just the
        # green-crab crops the pipeline actually accepted, but always keep hard-negative (rock/jonah)
        # crops since those are exactly the examples the atlas needs.
        if label in CRAB_CLASS_NAMES:
            drop_unaccepted_target = accepted_only and label == "european_green_crab" and not accepted
            if not drop_unaccepted_target:
                _write_image(dest_root / CLASSIFICATION_DIR / label / f"{stem}.png", crop)
                summary.classification[label] += 1

    _extract_board_crop(processed_image, dest_root, board_margin_fraction, summary)


def _clear_dest(dest_root: Path) -> None:
    import shutil

    for sub in (CLASSIFICATION_DIR, DETECTOR_DIR, BOARD_DIR):
        target = dest_root / sub
        if target.exists():
            shutil.rmtree(target)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workspace", default=None, help="TritonAnalysis workspace root (for the default --from).")
    parser.add_argument(
        "--from",
        dest="runs_root",
        default=None,
        help="Folder of saved crab_counter runs to scan. Defaults to <workspace>/results/crab_counter.",
    )
    parser.add_argument(
        "--dest",
        default=None,
        help="Reference output root. Defaults to the repo data/crab/references the counter reads.",
    )
    parser.add_argument("--min-confidence", type=float, default=0.0, help="Drop crops whose species confidence is below this.")
    parser.add_argument("--max-per-class", type=int, default=0, help="Keep at most N classification crops per class (0 = all).")
    parser.add_argument("--accepted-only", action="store_true", help="Only keep accepted European green crab classification crops.")
    parser.add_argument("--crop-pad", type=float, default=DEFAULT_CROP_PAD_FRACTION, help="Padding fraction around each crab crop.")
    parser.add_argument("--board-margin", type=float, default=DEFAULT_BOARD_MARGIN_FRACTION, help="Margin fraction around board crops.")
    parser.add_argument("--clear", action="store_true", help="Delete existing classification/detector/board crops before writing.")
    return parser


def _resolve_runs_root(args: argparse.Namespace) -> Path:
    if args.runs_root:
        return Path(args.runs_root).expanduser()
    workspace = workspace_paths(args.workspace, create=False)
    return workspace.results / "crab_counter"


def _resolve_dest_root(args: argparse.Namespace) -> Path:
    if args.dest:
        return Path(args.dest).expanduser()
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / "data" / "crab" / "references"


def _prune_per_class(dest_root: Path, max_per_class: int) -> None:
    if max_per_class <= 0:
        return
    for class_name in CRAB_CLASS_NAMES:
        folder = dest_root / CLASSIFICATION_DIR / class_name
        if not folder.is_dir():
            continue
        crops = sorted(
            (path for path in folder.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS),
            key=lambda path: path.name,
        )
        # Names sort by confidence tag descending? conf is zero-padded ascending, so reverse for best-first.
        crops.sort(key=lambda path: path.name, reverse=True)
        for path in crops[max_per_class:]:
            path.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runs_root = _resolve_runs_root(args)
    dest_root = _resolve_dest_root(args)
    if not runs_root.exists():
        print(f"No runs found at {runs_root}", file=sys.stderr)
        return 1
    if args.clear:
        _clear_dest(dest_root)

    summary = ExtractionSummary()
    for result_json in _iter_result_files(runs_root):
        extract_from_result_file(
            result_json,
            dest_root,
            min_confidence=args.min_confidence,
            accepted_only=args.accepted_only,
            crop_pad_fraction=args.crop_pad,
            board_margin_fraction=args.board_margin,
            summary=summary,
        )
    _prune_per_class(dest_root, args.max_per_class)

    print(f"Reference data written under {dest_root}")
    for line in summary.as_lines():
        print(line)
    if summary.runs == 0:
        print("No usable result files were found.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
