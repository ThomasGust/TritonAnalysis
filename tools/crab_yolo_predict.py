"""Run a trained YOLO European green crab detector on saved images."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis_workspace import fresh_output_subdir, workspace_paths  # noqa: E402
from crab_detector import detect_european_green_crabs  # noqa: E402
from tools.crab_image_detect import collect_image_paths  # noqa: E402
from tools.crab_yolo_train import choose_training_device  # noqa: E402


@dataclass(frozen=True)
class YoloPrediction:
    class_id: int
    confidence: float
    box_xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class YoloSource:
    image: np.ndarray
    inverse_matrix: np.ndarray
    inference_region: str


DEFAULT_BOARD_CROP_SCALES = (1.15, 1.55, 2.05)
DEFAULT_CONFIDENCE = 0.20


def _import_yolo():
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Ultralytics is required for YOLO prediction. Install it with:\n"
            "  python -m pip install ultralytics\n"
            "or reinstall TritonAnalysis requirements."
        ) from exc
    return YOLO


def latest_trained_weights() -> Path | None:
    """Return promoted crab YOLO weights, falling back to the newest run."""
    model_root = workspace_paths().root / "models" / "crab_yolo"
    promoted = model_root / "production" / "weights" / "best.pt"
    if promoted.exists():
        return promoted.resolve()

    best_candidates = sorted(
        model_root.glob("**/weights/best.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if best_candidates:
        return best_candidates[0]
    last_candidates = sorted(
        model_root.glob("**/weights/last.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return last_candidates[0] if last_candidates else None


def resolve_model_path(value: str | Path | None) -> Path:
    if value is None:
        latest = latest_trained_weights()
        if latest is None:
            raise SystemExit("No model provided and no Workspace/models/crab_yolo/**/weights/*.pt was found.")
        return latest.resolve()
    path = Path(value).expanduser()
    if not path.exists():
        raise SystemExit(f"Could not find YOLO model weights: {path}")
    return path.resolve()


def _iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return 0.0 if union <= 0.0 else inter / union


def _intersection_over_smaller(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
) -> float:
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    smaller = min(area_a, area_b)
    return 0.0 if smaller <= 0.0 else inter / smaller


def _deduplicate_predictions(
    predictions: list[YoloPrediction],
    *,
    overlap_iou: float = 0.35,
    containment: float = 0.72,
) -> list[YoloPrediction]:
    kept: list[YoloPrediction] = []
    for prediction in sorted(predictions, key=lambda item: item.confidence, reverse=True):
        if any(
            _iou(prediction.box_xyxy, kept_prediction.box_xyxy) >= overlap_iou
            or _intersection_over_smaller(prediction.box_xyxy, kept_prediction.box_xyxy) >= containment
            for kept_prediction in kept
        ):
            continue
        kept.append(prediction)
    return kept


def _boxes_text(predictions: list[YoloPrediction]) -> str:
    values: list[str] = []
    for prediction in predictions:
        x0, y0, x1, y1 = prediction.box_xyxy
        values.append(
            f"{prediction.class_id}:{prediction.confidence:.3f}:"
            f"{int(round(x0))}:{int(round(y0))}:{int(round(x1))}:{int(round(y1))}"
        )
    return ";".join(values)


def _draw_predictions(image: np.ndarray, predictions: list[YoloPrediction]) -> np.ndarray:
    annotated = image.copy()
    height, width = annotated.shape[:2]
    line_width = max(2, int(round(min(height, width) / 700.0)))
    badge_font_scale = 0.48
    count_font_scale = 0.82
    box_color = (0, 180, 255)
    text_dark = (20, 24, 28)
    for index, prediction in enumerate(predictions, start=1):
        x0, y0, x1, y1 = (int(round(value)) for value in prediction.box_xyxy)
        cv2.rectangle(annotated, (x0, y0), (x1, y1), box_color, line_width, cv2.LINE_AA)
        label = str(index)
        (label_width, label_height), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            badge_font_scale,
            1,
        )
        badge_x0 = max(0, min(x0, width - label_width - 10))
        badge_y0 = max(0, y0 - label_height - baseline - 7)
        if badge_y0 <= 2:
            badge_y0 = min(max(0, y0 + 4), max(0, height - label_height - baseline - 7))
        badge_x1 = min(width - 1, badge_x0 + label_width + 10)
        badge_y1 = min(height - 1, badge_y0 + label_height + baseline + 7)
        cv2.rectangle(annotated, (badge_x0, badge_y0), (badge_x1, badge_y1), box_color, -1, cv2.LINE_AA)
        cv2.putText(
            annotated,
            label,
            (badge_x0 + 5, badge_y0 + label_height + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            badge_font_scale,
            text_dark,
            1,
            cv2.LINE_AA,
        )
    count_label = f"European green crabs: {len(predictions)}"
    cv2.putText(annotated, count_label, (24, 42), cv2.FONT_HERSHEY_SIMPLEX, count_font_scale, text_dark, 4, cv2.LINE_AA)
    cv2.putText(annotated, count_label, (24, 42), cv2.FONT_HERSHEY_SIMPLEX, count_font_scale, box_color, 2, cv2.LINE_AA)
    return annotated


def _identity_source(image: np.ndarray, inference_region: str) -> YoloSource:
    identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    return YoloSource(image=image, inverse_matrix=identity, inference_region=inference_region)


def _crop_from_board_polygon(
    image: np.ndarray,
    board_polygon,
    *,
    image_size: int,
    crop_scale: float,
) -> YoloSource:
    board = np.asarray(board_polygon, dtype=np.float32).reshape(-1, 2)
    x0 = float(np.min(board[:, 0]))
    y0 = float(np.min(board[:, 1]))
    x1 = float(np.max(board[:, 0]))
    y1 = float(np.max(board[:, 1]))
    side = max(1.0, x1 - x0, y1 - y0) * max(1.0, crop_scale)
    center_x = (x0 + x1) * 0.5
    center_y = (y0 + y1) * 0.5
    scale = image_size / side
    matrix = np.array(
        [
            [scale, 0.0, image_size * 0.5 - center_x * scale],
            [0.0, scale, image_size * 0.5 - center_y * scale],
        ],
        dtype=np.float32,
    )
    edge_color = tuple(int(value) for value in np.median(image.reshape(-1, 3), axis=0))
    crop = cv2.warpAffine(
        image,
        matrix,
        (image_size, image_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=edge_color,
    )
    return YoloSource(
        image=crop,
        inverse_matrix=cv2.invertAffineTransform(matrix),
        inference_region=f"board crop x{crop_scale:.2f}",
    )


def _crop_around_board(
    image: np.ndarray,
    *,
    reference_image: str | None,
    image_size: int,
    crop_scale: float,
) -> tuple[np.ndarray, np.ndarray, str]:
    result = detect_european_green_crabs(image, reference_image=reference_image)
    if result is None:
        source = _identity_source(image, "full image: board not matched")
        return source.image, source.inverse_matrix, source.inference_region

    source = _crop_from_board_polygon(image, result.board_polygon, image_size=image_size, crop_scale=crop_scale)
    return source.image, source.inverse_matrix, source.inference_region


def _parse_crop_scales(value: str | None, *, single_scale: float | None = None) -> tuple[float, ...]:
    raw_values: list[float] = []
    if single_scale is not None:
        raw_values = [single_scale]
    elif value:
        try:
            raw_values = [float(part) for part in value.replace(",", " ").split()]
        except ValueError as exc:
            raise argparse.ArgumentTypeError("--board-crop-scales must be comma- or space-separated numbers.") from exc
    else:
        raw_values = list(DEFAULT_BOARD_CROP_SCALES)

    scales: list[float] = []
    for scale in raw_values:
        if scale < 1.0:
            raise argparse.ArgumentTypeError("Board crop scales must be at least 1.0.")
        rounded = round(scale, 4)
        if rounded not in scales:
            scales.append(rounded)
    if not scales:
        raise argparse.ArgumentTypeError("At least one board crop scale is required.")
    return tuple(scales)


def _board_crop_sources(
    image: np.ndarray,
    *,
    reference_image: str | None,
    image_size: int,
    crop_scales: tuple[float, ...],
) -> list[YoloSource]:
    result = detect_european_green_crabs(image, reference_image=reference_image)
    if result is None:
        return [_identity_source(image, "full image: board not matched")]
    return [
        _crop_from_board_polygon(image, result.board_polygon, image_size=image_size, crop_scale=crop_scale)
        for crop_scale in crop_scales
    ]


def _inference_region_text(sources: list[YoloSource]) -> str:
    if not sources:
        return ""
    if len(sources) == 1:
        return sources[0].inference_region
    return "; ".join(source.inference_region for source in sources)


def _predictions_from_result(result, inverse_matrix: np.ndarray, original_shape: tuple[int, int, int]) -> list[YoloPrediction]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []
    original_height, original_width = original_shape[:2]
    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)
    predictions: list[YoloPrediction] = []
    for class_id, confidence, box in zip(cls, conf, xyxy):
        x0, y0, x1, y1 = box
        corners = np.array([[[x0, y0], [x1, y0], [x1, y1], [x0, y1]]], dtype=np.float32)
        mapped = cv2.transform(corners, inverse_matrix).reshape(-1, 2)
        mapped_x0 = float(np.clip(np.min(mapped[:, 0]), 0, max(0, original_width - 1)))
        mapped_y0 = float(np.clip(np.min(mapped[:, 1]), 0, max(0, original_height - 1)))
        mapped_x1 = float(np.clip(np.max(mapped[:, 0]), 0, max(0, original_width - 1)))
        mapped_y1 = float(np.clip(np.max(mapped[:, 1]), 0, max(0, original_height - 1)))
        if mapped_x1 <= mapped_x0 or mapped_y1 <= mapped_y0:
            continue
        predictions.append(
            YoloPrediction(
                class_id=int(class_id),
                confidence=float(confidence),
                box_xyxy=(mapped_x0, mapped_y0, mapped_x1, mapped_y1),
            )
        )
    return predictions


def _predict_from_sources(
    model,
    image: np.ndarray,
    sources: list[YoloSource],
    *,
    imgsz: int,
    conf: float,
    iou: float,
    device: str,
) -> list[YoloPrediction]:
    predictions: list[YoloPrediction] = []
    for source in sources:
        results = model.predict(
            source=source.image,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device,
            verbose=False,
        )
        if not results:
            continue
        predictions.extend(_predictions_from_result(results[0], source.inverse_matrix, image.shape))
    return _deduplicate_predictions(predictions)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run trained YOLO green-crab detection on still images.")
    parser.add_argument("paths", nargs="+", help="Image files or folders to process.")
    parser.add_argument(
        "--model",
        default=None,
        help="Path to trained weights. Defaults to the latest Workspace/models/crab_yolo/**/weights/best.pt.",
    )
    parser.add_argument("--recursive", action="store_true", help="Recurse into source folders.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory for annotated images and CSV.")
    parser.add_argument("--imgsz", type=int, default=640, help="Prediction image size.")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONFIDENCE, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.25, help="NMS IoU threshold.")
    parser.add_argument("--device", default=None, help="Prediction device, for example cpu or 0.")
    parser.add_argument(
        "--full-image",
        action="store_true",
        help="Run YOLO on the full image instead of an automatic reference-board crop.",
    )
    parser.add_argument("--reference-image", default=None, help="Optional crab-board reference image for board crops.")
    parser.add_argument(
        "--board-crop-scales",
        default=",".join(f"{scale:.2f}" for scale in DEFAULT_BOARD_CROP_SCALES),
        help="Comma- or space-separated board crop scales for multi-scale inference.",
    )
    parser.add_argument(
        "--board-crop-scale",
        type=float,
        default=None,
        help="Legacy single board crop scale. When set, overrides --board-crop-scales.",
    )
    parser.add_argument(
        "--max-detections",
        type=int,
        default=0,
        help="Keep at most this many highest-confidence crab boxes after de-duplication. Defaults to 0 for no cap.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        crop_scales = _parse_crop_scales(args.board_crop_scales, single_scale=args.board_crop_scale)
    except argparse.ArgumentTypeError as exc:
        raise SystemExit(str(exc)) from exc
    model_path = resolve_model_path(args.model)
    image_paths = collect_image_paths(args.paths, recursive=args.recursive)
    if not image_paths:
        print("No supported images found.")
        return 2

    output_dir = args.output_dir or fresh_output_subdir(
        workspace_paths(create=True).results / "crab_yolo",
        "predictions",
        create=True,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.csv"
    device = choose_training_device(args.device)

    print(f"Model: {model_path}")
    print(f"Images: {len(image_paths)}")
    print(f"Device: {device}")
    print(f"Output: {output_dir}")

    YOLO = _import_yolo()
    model = YOLO(str(model_path))
    with summary_path.open("w", encoding="utf-8", newline="") as summary_file:
        writer = csv.DictWriter(
            summary_file,
            fieldnames=["image", "annotated_image", "european_green_count", "boxes", "status", "inference_region"],
        )
        writer.writeheader()
        for image_path in image_paths:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                writer.writerow(
                    {
                        "image": str(image_path),
                        "annotated_image": "",
                        "european_green_count": "",
                        "boxes": "",
                        "status": "could not read image",
                        "inference_region": "",
                    }
                )
                continue
            if args.full_image:
                sources = [_identity_source(image, "full image")]
            else:
                sources = _board_crop_sources(
                    image,
                    reference_image=args.reference_image,
                    image_size=args.imgsz,
                    crop_scales=crop_scales,
                )
            inference_region = _inference_region_text(sources)
            predictions = _predict_from_sources(
                model,
                image,
                sources,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                device=device,
            )
            if args.max_detections > 0:
                predictions = sorted(predictions, key=lambda item: item.confidence, reverse=True)[: args.max_detections]
            annotated = _draw_predictions(image, predictions)
            annotated_path = output_dir / f"{image_path.stem}_yolo_green_crabs.jpg"
            cv2.imwrite(str(annotated_path), annotated)
            writer.writerow(
                {
                    "image": str(image_path),
                    "annotated_image": str(annotated_path),
                    "european_green_count": len(predictions),
                    "boxes": _boxes_text(predictions),
                    "status": "ok",
                    "inference_region": inference_region,
                }
            )
            print(f"{image_path.name}: {len(predictions)} European green crab detections ({inference_region})")

    print(f"Saved predictions to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
