"""Generate synthetic YOLO fine-tuning data for European green crab detection.

The generator uses the fixed-board crab detector as a label seed, then applies
geometric and underwater-style photometric augmentation while transforming the
labels with the image. The output is a one-class YOLO dataset:

    images/train, labels/train, images/val, labels/val, data.yaml
"""

from __future__ import annotations

import argparse
import csv
import random
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


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_RECORDINGS_DIR = REPO_ROOT.parent / "TritonPilot" / "recordings"
CLASS_NAME = "european_green_crab"


@dataclass(frozen=True)
class SourceExample:
    """A source image and detector-generated labels used as augmentation seed."""

    path: Path
    image: np.ndarray
    crab_polygons: tuple[np.ndarray, ...]
    board_polygon: np.ndarray
    match_count: int
    inlier_count: int
    confidence: float


@dataclass(frozen=True)
class GeneratedSample:
    """One generated image and its YOLO labels."""

    image: np.ndarray
    labels: tuple[tuple[int, float, float, float, float], ...]
    source: SourceExample


def collect_image_paths(inputs: list[str | Path], *, recursive: bool = False) -> list[Path]:
    """Collect supported image paths from files/folders while preserving order."""
    paths: list[Path] = []
    seen: set[Path] = set()
    for raw_value in inputs:
        path = Path(raw_value).expanduser()
        if not path.exists():
            continue
        candidates = sorted(path.rglob("*") if recursive and path.is_dir() else path.iterdir()) if path.is_dir() else [path]
        for candidate in candidates:
            if not candidate.is_file() or candidate.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            paths.append(resolved)
    return paths


def _as_polygon_array(points: tuple[tuple[float, float], ...] | np.ndarray) -> np.ndarray:
    return np.asarray(points, dtype=np.float32).reshape(-1, 2)


def _transform_polygons_affine(polygons: tuple[np.ndarray, ...], matrix: np.ndarray) -> tuple[np.ndarray, ...]:
    transformed: list[np.ndarray] = []
    for polygon in polygons:
        pts = cv2.transform(polygon.reshape(1, -1, 2), matrix).reshape(-1, 2)
        transformed.append(pts.astype(np.float32))
    return tuple(transformed)


def _transform_polygons_perspective(polygons: tuple[np.ndarray, ...], matrix: np.ndarray) -> tuple[np.ndarray, ...]:
    transformed: list[np.ndarray] = []
    for polygon in polygons:
        pts = cv2.perspectiveTransform(polygon.reshape(-1, 1, 2), matrix).reshape(-1, 2)
        transformed.append(pts.astype(np.float32))
    return tuple(transformed)


def _bbox(points: np.ndarray) -> tuple[float, float, float, float]:
    return (
        float(np.min(points[:, 0])),
        float(np.min(points[:, 1])),
        float(np.max(points[:, 0])),
        float(np.max(points[:, 1])),
    )


def _border_value(image: np.ndarray) -> tuple[int, int, int]:
    """Estimate a neutral fill color from image edges for geometric warps."""
    height, width = image.shape[:2]
    band = max(2, min(height, width) // 32)
    samples = np.concatenate(
        [
            image[:band, :, :].reshape(-1, 3),
            image[-band:, :, :].reshape(-1, 3),
            image[:, :band, :].reshape(-1, 3),
            image[:, -band:, :].reshape(-1, 3),
        ],
        axis=0,
    )
    color = np.median(samples, axis=0)
    return tuple(int(np.clip(value, 0, 255)) for value in color)


def _crop_to_square(
    image: np.ndarray,
    polygons: tuple[np.ndarray, ...],
    board_polygon: np.ndarray,
    *,
    image_size: int,
    rng: random.Random,
    split: str,
) -> tuple[np.ndarray, tuple[np.ndarray, ...], np.ndarray]:
    """Crop around the board with jitter and scale, returning a square canvas."""
    board_x0, board_y0, board_x1, board_y1 = _bbox(board_polygon)
    board_w = max(1.0, board_x1 - board_x0)
    board_h = max(1.0, board_y1 - board_y0)
    base_side = max(board_w, board_h)
    center_x = (board_x0 + board_x1) * 0.5
    center_y = (board_y0 + board_y1) * 0.5

    if split == "train":
        if rng.random() < 0.34:
            side = base_side * rng.uniform(1.65, 4.10)
            jitter = base_side * rng.uniform(0.0, 0.55)
        else:
            side = base_side * rng.uniform(0.78, 1.80)
            jitter = base_side * rng.uniform(0.0, 0.24)
    else:
        if rng.random() < 0.22:
            side = base_side * rng.uniform(1.55, 3.20)
            jitter = base_side * rng.uniform(0.0, 0.32)
        else:
            side = base_side * rng.uniform(1.02, 1.42)
            jitter = base_side * rng.uniform(0.0, 0.09)
    center_x += rng.uniform(-jitter, jitter)
    center_y += rng.uniform(-jitter, jitter)

    scale = image_size / side
    matrix = np.array(
        [
            [scale, 0.0, image_size * 0.5 - center_x * scale],
            [0.0, scale, image_size * 0.5 - center_y * scale],
        ],
        dtype=np.float32,
    )
    warped = cv2.warpAffine(
        image,
        matrix,
        (image_size, image_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=_border_value(image),
    )
    return warped, _transform_polygons_affine(polygons, matrix), cv2.transform(
        board_polygon.reshape(1, -1, 2), matrix
    ).reshape(-1, 2)


def _apply_canvas_geometry(
    image: np.ndarray,
    polygons: tuple[np.ndarray, ...],
    *,
    rng: random.Random,
    split: str,
    hard_perspective: bool,
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    """Apply rotation, scale, translation, and mild perspective distortion."""
    height, width = image.shape[:2]
    center = (width * 0.5, height * 0.5)
    if hard_perspective:
        angle = rng.uniform(-82.0, 82.0)
        scale = rng.uniform(0.66, 1.42)
        translate = width * rng.uniform(0.03, 0.18)
        corner_jitter = width * rng.uniform(0.045, 0.130)
    elif split == "train":
        angle = rng.uniform(-65.0, 65.0)
        scale = rng.uniform(0.72, 1.34)
        translate = width * rng.uniform(0.0, 0.13)
        corner_jitter = width * rng.uniform(0.015, 0.105)
    else:
        angle = rng.uniform(-22.0, 22.0)
        scale = rng.uniform(0.90, 1.12)
        translate = width * rng.uniform(0.0, 0.045)
        corner_jitter = width * rng.uniform(0.0, 0.035)

    affine = cv2.getRotationMatrix2D(center, angle, scale).astype(np.float32)
    affine[0, 2] += rng.uniform(-translate, translate)
    affine[1, 2] += rng.uniform(-translate, translate)
    image = cv2.warpAffine(
        image,
        affine,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=_border_value(image),
    )
    polygons = _transform_polygons_affine(polygons, affine)

    src = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    dst = src + np.array(
        [
            [rng.uniform(-corner_jitter, corner_jitter), rng.uniform(-corner_jitter, corner_jitter)],
            [rng.uniform(-corner_jitter, corner_jitter), rng.uniform(-corner_jitter, corner_jitter)],
            [rng.uniform(-corner_jitter, corner_jitter), rng.uniform(-corner_jitter, corner_jitter)],
            [rng.uniform(-corner_jitter, corner_jitter), rng.uniform(-corner_jitter, corner_jitter)],
        ],
        dtype=np.float32,
    )
    if hard_perspective:
        mode = rng.choice(["left", "right", "top", "bottom"])
        x_pull = width * rng.uniform(0.08, 0.24)
        y_squeeze = height * rng.uniform(0.12, 0.32)
        y_pull = height * rng.uniform(0.08, 0.24)
        x_squeeze = width * rng.uniform(0.12, 0.32)
        if mode == "left":
            dst[0] += (x_pull, y_squeeze)
            dst[3] += (x_pull, -y_squeeze)
            dst[1] += (width * rng.uniform(-0.03, 0.06), height * rng.uniform(-0.05, 0.05))
            dst[2] += (width * rng.uniform(-0.03, 0.06), height * rng.uniform(-0.05, 0.05))
        elif mode == "right":
            dst[1] += (-x_pull, y_squeeze)
            dst[2] += (-x_pull, -y_squeeze)
            dst[0] += (width * rng.uniform(-0.06, 0.03), height * rng.uniform(-0.05, 0.05))
            dst[3] += (width * rng.uniform(-0.06, 0.03), height * rng.uniform(-0.05, 0.05))
        elif mode == "top":
            dst[0] += (x_squeeze, y_pull)
            dst[1] += (-x_squeeze, y_pull)
            dst[2] += (width * rng.uniform(-0.04, 0.04), height * rng.uniform(-0.04, 0.06))
            dst[3] += (width * rng.uniform(-0.04, 0.04), height * rng.uniform(-0.04, 0.06))
        else:
            dst[3] += (x_squeeze, -y_pull)
            dst[2] += (-x_squeeze, -y_pull)
            dst[0] += (width * rng.uniform(-0.04, 0.04), height * rng.uniform(-0.06, 0.04))
            dst[1] += (width * rng.uniform(-0.04, 0.04), height * rng.uniform(-0.06, 0.04))
    perspective = cv2.getPerspectiveTransform(src, dst)
    image = cv2.warpPerspective(
        image,
        perspective,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=_border_value(image),
    )
    return image, _transform_polygons_perspective(polygons, perspective)


def _blurred_noise_mask(height: int, width: int, rng: random.Random, *, scale: int = 64) -> np.ndarray:
    small_h = max(2, height // scale)
    small_w = max(2, width // scale)
    noise = np.random.default_rng(rng.randrange(2**32)).random((small_h, small_w), dtype=np.float32)
    mask = cv2.resize(noise, (width, height), interpolation=cv2.INTER_CUBIC)
    return cv2.GaussianBlur(mask, (0, 0), sigmaX=max(3.0, width / 32.0))


def _add_caustics(image: np.ndarray, rng: random.Random) -> np.ndarray:
    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=np.float32)
    for _ in range(rng.randint(4, 12)):
        y0 = rng.uniform(0, height)
        amplitude = rng.uniform(height * 0.01, height * 0.08)
        phase = rng.uniform(0.0, np.pi * 2.0)
        frequency = rng.uniform(1.0, 4.0)
        points = []
        for x in np.linspace(-width * 0.05, width * 1.05, 18):
            y = y0 + np.sin((x / width) * np.pi * 2.0 * frequency + phase) * amplitude
            points.append([int(round(x)), int(round(y))])
        cv2.polylines(mask, [np.asarray(points, dtype=np.int32)], False, 1.0, rng.randint(1, 4), cv2.LINE_AA)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=rng.uniform(3.0, 10.0))
    alpha = rng.uniform(0.04, 0.18)
    lifted = image.astype(np.float32) + (mask[..., None] * 255.0 * alpha)
    return np.clip(lifted, 0, 255).astype(np.uint8)


def _add_occlusions(image: np.ndarray, rng: random.Random) -> np.ndarray:
    height, width = image.shape[:2]
    overlay = image.copy()
    for _ in range(rng.randint(0, 3)):
        color_choices = [(30, 45, 45), (20, 80, 110), (120, 150, 120), (55, 55, 85)]
        color = color_choices[rng.randrange(len(color_choices))]
        alpha = rng.uniform(0.22, 0.58)
        if rng.random() < 0.55:
            rect_w = int(width * rng.uniform(0.04, 0.18))
            rect_h = int(height * rng.uniform(0.015, 0.09))
            center = (rng.randint(0, width - 1), rng.randint(0, height - 1))
            angle = rng.uniform(-75, 75)
            box = cv2.boxPoints((center, (rect_w, rect_h), angle)).astype(np.int32)
            cv2.fillConvexPoly(overlay, box, color, cv2.LINE_AA)
        else:
            axes = (
                int(width * rng.uniform(0.025, 0.09)),
                int(height * rng.uniform(0.02, 0.075)),
            )
            center = (rng.randint(0, width - 1), rng.randint(0, height - 1))
            cv2.ellipse(overlay, center, axes, rng.uniform(0, 180), 0, 360, color, -1, cv2.LINE_AA)
        image = cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0)
    return image


def _apply_photometric_effects(image: np.ndarray, *, rng: random.Random, split: str) -> np.ndarray:
    """Apply illumination, water, blur, noise, and compression variation."""
    img = image.astype(np.float32)
    if split == "train":
        contrast = rng.uniform(0.66, 1.48)
        brightness = rng.uniform(-42.0, 42.0)
        gains = np.array(
            [rng.uniform(0.88, 1.26), rng.uniform(0.84, 1.20), rng.uniform(0.60, 1.14)],
            dtype=np.float32,
        )
        haze_alpha = rng.uniform(0.0, 0.34)
    else:
        contrast = rng.uniform(0.86, 1.18)
        brightness = rng.uniform(-18.0, 18.0)
        gains = np.array(
            [rng.uniform(0.94, 1.12), rng.uniform(0.92, 1.10), rng.uniform(0.82, 1.06)],
            dtype=np.float32,
        )
        haze_alpha = rng.uniform(0.0, 0.14)

    img = (img - 127.5) * contrast + 127.5 + brightness
    img *= gains.reshape(1, 1, 3)

    if haze_alpha > 0.01:
        height, width = img.shape[:2]
        haze_color = np.array(
            [rng.uniform(95, 165), rng.uniform(120, 190), rng.uniform(80, 150)],
            dtype=np.float32,
        )
        mask = _blurred_noise_mask(height, width, rng, scale=44)[..., None]
        img = img * (1.0 - haze_alpha * mask) + haze_color.reshape(1, 1, 3) * (haze_alpha * mask)

    img = np.clip(img, 0, 255).astype(np.uint8)

    if split == "train" and rng.random() < 0.48:
        img = _add_caustics(img, rng)
    if split == "train" and rng.random() < 0.42:
        img = _add_occlusions(img, rng)
    if rng.random() < (0.46 if split == "train" else 0.20):
        sigma = rng.uniform(0.45, 1.55 if split == "train" else 0.9)
        img = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)
    if split == "train" and rng.random() < 0.18:
        kernel_size = rng.choice([3, 5, 7])
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        if rng.random() < 0.5:
            kernel[kernel_size // 2, :] = 1.0
        else:
            kernel[:, kernel_size // 2] = 1.0
        kernel /= np.sum(kernel)
        img = cv2.filter2D(img, -1, kernel)
    if rng.random() < (0.76 if split == "train" else 0.28):
        sigma = rng.uniform(2.0, 15.0 if split == "train" else 6.0)
        noise = np.random.default_rng(rng.randrange(2**32)).normal(0.0, sigma, img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if rng.random() < (0.78 if split == "train" else 0.35):
        quality = rng.randint(48, 95) if split == "train" else rng.randint(76, 96)
        ok, encoded = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if ok:
            decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if decoded is not None:
                img = decoded
    return img


def _labels_from_polygons(
    polygons: tuple[np.ndarray, ...],
    *,
    image_size: int,
    split: str,
) -> tuple[tuple[int, float, float, float, float], ...]:
    labels: list[tuple[int, float, float, float, float]] = []
    minimum_visible = 0.16 if split == "train" else 0.45
    minimum_size = max(8.0, image_size * (0.018 if split == "train" else 0.025))
    for polygon in polygons:
        if not np.all(np.isfinite(polygon)):
            continue
        x0, y0, x1, y1 = _bbox(polygon)
        full_w = max(1.0, x1 - x0)
        full_h = max(1.0, y1 - y0)
        clipped_x0 = max(0.0, min(float(image_size), x0))
        clipped_y0 = max(0.0, min(float(image_size), y0))
        clipped_x1 = max(0.0, min(float(image_size), x1))
        clipped_y1 = max(0.0, min(float(image_size), y1))
        clipped_w = clipped_x1 - clipped_x0
        clipped_h = clipped_y1 - clipped_y0
        visible = (clipped_w * clipped_h) / max(1.0, full_w * full_h)
        if visible < minimum_visible or clipped_w < minimum_size or clipped_h < minimum_size:
            continue
        x_center = (clipped_x0 + clipped_x1) * 0.5 / image_size
        y_center = (clipped_y0 + clipped_y1) * 0.5 / image_size
        width = clipped_w / image_size
        height = clipped_h / image_size
        labels.append((0, x_center, y_center, width, height))
    return tuple(labels)


def _generate_sample(
    source: SourceExample,
    *,
    image_size: int,
    rng: random.Random,
    split: str,
    hard_perspective_ratio: float,
) -> GeneratedSample | None:
    image, polygons, _board = _crop_to_square(
        source.image,
        source.crab_polygons,
        source.board_polygon,
        image_size=image_size,
        rng=rng,
        split=split,
    )
    image, polygons = _apply_canvas_geometry(
        image,
        polygons,
        rng=rng,
        split=split,
        hard_perspective=rng.random() < hard_perspective_ratio,
    )
    labels = _labels_from_polygons(polygons, image_size=image_size, split=split)
    if not labels:
        return None
    image = _apply_photometric_effects(image, rng=rng, split=split)
    return GeneratedSample(image=image, labels=labels, source=source)


def _write_yolo_labels(path: Path, labels: tuple[tuple[int, float, float, float, float], ...]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as label_file:
        for class_id, x_center, y_center, width, height in labels:
            label_file.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


def _draw_labels(image: np.ndarray, labels: tuple[tuple[int, float, float, float, float], ...]) -> np.ndarray:
    annotated = image.copy()
    height, width = annotated.shape[:2]
    for _class_id, x_center, y_center, box_w, box_h in labels:
        x0 = int(round((x_center - box_w * 0.5) * width))
        y0 = int(round((y_center - box_h * 0.5) * height))
        x1 = int(round((x_center + box_w * 0.5) * width))
        y1 = int(round((y_center + box_h * 0.5) * height))
        cv2.rectangle(annotated, (x0, y0), (x1, y1), (0, 190, 255), 2, cv2.LINE_AA)
    return annotated


def _write_preview(output_dir: Path, generated: list[tuple[Path, tuple[tuple[int, float, float, float, float], ...]]]) -> None:
    if not generated:
        return
    thumbs: list[np.ndarray] = []
    thumb_size = 192
    for image_path, labels in generated:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        image = _draw_labels(image, labels)
        thumbs.append(cv2.resize(image, (thumb_size, thumb_size), interpolation=cv2.INTER_AREA))
    if not thumbs:
        return
    columns = min(6, len(thumbs))
    rows = int(np.ceil(len(thumbs) / columns))
    sheet = np.full((rows * thumb_size, columns * thumb_size, 3), 242, dtype=np.uint8)
    for index, thumb in enumerate(thumbs):
        row = index // columns
        col = index % columns
        sheet[row * thumb_size : (row + 1) * thumb_size, col * thumb_size : (col + 1) * thumb_size] = thumb
    metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(metadata_dir / "preview.jpg"), sheet)


def _load_sources(
    image_paths: list[Path],
    *,
    reference_image: str | None,
    required_count: int,
) -> tuple[list[SourceExample], list[dict[str, str]]]:
    sources: list[SourceExample] = []
    rows: list[dict[str, str]] = []
    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            rows.append(
                {
                    "image": str(image_path),
                    "status": "could not read image",
                    "count": "",
                    "match_count": "",
                    "inlier_count": "",
                    "confidence": "",
                }
            )
            continue
        result = detect_european_green_crabs(image, reference_image=reference_image)
        if result is None:
            rows.append(
                {
                    "image": str(image_path),
                    "status": "board not matched",
                    "count": "0",
                    "match_count": "0",
                    "inlier_count": "0",
                    "confidence": "0.000",
                }
            )
            continue
        status = "ok"
        if required_count > 0 and result.count != required_count:
            status = f"skipped: expected {required_count}"
        rows.append(
            {
                "image": str(image_path),
                "status": status,
                "count": str(result.count),
                "match_count": str(result.match_count),
                "inlier_count": str(result.inlier_count),
                "confidence": f"{result.confidence:.3f}",
            }
        )
        if status != "ok":
            continue
        sources.append(
            SourceExample(
                path=image_path,
                image=image,
                crab_polygons=tuple(_as_polygon_array(detection.polygon) for detection in result.detections),
                board_polygon=_as_polygon_array(result.board_polygon),
                match_count=result.match_count,
                inlier_count=result.inlier_count,
                confidence=result.confidence,
            )
        )
    return sources, rows


def _write_dataset_metadata(
    output_dir: Path,
    *,
    source_rows: list[dict[str, str]],
    generated_rows: list[dict[str, str]],
    args: argparse.Namespace,
) -> None:
    metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    with (metadata_dir / "source_labels.csv").open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["image", "status", "count", "match_count", "inlier_count", "confidence"],
        )
        writer.writeheader()
        writer.writerows(source_rows)
    with (metadata_dir / "generated_samples.csv").open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["split", "image", "label", "source_image", "label_count"],
        )
        writer.writeheader()
        writer.writerows(generated_rows)
    with (metadata_dir / "README.txt").open("w", encoding="utf-8", newline="\n") as readme:
        readme.write(
            "\n".join(
                [
                    "Synthetic European green crab YOLO dataset",
                    "",
                    f"train_count: {args.train_count}",
                    f"val_count: {args.val_count}",
                    f"test_count: {args.test_count}",
                    f"image_size: {args.image_size}",
                    f"seed: {args.seed}",
                    f"required_source_count: {args.required_count}",
                    f"hard_perspective_ratio: {args.hard_perspective_ratio}",
                    f"val_hard_perspective_ratio: {args.val_hard_perspective_ratio}",
                    "",
                    "Labels are one-class YOLO xywh normalized boxes.",
                    "Class 0: european_green_crab",
                    "",
                ]
            )
        )


def _write_data_yaml(output_dir: Path, *, include_test: bool) -> None:
    dataset_root = output_dir.resolve().as_posix()
    lines = [
        f"path: {dataset_root}",
        "train: images/train",
        "val: images/val",
    ]
    if include_test:
        lines.append("test: images/test")
    lines.extend(["names:", f"  0: {CLASS_NAME}", ""])
    (output_dir / "data.yaml").write_text("\n".join(lines), encoding="utf-8")
    (output_dir / "classes.txt").write_text(f"{CLASS_NAME}\n", encoding="utf-8")


def _make_output_dirs(output_dir: Path, splits: list[str]) -> None:
    for split in splits:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    (output_dir / "metadata").mkdir(parents=True, exist_ok=True)


def _generate_split(
    *,
    split: str,
    count: int,
    output_dir: Path,
    sources: list[SourceExample],
    image_size: int,
    rng: random.Random,
    max_retries: int,
    preview_remaining: int,
    hard_perspective_ratio: float,
) -> tuple[list[dict[str, str]], list[tuple[Path, tuple[tuple[int, float, float, float, float], ...]]]]:
    rows: list[dict[str, str]] = []
    preview: list[tuple[Path, tuple[tuple[int, float, float, float, float], ...]]] = []
    failures = 0
    for index in range(count):
        sample: GeneratedSample | None = None
        for _attempt in range(max_retries):
            source = sources[rng.randrange(len(sources))]
            sample = _generate_sample(
                source,
                image_size=image_size,
                rng=rng,
                split=split,
                hard_perspective_ratio=hard_perspective_ratio,
            )
            if sample is not None:
                break
        if sample is None:
            failures += 1
            continue

        stem = f"crab_{split}_{index + 1:06d}"
        image_path = output_dir / "images" / split / f"{stem}.jpg"
        label_path = output_dir / "labels" / split / f"{stem}.txt"
        cv2.imwrite(str(image_path), sample.image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        _write_yolo_labels(label_path, sample.labels)
        rows.append(
            {
                "split": split,
                "image": str(image_path),
                "label": str(label_path),
                "source_image": str(sample.source.path),
                "label_count": str(len(sample.labels)),
            }
        )
        if len(preview) < preview_remaining:
            preview.append((image_path, sample.labels))

    if failures:
        print(f"{split}: skipped {failures} samples after retry limits")
    return rows, preview


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate synthetic one-class YOLO fine-tuning data for European green crab detection.",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help=(
            "Source image files or folders. Defaults to TritonPilot/recordings if available. "
            "Folder inputs are non-recursive unless --recursive is set."
        ),
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recurse into source folders. By default, only direct image files are used.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Dataset output directory. Defaults to Workspace/datasets/crab_green_yolo_<timestamp>.",
    )
    parser.add_argument("--reference-image", default=None, help="Optional crab-board reference image.")
    parser.add_argument("--train-count", type=int, default=700, help="Number of synthetic training images.")
    parser.add_argument("--val-count", type=int, default=180, help="Number of synthetic validation images.")
    parser.add_argument("--test-count", type=int, default=0, help="Optional number of synthetic test images.")
    parser.add_argument("--image-size", type=int, default=640, help="Square image size to generate.")
    parser.add_argument(
        "--required-count",
        type=int,
        default=4,
        help="Only seed from source images with this many green crabs. Use 0 to accept any matched count.",
    )
    parser.add_argument("--seed", type=int, default=20260531, help="Random seed for reproducible generation.")
    parser.add_argument("--max-retries", type=int, default=60, help="Retry limit per generated image.")
    parser.add_argument("--preview-count", type=int, default=30, help="Number of labeled samples in metadata/preview.jpg.")
    parser.add_argument(
        "--hard-perspective-ratio",
        type=float,
        default=0.40,
        help="Training-sample probability for strong perspective warps that mimic steep board angles.",
    )
    parser.add_argument(
        "--val-hard-perspective-ratio",
        type=float,
        default=0.22,
        help="Validation/test-sample probability for strong perspective warps.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.image_size < 128:
        print("--image-size must be at least 128.")
        return 2
    if args.train_count < 0 or args.val_count < 0 or args.test_count < 0:
        print("Split counts cannot be negative.")
        return 2
    if not (0.0 <= args.hard_perspective_ratio <= 1.0 and 0.0 <= args.val_hard_perspective_ratio <= 1.0):
        print("Perspective ratios must be between 0 and 1.")
        return 2

    input_values = args.paths or ([DEFAULT_RECORDINGS_DIR] if DEFAULT_RECORDINGS_DIR.exists() else [])
    image_paths = collect_image_paths(input_values, recursive=args.recursive)
    if not image_paths:
        print("No supported source images found.")
        return 2

    print(f"Scanning {len(image_paths)} source images...")
    sources, source_rows = _load_sources(
        image_paths,
        reference_image=args.reference_image,
        required_count=args.required_count,
    )
    if not sources:
        print("No usable source images found after crab-board labeling.")
        return 1
    print(f"Using {len(sources)} labeled source images.")

    workspace = workspace_paths(create=True)
    output_dir = args.output_dir or fresh_output_subdir(workspace.root / "datasets", "crab_green_yolo", create=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = ["train", "val"] + (["test"] if args.test_count else [])
    _make_output_dirs(output_dir, splits)

    rng = random.Random(args.seed)
    generated_rows: list[dict[str, str]] = []
    preview_samples: list[tuple[Path, tuple[tuple[int, float, float, float, float], ...]]] = []
    split_counts = {"train": args.train_count, "val": args.val_count, "test": args.test_count}
    for split in splits:
        hard_ratio = args.hard_perspective_ratio if split == "train" else args.val_hard_perspective_ratio
        rows, preview = _generate_split(
            split=split,
            count=split_counts[split],
            output_dir=output_dir,
            sources=sources,
            image_size=args.image_size,
            rng=rng,
            max_retries=args.max_retries,
            preview_remaining=max(0, args.preview_count - len(preview_samples)),
            hard_perspective_ratio=hard_ratio,
        )
        generated_rows.extend(rows)
        preview_samples.extend(preview)
        print(f"{split}: generated {len(rows)} images")

    _write_data_yaml(output_dir, include_test=bool(args.test_count))
    _write_dataset_metadata(output_dir, source_rows=source_rows, generated_rows=generated_rows, args=args)
    _write_preview(output_dir, preview_samples[: args.preview_count])

    print(f"Saved YOLO dataset to {output_dir}")
    print(f"Use data file: {output_dir / 'data.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
