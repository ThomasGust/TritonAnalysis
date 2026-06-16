"""Synthetic YOLO dataset generation for the crab detection task."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import cv2
import numpy as np


CRAB_CLASS_NAMES: tuple[str, ...] = (
    "european_green_crab",
    "native_rock_crab",
    "jonah_crab",
)

LAYOUT_PROFILE_NAMES: tuple[str, ...] = ("sparse", "normal", "full")

IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4"}


@dataclass(frozen=True)
class YoloBox:
    """One generated object box in absolute pixel coordinates."""

    class_id: int
    class_name: str
    xyxy: tuple[float, float, float, float]

    def to_yolo_line(self, width: int, height: int) -> str:
        x0, y0, x1, y1 = self.xyxy
        cx = ((x0 + x1) * 0.5) / max(1, width)
        cy = ((y0 + y1) * 0.5) / max(1, height)
        bw = (x1 - x0) / max(1, width)
        bh = (y1 - y0) / max(1, height)
        return f"{self.class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


@dataclass
class CrabTemplate:
    """A crab source image with an extracted foreground mask."""

    class_id: int
    class_name: str
    bgr: np.ndarray
    mask: np.ndarray


@dataclass
class SyntheticDatasetConfig:
    """Configuration for a synthetic crab YOLO dataset."""

    output_dir: Path
    reference_paths: Mapping[str, Path]
    background_paths: Sequence[Path] = field(default_factory=tuple)
    image_count: int = 2000
    image_size: tuple[int, int] = (1280, 720)
    val_fraction: float = 0.2
    seed: int = 7
    min_crabs: int = 4
    max_crabs: int = 11
    crab_long_edge_range: tuple[float, float] = (0.10, 0.42)
    large_crab_fraction: float = 0.32
    large_crab_long_edge_range: tuple[float, float] = (0.34, 0.52)
    sparse_layout_fraction: float = 0.25
    full_layout_fraction: float = 0.3
    even_placement_fraction: float = 0.55
    even_placement_jitter: float = 0.36
    max_crab_iou: float = 0.025
    crab_spacing_fraction: float = 0.012
    min_crab_box_long_edge_px: int = 40
    board_long_edge_range: tuple[float, float] = (0.58, 1.18)
    board_roll_range_degrees: tuple[float, float] = (-50.0, 50.0)
    board_tilt_range: tuple[float, float] = (0.0, 0.58)
    board_center_jitter: float = 0.16
    board_corner_jitter: float = 0.06
    board_min_visible_fraction: float = 0.92
    board_min_frame_area_fraction: float = 0.08
    green_positive_fraction: float = 0.7
    empty_fraction: float = 0.0
    real_background_fraction: float = 0.75
    max_background_sources: int = 500
    camera_blur_fraction: float = 0.12
    camera_blur_sigma_range: tuple[float, float] = (0.05, 0.45)
    jpeg_artifact_fraction: float = 0.15
    jpeg_quality_range: tuple[int, int] = (88, 100)
    preview_count: int = 12


@dataclass(frozen=True)
class SyntheticDatasetResult:
    """Summary of a generated dataset."""

    output_dir: Path
    data_yaml: Path
    preview_image: Path | None
    train_images: int
    val_images: int
    class_counts: dict[str, int]


def discover_background_media(
    roots: Sequence[str | Path],
    *,
    limit: int | None = 500,
) -> list[Path]:
    """Find image and video files that can act as real pool backgrounds."""
    per_root: list[list[Path]] = []
    seen_candidates: set[Path] = set()
    for root_value in roots:
        candidates_for_root: list[Path] = []
        root = Path(root_value).expanduser()
        if not root.exists():
            continue
        candidates = [root] if root.is_file() else root.rglob("*")
        for path in candidates:
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS:
                continue
            try:
                key = path.resolve()
            except OSError:
                key = path
            if key in seen_candidates:
                continue
            seen_candidates.add(key)
            candidates_for_root.append(path)
        if candidates_for_root:
            per_root.append(candidates_for_root)

    paths: list[Path] = []
    seen: set[Path] = set()
    index = 0
    while any(index < len(candidates) for candidates in per_root):
        for candidates in per_root:
            if index >= len(candidates):
                continue
            path = candidates[index]
            try:
                key = path.resolve()
            except OSError:
                key = path
            if key in seen:
                continue
            seen.add(key)
            paths.append(path)
            if limit is not None and len(paths) >= limit:
                return paths
        index += 1
    return paths


def generate_synthetic_dataset(config: SyntheticDatasetConfig) -> SyntheticDatasetResult:
    """Generate a YOLO-format synthetic crab dataset."""
    rng = np.random.default_rng(int(config.seed))
    output_dir = Path(config.output_dir).expanduser()
    templates = _load_templates(config.reference_paths)
    backgrounds = list(config.background_paths)[: max(0, int(config.max_background_sources))]

    split_dirs = _prepare_dataset_dirs(output_dir)
    class_counts = {name: 0 for name in CRAB_CLASS_NAMES}
    train_images = 0
    val_images = 0
    preview_items: list[tuple[Path, list[YoloBox]]] = []

    image_count = max(0, int(config.image_count))
    for index in range(image_count):
        image_bgr, boxes = _render_scene(templates, backgrounds, config, rng)
        split = "val" if rng.random() < float(config.val_fraction) else "train"
        if split == "val":
            val_images += 1
        else:
            train_images += 1

        stem = f"crab_synth_{index:06d}"
        image_path = split_dirs[split]["images"] / f"{stem}.jpg"
        label_path = split_dirs[split]["labels"] / f"{stem}.txt"
        _write_jpeg(image_path, image_bgr, quality=92)
        label_path.write_text(
            "\n".join(box.to_yolo_line(config.image_size[0], config.image_size[1]) for box in boxes),
            encoding="utf-8",
        )

        for box in boxes:
            class_counts[box.class_name] += 1
        if len(preview_items) < max(0, int(config.preview_count)):
            preview_items.append((image_path, boxes))

    data_yaml = output_dir / "data.yaml"
    _write_data_yaml(data_yaml, output_dir)
    (output_dir / "classes.txt").write_text("\n".join(CRAB_CLASS_NAMES) + "\n", encoding="utf-8")
    _write_manifest(output_dir, config, class_counts, train_images, val_images)
    preview_image = _write_preview(output_dir, preview_items, config.image_size)

    return SyntheticDatasetResult(
        output_dir=output_dir,
        data_yaml=data_yaml,
        preview_image=preview_image,
        train_images=train_images,
        val_images=val_images,
        class_counts=class_counts,
    )


def _prepare_dataset_dirs(output_dir: Path) -> dict[str, dict[str, Path]]:
    split_dirs: dict[str, dict[str, Path]] = {}
    for split in ("train", "val"):
        image_dir = output_dir / "images" / split
        label_dir = output_dir / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        split_dirs[split] = {"images": image_dir, "labels": label_dir}
    return split_dirs


def _load_templates(reference_paths: Mapping[str, Path]) -> list[CrabTemplate]:
    templates: list[CrabTemplate] = []
    for class_id, class_name in enumerate(CRAB_CLASS_NAMES):
        if class_name not in reference_paths:
            raise ValueError(f"missing reference image for {class_name}")
        path = Path(reference_paths[class_name]).expanduser()
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(path)
        mask = _extract_crab_mask(image)
        x, y, w, h = cv2.boundingRect(mask)
        pad = max(3, int(max(w, h) * 0.04))
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(image.shape[1], x + w + pad)
        y1 = min(image.shape[0], y + h + pad)
        crop = image[y0:y1, x0:x1].copy()
        crop_mask = mask[y0:y1, x0:x1].copy()
        templates.append(CrabTemplate(class_id, class_name, crop, crop_mask))
    return templates


def _extract_crab_mask(image_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    pale_background = (saturation < 38) & (value > 213)
    mask = np.where(pale_background, 0, 255).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)

    num, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, 8)
    cleaned = np.zeros_like(mask)
    min_area = max(12, int(image_bgr.shape[0] * image_bgr.shape[1] * 0.0008))
    for idx in range(1, num):
        if int(stats[idx, cv2.CC_STAT_AREA]) >= min_area:
            cleaned[labels == idx] = 255
    if np.count_nonzero(cleaned) == 0:
        raise ValueError("could not extract foreground mask from crab template")
    return cleaned


def _render_scene(
    templates: Sequence[CrabTemplate],
    backgrounds: Sequence[Path],
    config: SyntheticDatasetConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[YoloBox]]:
    min_long_edge = max(0.0, float(config.min_crab_box_long_edge_px))
    best_scene: np.ndarray | None = None
    best_boxes: list[YoloBox] = []
    best_min_edge = -1.0

    for _attempt in range(1):
        scene, boxes, expected_count = _render_scene_once(templates, backgrounds, config, rng)
        if expected_count == 0:
            return scene, boxes

        min_box_edge = min((_box_long_edge(box.xyxy) for box in boxes), default=0.0)
        if len(boxes) == expected_count and min_box_edge >= min_long_edge:
            return scene, boxes

        if min_box_edge > best_min_edge:
            best_scene = scene
            best_boxes = boxes
            best_min_edge = min_box_edge

    if best_scene is not None:
        return best_scene, best_boxes
    return _render_scene_once(templates, backgrounds, config, rng)[:2]


def _render_scene_once(
    templates: Sequence[CrabTemplate],
    backgrounds: Sequence[Path],
    config: SyntheticDatasetConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[YoloBox], int]:
    width, height = config.image_size
    background = _make_background(backgrounds, width, height, config, rng)
    board_size = int(max(720, min(1400, max(width, height))))
    board = _make_corrugated_board(board_size, rng)
    object_masks: list[tuple[int, str, np.ndarray]] = []

    if rng.random() >= float(config.empty_fraction):
        object_masks = _place_crabs_on_board(board, templates, config, rng)
    _add_laminate_and_tape(board, rng)
    board_boxes = _board_boxes_from_object_masks(object_masks, board_size)

    source = np.array(
        [[0, 0], [board_size - 1, 0], [board_size - 1, board_size - 1], [0, board_size - 1]],
        dtype=np.float32,
    )
    min_long_edge = max(0.0, float(config.min_crab_box_long_edge_px))
    expected_count = len(object_masks)
    transform: np.ndarray | None = None
    boxes: list[YoloBox] = []
    best_transform: np.ndarray | None = None
    best_boxes: list[YoloBox] = []
    best_min_edge = -1.0

    for _quad_attempt in range(20):
        board_quad = _sample_board_quad(width, height, config, rng)
        candidate_transform = cv2.getPerspectiveTransform(source, board_quad.astype(np.float32))
        candidate_boxes = _project_board_boxes(board_boxes, candidate_transform, width, height)
        min_box_edge = min((_box_long_edge(box.xyxy) for box in candidate_boxes), default=0.0)
        if expected_count == 0 or (len(candidate_boxes) == expected_count and min_box_edge >= min_long_edge):
            transform = candidate_transform
            break
        if min_box_edge > best_min_edge:
            best_transform = candidate_transform
            best_boxes = _project_board_boxes(board_boxes, candidate_transform, width, height)
            best_min_edge = min_box_edge

    if transform is None:
        transform = best_transform
    if transform is None:
        board_quad = _sample_board_quad(width, height, config, rng)
        transform = cv2.getPerspectiveTransform(source, board_quad.astype(np.float32))
    boxes = _boxes_from_object_masks(object_masks, transform, width, height)
    if boxes and min(_box_long_edge(box.xyxy) for box in boxes) < min_long_edge and best_boxes:
        if best_transform is not None:
            transform = best_transform
        boxes = _boxes_from_object_masks(object_masks, transform, width, height)

    warped_board = cv2.warpPerspective(board, transform, (width, height), flags=cv2.INTER_LINEAR)
    board_alpha_src = np.full((board_size, board_size), 255, dtype=np.uint8)
    board_alpha = cv2.warpPerspective(board_alpha_src, transform, (width, height), flags=cv2.INTER_LINEAR)
    scene = _alpha_blend(background, warped_board, board_alpha)

    scene = _apply_underwater_camera_effects(scene, config, rng)
    return scene, boxes, expected_count


def _board_boxes_from_object_masks(
    object_masks: Sequence[tuple[int, str, np.ndarray]],
    board_size: int,
) -> list[tuple[int, str, tuple[float, float, float, float]]]:
    boxes: list[tuple[int, str, tuple[float, float, float, float]]] = []
    for class_id, class_name, mask in object_masks:
        box = _mask_to_box(mask, board_size, board_size)
        if box is not None:
            boxes.append((class_id, class_name, box))
    return boxes


def _project_board_boxes(
    board_boxes: Sequence[tuple[int, str, tuple[float, float, float, float]]],
    transform: np.ndarray,
    width: int,
    height: int,
) -> list[YoloBox]:
    boxes: list[YoloBox] = []
    for class_id, class_name, box in board_boxes:
        x0, y0, x1, y1 = box
        corners = np.array(
            [[[x0, y0], [x1, y0], [x1, y1], [x0, y1]]],
            dtype=np.float32,
        )
        projected = cv2.perspectiveTransform(corners, transform)[0]
        px0 = float(np.clip(projected[:, 0].min(), 0, width - 1))
        py0 = float(np.clip(projected[:, 1].min(), 0, height - 1))
        px1 = float(np.clip(projected[:, 0].max(), 0, width - 1))
        py1 = float(np.clip(projected[:, 1].max(), 0, height - 1))
        if px1 <= px0 or py1 <= py0:
            continue
        boxes.append(YoloBox(class_id=class_id, class_name=class_name, xyxy=(px0, py0, px1, py1)))
    return boxes


def _boxes_from_object_masks(
    object_masks: Sequence[tuple[int, str, np.ndarray]],
    transform: np.ndarray,
    width: int,
    height: int,
) -> list[YoloBox]:
    boxes: list[YoloBox] = []
    for class_id, class_name, mask in object_masks:
        warped_mask = cv2.warpPerspective(mask, transform, (width, height), flags=cv2.INTER_NEAREST)
        box = _mask_to_box(warped_mask, width, height)
        if box is None:
            continue
        boxes.append(YoloBox(class_id=class_id, class_name=class_name, xyxy=box))
    return boxes


def _make_background(
    backgrounds: Sequence[Path],
    width: int,
    height: int,
    config: SyntheticDatasetConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    if backgrounds and rng.random() < float(config.real_background_fraction):
        for _attempt in range(8):
            path = backgrounds[int(rng.integers(0, len(backgrounds)))]
            image = _read_media_frame(path, rng)
            if image is not None:
                return _resize_fill(image, width, height, rng)
    return _make_pool_background(width, height, rng)


def _read_media_frame(path: Path, rng: np.random.Generator) -> np.ndarray | None:
    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        try:
            data = np.fromfile(str(path), dtype=np.uint8)
        except OSError:
            return None
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    if suffix in VIDEO_EXTENSIONS:
        capture = cv2.VideoCapture(str(path))
        try:
            if not capture.isOpened():
                return None
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if frame_count > 1:
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(rng.integers(0, frame_count)))
            ok, frame = capture.read()
            return frame if ok else None
        finally:
            capture.release()
    return None


def _resize_fill(image: np.ndarray, width: int, height: int, rng: np.random.Generator) -> np.ndarray:
    src_h, src_w = image.shape[:2]
    scale = max(width / max(1, src_w), height / max(1, src_h))
    resized = cv2.resize(
        image,
        (max(width, int(round(src_w * scale))), max(height, int(round(src_h * scale)))),
        interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
    )
    y_max = max(0, resized.shape[0] - height)
    x_max = max(0, resized.shape[1] - width)
    y = int(rng.integers(0, y_max + 1)) if y_max else 0
    x = int(rng.integers(0, x_max + 1)) if x_max else 0
    return resized[y : y + height, x : x + width].copy()


def _make_pool_background(width: int, height: int, rng: np.random.Generator) -> np.ndarray:
    top = np.array(
        [
            rng.uniform(95, 145),
            rng.uniform(125, 180),
            rng.uniform(105, 155),
        ],
        dtype=np.float32,
    )
    bottom = np.array(
        [
            rng.uniform(55, 110),
            rng.uniform(95, 155),
            rng.uniform(95, 145),
        ],
        dtype=np.float32,
    )
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None, None]
    base = top * (1.0 - y) + bottom * y
    image = np.repeat(base, width, axis=1)
    noise = rng.normal(0.0, 7.0, image.shape).astype(np.float32)
    image += noise

    for _ in range(int(rng.integers(2, 6))):
        cx = rng.uniform(0, width)
        cy = rng.uniform(0, height)
        radius = rng.uniform(width * 0.18, width * 0.55)
        yy, xx = np.mgrid[0:height, 0:width]
        blob = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / max(1.0, radius**2)))
        image += blob[:, :, None] * rng.uniform(-18, 20)
    return np.clip(image, 0, 255).astype(np.uint8)


def _make_corrugated_board(size: int, rng: np.random.Generator) -> np.ndarray:
    base_color = np.array(
        [rng.uniform(225, 250), rng.uniform(228, 252), rng.uniform(226, 250)],
        dtype=np.float32,
    )
    board = np.full((size, size, 3), base_color, dtype=np.float32)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    spacing = rng.uniform(10.0, 17.0)
    phase = rng.uniform(0.0, spacing)
    grooves = 0.5 + 0.5 * np.sin((yy + phase) * (2.0 * math.pi / spacing))
    grooves = np.power(grooves, rng.uniform(7.0, 12.0))
    board -= grooves[:, :, None] * rng.uniform(8.0, 19.0)
    if rng.random() < 0.35:
        vertical = 0.5 + 0.5 * np.sin((xx + phase) * (2.0 * math.pi / rng.uniform(28.0, 42.0)))
        board -= np.power(vertical, 14.0)[:, :, None] * rng.uniform(2.0, 6.0)
    board += rng.normal(0.0, 2.2, board.shape)

    edge = max(5, size // 130)
    board[:edge, :, :] *= 0.84
    board[-edge:, :, :] *= 0.84
    board[:, :edge, :] *= 0.84
    board[:, -edge:, :] *= 0.84

    for cx, cy in ((0.04, 0.04), (0.96, 0.04), (0.04, 0.96), (0.96, 0.96)):
        if rng.random() < 0.65:
            center = (int(cx * size + rng.normal(0, size * 0.01)), int(cy * size + rng.normal(0, size * 0.01)))
            cv2.circle(board, center, max(3, size // 105), (70, 80, 85), thickness=-1, lineType=cv2.LINE_AA)
            cv2.circle(board, center, max(5, size // 70), (215, 220, 220), thickness=2, lineType=cv2.LINE_AA)
    return np.clip(board, 0, 255).astype(np.uint8)


def _place_crabs_on_board(
    board: np.ndarray,
    templates: Sequence[CrabTemplate],
    config: SyntheticDatasetConfig,
    rng: np.random.Generator,
) -> list[tuple[int, str, np.ndarray]]:
    board_size = board.shape[0]
    layout_profile = _sample_layout_profile(config, rng)
    object_count = _sample_object_count(config, layout_profile, rng)
    class_ids = _sample_class_ids(object_count, config, rng)
    placed_boxes: list[tuple[float, float, float, float]] = []
    object_masks: list[tuple[int, str, np.ndarray]] = []
    min_spacing_px = float(board_size) * max(0.0, float(config.crab_spacing_fraction))
    max_iou = max(0.0, float(config.max_crab_iou))
    even_anchors = _sample_even_anchor_centers(board_size, object_count, config, rng)

    for object_index, class_id in enumerate(class_ids):
        template = templates[class_id]
        for appearance_attempt in range(48):
            target_long = int(board_size * _sample_crab_scale(config, layout_profile, rng))
            rotation = float(rng.uniform(0.0, 360.0))
            patch, alpha, label_mask = _render_crab_patch(template, target_long, rotation, rng)
            ph, pw = patch.shape[:2]
            if ph >= board_size or pw >= board_size:
                continue
            local_box = _mask_to_box(label_mask, pw, ph)
            if local_box is None:
                continue

            for position_attempt in range(18):
                use_anchor = (
                    even_anchors[object_index]
                    if object_index < len(even_anchors) and appearance_attempt < 30 and position_attempt < 12
                    else None
                )
                x, y = _sample_patch_position(
                    board_size,
                    pw,
                    ph,
                    rng,
                    anchor=use_anchor,
                    jitter_fraction=config.even_placement_jitter,
                )
                box = (local_box[0] + x, local_box[1] + y, local_box[2] + x, local_box[3] + y)
                blocked_box = _expand_box_xyxy(box, min_spacing_px, board_size)
                if any(
                    _iou_xyxy(box, existing) > max_iou
                    or _boxes_intersect(blocked_box, _expand_box_xyxy(existing, min_spacing_px, board_size))
                    for existing in placed_boxes
                ):
                    continue
                _paste_patch(board, patch, alpha, x, y)
                mask = np.zeros((board_size, board_size), dtype=np.uint8)
                mask[y : y + ph, x : x + pw] = np.maximum(mask[y : y + ph, x : x + pw], label_mask)
                object_masks.append((template.class_id, template.class_name, mask))
                placed_boxes.append(box)
                break
            else:
                continue
            break
    return object_masks


def _sample_even_anchor_centers(
    board_size: int,
    object_count: int,
    config: SyntheticDatasetConfig,
    rng: np.random.Generator,
) -> list[tuple[float, float]]:
    if object_count <= 0 or rng.random() >= max(0.0, min(1.0, float(config.even_placement_fraction))):
        return []

    margin = float(board_size) * 0.11
    cols = int(math.ceil(math.sqrt(object_count * rng.uniform(0.9, 1.35))))
    cols = max(1, cols)
    rows = int(math.ceil(object_count / cols))
    cell_w = max(1.0, (board_size - 2.0 * margin) / cols)
    cell_h = max(1.0, (board_size - 2.0 * margin) / rows)
    anchors: list[tuple[float, float]] = []
    jitter = min(cell_w, cell_h) * max(0.0, min(0.8, float(config.even_placement_jitter)))
    for row in range(rows):
        for col in range(cols):
            cx = margin + (col + 0.5) * cell_w + float(rng.uniform(-jitter, jitter))
            cy = margin + (row + 0.5) * cell_h + float(rng.uniform(-jitter, jitter))
            cx = float(np.clip(cx, margin, board_size - margin))
            cy = float(np.clip(cy, margin, board_size - margin))
            anchors.append((cx, cy))
    rng.shuffle(anchors)
    return anchors[:object_count]


def _sample_patch_position(
    board_size: int,
    patch_width: int,
    patch_height: int,
    rng: np.random.Generator,
    *,
    anchor: tuple[float, float] | None,
    jitter_fraction: float,
) -> tuple[int, int]:
    margin = board_size // 35
    x_max = max(margin, board_size - patch_width - margin)
    y_max = max(margin, board_size - patch_height - margin)
    if anchor is None:
        return (
            int(rng.integers(margin, x_max + 1)),
            int(rng.integers(margin, y_max + 1)),
        )

    jitter = float(board_size) * max(0.0, min(0.8, float(jitter_fraction))) / max(2.0, math.sqrt(board_size))
    cx = float(anchor[0]) + float(rng.normal(0.0, jitter))
    cy = float(anchor[1]) + float(rng.normal(0.0, jitter))
    x = int(round(cx - patch_width / 2.0))
    y = int(round(cy - patch_height / 2.0))
    return (
        int(np.clip(x, margin, x_max)),
        int(np.clip(y, margin, y_max)),
    )


def _sample_layout_profile(config: SyntheticDatasetConfig, rng: np.random.Generator) -> str:
    sparse = max(0.0, float(config.sparse_layout_fraction))
    full = max(0.0, float(config.full_layout_fraction))
    total = sparse + full
    if total > 0.95:
        sparse *= 0.95 / total
        full *= 0.95 / total
    normal = max(0.0, 1.0 - sparse - full)
    return str(rng.choice(LAYOUT_PROFILE_NAMES, p=[sparse, normal, full]))


def _sample_object_count(
    config: SyntheticDatasetConfig,
    layout_profile: str,
    rng: np.random.Generator,
) -> int:
    min_count = max(0, int(config.min_crabs))
    max_count = max(min_count, int(config.max_crabs))
    if min_count == max_count:
        return min_count

    span = max_count - min_count
    if layout_profile == "sparse":
        lo = min_count
        hi = min(max_count, min_count + max(1, int(math.ceil(span * 0.35))))
    elif layout_profile == "full":
        lo = min(max_count, max(min_count, max_count - max(1, int(math.ceil(span * 0.3)))))
        hi = max_count
    else:
        lo = min(max_count, min_count + max(0, int(math.floor(span * 0.2))))
        hi = max(lo, max_count - max(0, int(math.floor(span * 0.2))))
    return int(rng.integers(lo, hi + 1))


def _sample_crab_scale(
    config: SyntheticDatasetConfig,
    layout_profile: str,
    rng: np.random.Generator,
) -> float:
    profile_scale = {
        "sparse": 0.92,
        "normal": 1.0,
        "full": 1.06,
    }.get(layout_profile, 1.0)
    large_fraction = max(0.0, min(1.0, float(config.large_crab_fraction)))
    if layout_profile == "sparse":
        large_fraction *= 0.8
    elif layout_profile == "full":
        large_fraction *= 1.2

    if rng.random() < min(1.0, large_fraction):
        return _sample_range(rng, config.large_crab_long_edge_range, log=False)
    return _sample_range(rng, config.crab_long_edge_range, log=False) * profile_scale


def _sample_class_ids(
    object_count: int,
    config: SyntheticDatasetConfig,
    rng: np.random.Generator,
) -> list[int]:
    if object_count <= 0:
        return []
    has_green = rng.random() < float(config.green_positive_fraction)
    if has_green:
        class_ids = rng.choice([0, 1, 2], size=object_count, p=[0.42, 0.29, 0.29]).astype(int).tolist()
        if 0 not in class_ids:
            class_ids[int(rng.integers(0, len(class_ids)))] = 0
        return class_ids
    return rng.choice([1, 2], size=object_count, p=[0.5, 0.5]).astype(int).tolist()


def _render_crab_patch(
    template: CrabTemplate,
    target_long_edge: int,
    rotation_deg: float,
    rng: np.random.Generator,
    *,
    color_jitter_strength: float = 1.0,
    paper_pad_fraction_range: tuple[float, float] = (0.08, 0.18),
    paper_radius_fraction_range: tuple[float, float] = (0.075, 0.13),
    paper_alpha_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    src = _jitter_template_color(template.bgr, template.mask, rng, strength=color_jitter_strength)
    mask = template.mask.copy()
    pad_fraction = float(_sample_range(rng, paper_pad_fraction_range))
    source_long_edge = max(1, max(src.shape[:2]))
    scaled_crab_long_edge = max(4.0, float(target_long_edge) / (1.0 + 2.0 * pad_fraction))
    scale = max(0.05, scaled_crab_long_edge / source_long_edge)
    scaled_size = (max(4, int(round(src.shape[1] * scale))), max(4, int(round(src.shape[0] * scale))))
    src = cv2.resize(src, scaled_size, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    mask = cv2.resize(mask, scaled_size, interpolation=cv2.INTER_NEAREST)

    pad = max(3, int(round(max(src.shape[:2]) * pad_fraction)))
    if paper_alpha_scale > 0.001:
        paper_shape = _paper_mask(mask, pad, rng, radius_fraction_range=paper_radius_fraction_range)
        patch_h, patch_w = paper_shape.shape[:2]
        paper_color = rng.uniform(225, 255, size=(1, 1, 3)).astype(np.float32)
        paper = np.full((patch_h, patch_w, 3), paper_color, dtype=np.float32)
        paper += rng.normal(0.0, 2.6, paper.shape)
        patch = np.clip(paper, 0, 255).astype(np.uint8)
        patch[pad : pad + src.shape[0], pad : pad + src.shape[1]] = _alpha_blend(
            patch[pad : pad + src.shape[0], pad : pad + src.shape[1]],
            src,
            _soften_mask(mask, blur=3),
        )
    else:
        patch_h = src.shape[0] + pad * 2
        patch_w = src.shape[1] + pad * 2
        patch = np.zeros((patch_h, patch_w, 3), dtype=np.uint8)
        patch[pad : pad + src.shape[0], pad : pad + src.shape[1]] = src
        paper_shape = np.zeros((patch_h, patch_w), dtype=np.uint8)
    label_mask = np.zeros((patch_h, patch_w), dtype=np.uint8)
    label_mask[pad : pad + mask.shape[0], pad : pad + mask.shape[1]] = mask
    crab_alpha = _soften_mask(label_mask, blur=3)
    paper_alpha = np.clip(paper_shape.astype(np.float32) * max(0.0, float(paper_alpha_scale)), 0, 255).astype(
        np.uint8
    )
    alpha = np.maximum(paper_alpha, crab_alpha)
    return _rotate_patch(patch, alpha, label_mask, rotation_deg)


def _jitter_template_color(
    image: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
    *,
    strength: float = 1.0,
) -> np.ndarray:
    strength = max(0.0, min(1.0, float(strength)))

    def blend_neutral(neutral: float, sampled: float) -> float:
        return neutral + (sampled - neutral) * strength

    out = image.astype(np.float32)
    channel_gains = np.array(
        [
            blend_neutral(1.0, float(rng.uniform(0.72, 1.08))),
            blend_neutral(1.0, float(rng.uniform(0.72, 1.08))),
            blend_neutral(1.0, float(rng.uniform(0.55, 1.0))),
        ],
        dtype=np.float32,
    )
    out *= channel_gains
    out = (out - 127.5) * blend_neutral(1.0, float(rng.uniform(0.72, 1.12))) + 127.5 + blend_neutral(
        0.0,
        float(rng.uniform(-24, 6)),
    )
    out = np.clip(out, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= blend_neutral(1.0, float(rng.uniform(0.42, 1.05)))
    hsv[:, :, 2] *= blend_neutral(1.0, float(rng.uniform(0.68, 1.03)))
    if rng.random() < 0.18 * strength:
        hsv[:, :, 1] *= blend_neutral(1.0, float(rng.uniform(1.0, 1.14)))
    out = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    if rng.random() < 0.2 * strength:
        out = cv2.GaussianBlur(out, (3, 3), 0)
    return _alpha_blend(image, out, mask)


def _paper_mask(
    mask: np.ndarray,
    pad: int,
    rng: np.random.Generator,
    *,
    radius_fraction_range: tuple[float, float] = (0.075, 0.13),
) -> np.ndarray:
    padded = cv2.copyMakeBorder(mask, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    radius = max(3, int(max(mask.shape[:2]) * _sample_range(rng, radius_fraction_range)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    paper = cv2.dilate(padded, kernel, iterations=1)
    paper = cv2.morphologyEx(paper, cv2.MORPH_CLOSE, kernel)
    return _soften_mask(paper, blur=max(5, radius // 3 * 2 + 1))


def _rotate_patch(
    patch: np.ndarray,
    alpha: np.ndarray,
    label_mask: np.ndarray,
    rotation_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = patch.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, rotation_deg, 1.0)
    cos_v = abs(matrix[0, 0])
    sin_v = abs(matrix[0, 1])
    new_w = int((h * sin_v) + (w * cos_v))
    new_h = int((h * cos_v) + (w * sin_v))
    matrix[0, 2] += (new_w / 2.0) - center[0]
    matrix[1, 2] += (new_h / 2.0) - center[1]
    rotated_patch = cv2.warpAffine(patch, matrix, (new_w, new_h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    rotated_alpha = cv2.warpAffine(alpha, matrix, (new_w, new_h), flags=cv2.INTER_LINEAR, borderValue=0)
    rotated_label = cv2.warpAffine(label_mask, matrix, (new_w, new_h), flags=cv2.INTER_NEAREST, borderValue=0)
    return rotated_patch, rotated_alpha, rotated_label


def _add_laminate_and_tape(board: np.ndarray, rng: np.random.Generator) -> None:
    h, w = board.shape[:2]
    overlay = board.copy()
    for _ in range(int(rng.integers(5, 13))):
        cx = int(rng.uniform(w * 0.08, w * 0.92))
        cy = int(rng.uniform(h * 0.08, h * 0.92))
        length = int(rng.uniform(w * 0.08, w * 0.24))
        thickness = int(rng.uniform(h * 0.012, h * 0.035))
        angle = rng.uniform(-45, 45)
        rect = ((cx, cy), (length, thickness), angle)
        pts = cv2.boxPoints(rect).astype(np.int32)
        color = (245 + int(rng.integers(-8, 9)),) * 3
        cv2.fillConvexPoly(overlay, pts, color, lineType=cv2.LINE_AA)
    tape_alpha = float(rng.uniform(0.08, 0.18))
    cv2.addWeighted(overlay, tape_alpha, board, 1.0 - tape_alpha, 0, dst=board)

    glare = np.zeros_like(board, dtype=np.float32)
    for _ in range(int(rng.integers(1, 4))):
        x0 = rng.uniform(-w * 0.15, w * 0.85)
        y0 = rng.uniform(0, h)
        x1 = x0 + rng.uniform(w * 0.2, w * 0.65)
        y1 = y0 + rng.uniform(-h * 0.2, h * 0.2)
        cv2.line(
            glare,
            (int(x0), int(y0)),
            (int(x1), int(y1)),
            (rng.uniform(28, 70),) * 3,
            thickness=int(rng.uniform(h * 0.006, h * 0.02)),
            lineType=cv2.LINE_AA,
        )
    glare = cv2.GaussianBlur(glare, (0, 0), sigmaX=rng.uniform(3.0, 9.0))
    board[:] = np.clip(board.astype(np.float32) + glare, 0, 255).astype(np.uint8)


def _sample_board_quad(
    width: int,
    height: int,
    config: SyntheticDatasetConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    image_area = float(width * height)
    min_visible_fraction = max(0.0, min(1.0, float(config.board_min_visible_fraction)))
    min_frame_area = image_area * max(0.0, float(config.board_min_frame_area_fraction))
    for _attempt in range(80):
        side = _sample_range(rng, config.board_long_edge_range, log=True) * min(width, height)
        board_w = side * rng.uniform(0.95, 1.18)
        board_h = side * rng.uniform(0.9, 1.12)
        center = np.array(
            [
                width * rng.uniform(0.5 - config.board_center_jitter, 0.5 + config.board_center_jitter),
                height * rng.uniform(0.5 - config.board_center_jitter, 0.5 + config.board_center_jitter),
            ],
            dtype=np.float32,
        )
        corners = np.array(
            [
                [-0.5, -0.5],
                [0.5, -0.5],
                [0.5, 0.5],
                [-0.5, 0.5],
            ],
            dtype=np.float32,
        )
        tilt = _sample_range(rng, config.board_tilt_range, log=False)
        yaw = float(rng.choice([-1.0, 1.0]) * tilt * rng.uniform(0.35, 1.0))
        pitch = float(rng.choice([-1.0, 1.0]) * tilt * rng.uniform(0.35, 1.0))
        depth = 1.0 + corners[:, 0] * yaw + corners[:, 1] * pitch
        if float(depth.min()) <= 0.2:
            continue
        projected = corners / depth[:, None]
        projected[:, 0] *= board_w
        projected[:, 1] *= board_h
        projected += rng.normal(0.0, side * max(0.0, float(config.board_corner_jitter)), size=projected.shape).astype(
            np.float32
        )

        angle = math.radians(float(_sample_range(rng, config.board_roll_range_degrees)))
        rot = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]], dtype=np.float32)
        quad = projected @ rot.T + center
        visible_fraction, visible_area = _quad_visible_metrics(quad, width, height)
        x0, y0 = quad.min(axis=0)
        x1, y1 = quad.max(axis=0)
        visible_w = min(width, x1) - max(0, x0)
        visible_h = min(height, y1) - max(0, y0)
        if (
            visible_fraction >= min_visible_fraction
            and visible_area >= min_frame_area
            and visible_w > 60
            and visible_h > 60
        ):
            return quad.astype(np.float32)
    margin = min(width, height) * 0.08
    return np.array(
        [[margin, margin], [width - margin, margin], [width - margin, height - margin], [margin, height - margin]],
        dtype=np.float32,
    )


def _sample_range(
    rng: np.random.Generator,
    values: tuple[float, float],
    *,
    log: bool = False,
) -> float:
    lo, hi = sorted((float(values[0]), float(values[1])))
    if math.isclose(lo, hi):
        return lo
    if log and lo > 0.0 and hi > 0.0:
        return float(math.exp(rng.uniform(math.log(lo), math.log(hi))))
    return float(rng.uniform(lo, hi))


def _box_long_edge(box: tuple[float, float, float, float]) -> float:
    return max(0.0, float(max(box[2] - box[0], box[3] - box[1])))


def _quad_visible_metrics(quad: np.ndarray, width: int, height: int) -> tuple[float, float]:
    total_area = abs(float(cv2.contourArea(quad.astype(np.float32))))
    if total_area <= 1.0:
        return 0.0, 0.0
    frame = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    try:
        visible_area, _polygon = cv2.intersectConvexConvex(quad.astype(np.float32), frame)
    except cv2.error:
        visible_area = 0.0
    visible_area = max(0.0, float(visible_area))
    return min(1.0, visible_area / total_area), visible_area


def _apply_underwater_camera_effects(
    image: np.ndarray,
    config: SyntheticDatasetConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    out = image.astype(np.float32)
    out *= np.array(
        [
            rng.uniform(0.9, 1.16),
            rng.uniform(0.9, 1.14),
            rng.uniform(0.58, 1.08),
        ],
        dtype=np.float32,
    )
    haze_color = np.array(
        [rng.uniform(80, 150), rng.uniform(125, 190), rng.uniform(100, 165)],
        dtype=np.float32,
    )
    haze = rng.uniform(0.0, 0.22)
    out = out * (1.0 - haze) + haze_color * haze
    out = (out - 127.5) * rng.uniform(0.78, 1.18) + 127.5 + rng.uniform(-15, 15)

    if rng.random() < max(0.0, min(1.0, float(config.camera_blur_fraction))):
        out = cv2.GaussianBlur(
            np.clip(out, 0, 255).astype(np.uint8),
            (0, 0),
            sigmaX=_sample_range(rng, config.camera_blur_sigma_range),
        ).astype(np.float32)
    out += rng.normal(0.0, rng.uniform(1.0, 7.5), out.shape)

    h, w = out.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    distance = np.sqrt(((xx - w / 2) / max(1, w / 2)) ** 2 + ((yy - h / 2) / max(1, h / 2)) ** 2)
    vignette = 1.0 - np.clip(distance - rng.uniform(0.35, 0.65), 0, 1) * rng.uniform(0.08, 0.28)
    out *= vignette[:, :, None]

    out_u8 = np.clip(out, 0, 255).astype(np.uint8)
    if rng.random() < max(0.0, min(1.0, float(config.jpeg_artifact_fraction))):
        quality_min, quality_max = sorted((int(config.jpeg_quality_range[0]), int(config.jpeg_quality_range[1])))
        quality = int(rng.integers(max(1, quality_min), min(100, quality_max) + 1))
        ok, encoded = cv2.imencode(".jpg", out_u8, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if ok:
            decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if decoded is not None:
                out_u8 = decoded
    return out_u8


def _mask_to_box(mask: np.ndarray, width: int, height: int) -> tuple[float, float, float, float] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x0 = float(np.clip(xs.min(), 0, width - 1))
    y0 = float(np.clip(ys.min(), 0, height - 1))
    x1 = float(np.clip(xs.max() + 1, 0, width))
    y1 = float(np.clip(ys.max() + 1, 0, height))
    if (x1 - x0) < 5 or (y1 - y0) < 5:
        return None
    visible_area = float(np.count_nonzero(mask))
    if visible_area < 35:
        return None
    return x0, y0, x1, y1


def _paste_patch(base: np.ndarray, patch: np.ndarray, alpha: np.ndarray, x: int, y: int) -> None:
    ph, pw = patch.shape[:2]
    h, w = base.shape[:2]
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(w, x + pw)
    y1 = min(h, y + ph)
    if x0 >= x1 or y0 >= y1:
        return
    px0 = x0 - x
    py0 = y0 - y
    roi = base[y0:y1, x0:x1]
    base[y0:y1, x0:x1] = _alpha_blend(roi, patch[py0 : py0 + (y1 - y0), px0 : px0 + (x1 - x0)], alpha[py0 : py0 + (y1 - y0), px0 : px0 + (x1 - x0)])


def _alpha_blend(base: np.ndarray, top: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    a = (alpha.astype(np.float32) / 255.0)[:, :, None]
    return np.clip(top.astype(np.float32) * a + base.astype(np.float32) * (1.0 - a), 0, 255).astype(np.uint8)


def _soften_mask(mask: np.ndarray, *, blur: int) -> np.ndarray:
    blur = max(1, int(blur))
    if blur % 2 == 0:
        blur += 1
    return cv2.GaussianBlur(mask, (blur, blur), 0)


def _iou_xyxy(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    denom = area_a + area_b - inter
    return inter / denom if denom else 0.0


def _boxes_intersect(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    return max(a[0], b[0]) < min(a[2], b[2]) and max(a[1], b[1]) < min(a[3], b[3])


def _expand_box_xyxy(
    box: tuple[float, float, float, float],
    margin: float,
    limit: int,
) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = box
    return (
        max(0.0, x0 - margin),
        max(0.0, y0 - margin),
        min(float(limit), x1 + margin),
        min(float(limit), y1 + margin),
    )


def _write_jpeg(path: Path, image: np.ndarray, *, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise OSError(f"could not write image: {path}")


def _write_data_yaml(path: Path, output_dir: Path) -> None:
    names = "\n".join(f"  {idx}: {name}" for idx, name in enumerate(CRAB_CLASS_NAMES))
    yaml_text = (
        f"path: {output_dir.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        f"{names}\n"
    )
    path.write_text(yaml_text, encoding="utf-8")


def _write_manifest(
    output_dir: Path,
    config: SyntheticDatasetConfig,
    class_counts: Mapping[str, int],
    train_images: int,
    val_images: int,
) -> None:
    manifest = {
        "image_count": int(config.image_count),
        "image_size": list(config.image_size),
        "seed": int(config.seed),
        "classes": list(CRAB_CLASS_NAMES),
        "class_counts": dict(class_counts),
        "train_images": train_images,
        "val_images": val_images,
        "crab_long_edge_range": list(config.crab_long_edge_range),
        "large_crab_fraction": float(config.large_crab_fraction),
        "large_crab_long_edge_range": list(config.large_crab_long_edge_range),
        "sparse_layout_fraction": float(config.sparse_layout_fraction),
        "full_layout_fraction": float(config.full_layout_fraction),
        "even_placement_fraction": float(config.even_placement_fraction),
        "even_placement_jitter": float(config.even_placement_jitter),
        "max_crab_iou": float(config.max_crab_iou),
        "crab_spacing_fraction": float(config.crab_spacing_fraction),
        "min_crab_box_long_edge_px": int(config.min_crab_box_long_edge_px),
        "board_long_edge_range": list(config.board_long_edge_range),
        "board_roll_range_degrees": list(config.board_roll_range_degrees),
        "board_tilt_range": list(config.board_tilt_range),
        "board_center_jitter": float(config.board_center_jitter),
        "board_corner_jitter": float(config.board_corner_jitter),
        "board_min_visible_fraction": float(config.board_min_visible_fraction),
        "board_min_frame_area_fraction": float(config.board_min_frame_area_fraction),
        "reference_paths": {key: str(Path(value)) for key, value in config.reference_paths.items()},
        "background_source_count": len(config.background_paths),
        "camera_blur_fraction": float(config.camera_blur_fraction),
        "camera_blur_sigma_range": list(config.camera_blur_sigma_range),
        "jpeg_artifact_fraction": float(config.jpeg_artifact_fraction),
        "jpeg_quality_range": list(config.jpeg_quality_range),
        "green_positive_fraction": float(config.green_positive_fraction),
        "empty_fraction": float(config.empty_fraction),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _write_preview(
    output_dir: Path,
    preview_items: Sequence[tuple[Path, list[YoloBox]]],
    image_size: tuple[int, int],
) -> Path | None:
    if not preview_items:
        return None
    cells: list[np.ndarray] = []
    for image_path, boxes in preview_items:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        cells.append(_draw_boxes(image, boxes))
    if not cells:
        return None
    width, height = image_size
    thumb_w = 320
    thumb_h = max(1, int(round(height * thumb_w / max(1, width))))
    thumbs = [cv2.resize(cell, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA) for cell in cells]
    cols = min(4, len(thumbs))
    rows = int(math.ceil(len(thumbs) / cols))
    sheet = np.full((rows * thumb_h, cols * thumb_w, 3), 245, dtype=np.uint8)
    for idx, thumb in enumerate(thumbs):
        row = idx // cols
        col = idx % cols
        sheet[row * thumb_h : (row + 1) * thumb_h, col * thumb_w : (col + 1) * thumb_w] = thumb
    preview_path = output_dir / "preview.jpg"
    _write_jpeg(preview_path, sheet, quality=92)
    return preview_path


def _draw_boxes(image: np.ndarray, boxes: Sequence[YoloBox]) -> np.ndarray:
    out = image.copy()
    colors = {
        "european_green_crab": (45, 230, 75),
        "native_rock_crab": (255, 160, 45),
        "jonah_crab": (235, 80, 80),
    }
    for box in boxes:
        x0, y0, x1, y1 = [int(round(v)) for v in box.xyxy]
        color = colors.get(box.class_name, (255, 255, 255))
        cv2.rectangle(out, (x0, y0), (x1, y1), color, 2, lineType=cv2.LINE_AA)
        label = box.class_name.replace("_", " ")
        cv2.putText(out, label, (x0, max(16, y0 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
    return out
