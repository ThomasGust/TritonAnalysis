"""Plane-anchored synthetic YOLO dataset generation for crab boards."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import cv2
import numpy as np

from triton_analysis.crab.synthetic import (
    CRAB_CLASS_NAMES,
    IMAGE_EXTENSIONS,
    CrabTemplate,
    SyntheticDatasetResult,
    YoloBox,
    _alpha_blend,
    _boxes_intersect,
    _expand_box_xyxy,
    _extract_crab_mask,
    _iou_xyxy,
    _mask_to_box,
    _project_board_boxes,
    _render_crab_patch,
    _sample_class_ids,
    _sample_crab_scale,
    _sample_even_anchor_centers,
    _sample_layout_profile,
    _sample_object_count,
    _sample_patch_position,
    _write_jpeg,
)


ProgressCallback = Callable[[dict[str, object]], None]
REPO_ROOT = Path(__file__).resolve().parents[2]
BUNDLED_CRAB_TEMPLATE_DIRS = (
    Path("data") / "crab" / "templates",
    Path("data") / "crab" / "real crabs",
    Path("data") / "crab" / "crab templates",
)
WORKSPACE_CRAB_TEMPLATE_DIRS = (
    Path("data") / "real crabs",
    Path("data") / "crabs",
    Path("data") / "crab templates",
)


@dataclass(frozen=True)
class BoardPlaneAnnotation:
    """One empty-board image and the visible board plane quadrilateral."""

    image_path: Path
    quad_xy: tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]]
    image_size: tuple[int, int] | None = None
    label: str = ""


@dataclass
class PlaneProjectedDatasetConfig:
    """Configuration for compositing crabs onto annotated real board planes."""

    output_dir: Path
    annotations: Sequence[BoardPlaneAnnotation]
    template_paths: Mapping[str, Sequence[Path]]
    image_count: int = 2000
    val_fraction: float = 0.2
    seed: int = 7
    board_size: int = 768
    min_crabs: int = 4
    max_crabs: int = 11
    crab_long_edge_range: tuple[float, float] = (0.10, 0.42)
    large_crab_fraction: float = 0.32
    large_crab_long_edge_range: tuple[float, float] = (0.34, 0.52)
    sparse_layout_fraction: float = 0.25
    full_layout_fraction: float = 0.3
    even_placement_fraction: float = 0.65
    even_placement_jitter: float = 0.36
    max_crab_iou: float = 0.025
    crab_spacing_fraction: float = 0.012
    min_crab_box_long_edge_px: int = 32
    green_positive_fraction: float = 0.7
    empty_fraction: float = 0.0
    jpeg_artifact_fraction: float = 0.1
    jpeg_quality_range: tuple[int, int] = (90, 100)
    crab_color_jitter_strength: float = 0.35
    paper_pad_fraction_range: tuple[float, float] = (0.025, 0.075)
    paper_radius_fraction_range: tuple[float, float] = (0.025, 0.065)
    paper_alpha_scale: float = 0.0
    scene_retry_count: int = 3
    preview_count: int = 12


def discover_board_images(root: str | Path) -> list[Path]:
    """Return image files below *root* that can be annotated as empty boards."""
    root_path = Path(root).expanduser()
    if not root_path.exists():
        return []
    candidates = [root_path] if root_path.is_file() else root_path.rglob("*")
    return sorted(
        (path for path in candidates if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda path: str(path).lower(),
    )


def discover_default_crab_template_paths(workspace_root: str | Path) -> dict[str, list[Path]]:
    """Find bundled, workspace, and legacy local crab template images."""
    templates: dict[str, list[Path]] = {name: [] for name in CRAB_CLASS_NAMES}
    repo = REPO_ROOT.expanduser()
    bundled_mate_found = {name: False for name in CRAB_CLASS_NAMES}
    bundled_mate_candidates = {
        "european_green_crab": (
            repo / "data" / "crab" / "templates" / "european_green_crab_mate_reference.jpg",
        ),
        "native_rock_crab": (
            repo / "data" / "crab" / "templates" / "native_rock_crab_mate_reference.jpg",
        ),
        "jonah_crab": (
            repo / "data" / "crab" / "templates" / "jonah_crab_mate_reference.png",
        ),
    }
    for class_name, paths in bundled_mate_candidates.items():
        for path in paths:
            if path.exists():
                templates[class_name].append(path)
                bundled_mate_found[class_name] = True
                break

    downloads = _default_downloads_dir()
    mate_candidates = {
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
    for class_name, paths in mate_candidates.items():
        if bundled_mate_found[class_name]:
            continue
        for path in paths:
            if path.exists():
                templates[class_name].append(path)
                break

    for rel in BUNDLED_CRAB_TEMPLATE_DIRS:
        for path in discover_board_images(repo / rel):
            class_name = _class_name_from_template_path(path)
            if class_name is not None:
                templates[class_name].append(path)

    workspace = Path(workspace_root).expanduser()
    for rel in WORKSPACE_CRAB_TEMPLATE_DIRS:
        for path in discover_board_images(workspace / rel):
            class_name = _class_name_from_template_path(path)
            if class_name is not None and not templates[class_name]:
                templates[class_name].append(path)

    return {class_name: _dedupe_paths(paths) for class_name, paths in templates.items()}


def _default_downloads_dir() -> Path:
    return Path.home() / "Downloads"


def load_board_plane_annotations(
    path: str | Path,
    *,
    image_root: str | Path | None = None,
) -> list[BoardPlaneAnnotation]:
    """Load saved board-plane annotations."""
    annotation_path = Path(path).expanduser()
    if not annotation_path.exists():
        return []
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    root = Path(image_root).expanduser() if image_root is not None else annotation_path.parent
    annotations: list[BoardPlaneAnnotation] = []
    for item in payload.get("annotations", []):
        raw_path = Path(str(item.get("image", ""))).expanduser()
        image_path = raw_path if raw_path.is_absolute() else root / raw_path
        points = item.get("quad_xy", [])
        if len(points) != 4:
            continue
        size_value = item.get("image_size")
        image_size = None
        if isinstance(size_value, list) and len(size_value) == 2:
            image_size = (int(size_value[0]), int(size_value[1]))
        annotations.append(
            BoardPlaneAnnotation(
                image_path=image_path,
                quad_xy=tuple((float(point[0]), float(point[1])) for point in points),  # type: ignore[arg-type]
                image_size=image_size,
                label=str(item.get("label") or ""),
            )
        )
    return annotations


def save_board_plane_annotations(
    path: str | Path,
    annotations: Sequence[BoardPlaneAnnotation],
    *,
    image_root: str | Path | None = None,
) -> None:
    """Save board-plane annotations as portable JSON."""
    annotation_path = Path(path).expanduser()
    root = Path(image_root).expanduser() if image_root is not None else annotation_path.parent
    rows = []
    for annotation in annotations:
        image_path = Path(annotation.image_path).expanduser()
        try:
            image_value = image_path.resolve().relative_to(root.resolve())
        except (OSError, ValueError):
            image_value = image_path
        rows.append(
            {
                "image": str(image_value),
                "label": annotation.label or image_path.name,
                "image_size": list(annotation.image_size) if annotation.image_size else None,
                "quad_xy": [[round(float(x), 3), round(float(y), 3)] for x, y in annotation.quad_xy],
            }
        )
    payload = {
        "version": 1,
        "description": "Empty board plane annotations for crab dataset generation.",
        "annotations": rows,
    }
    annotation_path.parent.mkdir(parents=True, exist_ok=True)
    annotation_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def generate_plane_projected_dataset(
    config: PlaneProjectedDatasetConfig,
    *,
    progress_callback: ProgressCallback | None = None,
) -> SyntheticDatasetResult:
    """Generate YOLO data by projecting crab cutouts onto annotated board planes."""
    annotations = [annotation for annotation in config.annotations if len(annotation.quad_xy) == 4]
    if not annotations:
        raise ValueError("at least one board-plane annotation is required")

    template_sets = _load_template_sets(config.template_paths)
    missing = [class_name for class_name in CRAB_CLASS_NAMES if not template_sets[class_name]]
    if missing:
        raise ValueError("missing crab template image(s): " + ", ".join(missing))
    board_images = _load_board_images(annotations)

    output_dir = Path(config.output_dir).expanduser()
    split_dirs = _prepare_dataset_dirs(output_dir)
    rng = np.random.default_rng(int(config.seed))
    class_counts = {name: 0 for name in CRAB_CLASS_NAMES}
    preview_items: list[tuple[Path, list[YoloBox]]] = []
    train_images = 0
    val_images = 0
    image_count = max(0, int(config.image_count))

    for index in range(image_count):
        scene, boxes, source_path = _render_plane_scene(annotations, board_images, template_sets, config, rng)
        split = "val" if rng.random() < float(config.val_fraction) else "train"
        if split == "val":
            val_images += 1
        else:
            train_images += 1
        height, width = scene.shape[:2]
        stem = f"crab_plane_{index:06d}"
        image_path = split_dirs[split]["images"] / f"{stem}.jpg"
        label_path = split_dirs[split]["labels"] / f"{stem}.txt"
        _write_jpeg(image_path, scene, quality=92)
        label_path.write_text("\n".join(box.to_yolo_line(width, height) for box in boxes), encoding="utf-8")
        for box in boxes:
            class_counts[box.class_name] += 1
        if len(preview_items) < max(0, int(config.preview_count)):
            preview_items.append((image_path, boxes))
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "image",
                    "index": index + 1,
                    "total": image_count,
                    "source": str(source_path),
                    "objects": len(boxes),
                }
            )

    data_yaml = output_dir / "data.yaml"
    _write_data_yaml(data_yaml, output_dir)
    (output_dir / "classes.txt").write_text("\n".join(CRAB_CLASS_NAMES) + "\n", encoding="utf-8")
    _write_manifest(output_dir, config, class_counts, train_images, val_images)
    preview_image = _write_preview(output_dir, preview_items)
    return SyntheticDatasetResult(
        output_dir=output_dir,
        data_yaml=data_yaml,
        preview_image=preview_image,
        train_images=train_images,
        val_images=val_images,
        class_counts=class_counts,
    )


def _class_name_from_template_path(path: Path) -> str | None:
    text = path.stem.lower().replace("-", "_").replace(" ", "_")
    if "jonah" in text:
        return "jonah_crab"
    if "rock" in text or "native" in text:
        return "native_rock_crab"
    if "green" in text or "euro" in text:
        return "european_green_crab"
    return None


def _dedupe_paths(paths: Sequence[Path]) -> list[Path]:
    out: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        try:
            key = path.expanduser().resolve()
        except OSError:
            key = path.expanduser()
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def _load_template_sets(template_paths: Mapping[str, Sequence[Path]]) -> dict[str, list[CrabTemplate]]:
    template_sets = {name: [] for name in CRAB_CLASS_NAMES}
    for class_id, class_name in enumerate(CRAB_CLASS_NAMES):
        for path in template_paths.get(class_name, ()):
            template_sets[class_name].append(_load_template(path, class_id, class_name))
    return template_sets


def _load_template(path: str | Path, class_id: int, class_name: str) -> CrabTemplate:
    image_path = Path(path).expanduser()
    image = _read_image(image_path)
    if image is None:
        raise FileNotFoundError(image_path)
    mask = _extract_crab_mask(image)
    x, y, w, h = cv2.boundingRect(mask)
    pad = max(3, int(max(w, h) * 0.04))
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(image.shape[1], x + w + pad)
    y1 = min(image.shape[0], y + h + pad)
    return CrabTemplate(
        class_id=class_id,
        class_name=class_name,
        bgr=image[y0:y1, x0:x1].copy(),
        mask=mask[y0:y1, x0:x1].copy(),
    )


def _read_image(path: str | Path) -> np.ndarray | None:
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _load_board_images(annotations: Sequence[BoardPlaneAnnotation]) -> dict[Path, np.ndarray]:
    images: dict[Path, np.ndarray] = {}
    for annotation in annotations:
        key = Path(annotation.image_path)
        if key in images:
            continue
        image = _read_image(key)
        if image is None:
            raise FileNotFoundError(key)
        images[key] = image
    return images


def _render_plane_scene(
    annotations: Sequence[BoardPlaneAnnotation],
    board_images: Mapping[Path, np.ndarray],
    template_sets: Mapping[str, Sequence[CrabTemplate]],
    config: PlaneProjectedDatasetConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[YoloBox], Path]:
    min_long_edge = max(0.0, float(config.min_crab_box_long_edge_px))
    best_scene: np.ndarray | None = None
    best_boxes: list[YoloBox] = []
    best_source = Path(annotations[0].image_path)
    best_min_edge = -1.0

    for _attempt in range(max(1, int(config.scene_retry_count))):
        scene, boxes, source_path = _render_plane_scene_once(annotations, board_images, template_sets, config, rng)
        if not boxes:
            return scene, boxes, source_path
        min_box_edge = min((_box_long_edge(box.xyxy) for box in boxes), default=0.0)
        if min_box_edge >= min_long_edge:
            return scene, boxes, source_path
        if min_box_edge > best_min_edge:
            best_scene = scene
            best_boxes = boxes
            best_source = source_path
            best_min_edge = min_box_edge
    if best_scene is None:
        return _render_plane_scene_once(annotations, board_images, template_sets, config, rng)
    return best_scene, best_boxes, best_source


def _render_plane_scene_once(
    annotations: Sequence[BoardPlaneAnnotation],
    board_images: Mapping[Path, np.ndarray],
    template_sets: Mapping[str, Sequence[CrabTemplate]],
    config: PlaneProjectedDatasetConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[YoloBox], Path]:
    annotation = annotations[int(rng.integers(0, len(annotations)))]
    image = board_images[Path(annotation.image_path)].copy()
    height, width = image.shape[:2]
    board_size = max(128, int(config.board_size))
    overlay, overlay_alpha, board_boxes = _render_crabs_on_plane(template_sets, config, rng, board_size)

    source = np.array(
        [[0, 0], [board_size - 1, 0], [board_size - 1, board_size - 1], [0, board_size - 1]],
        dtype=np.float32,
    )
    destination = np.array(annotation.quad_xy, dtype=np.float32)
    transform = cv2.getPerspectiveTransform(source, destination)
    warped_overlay = cv2.warpPerspective(overlay, transform, (width, height), flags=cv2.INTER_LINEAR)
    warped_alpha = cv2.warpPerspective(overlay_alpha, transform, (width, height), flags=cv2.INTER_LINEAR)
    scene = _alpha_blend(image, warped_overlay, warped_alpha)

    if rng.random() < max(0.0, min(1.0, float(config.jpeg_artifact_fraction))):
        lo, hi = sorted((int(config.jpeg_quality_range[0]), int(config.jpeg_quality_range[1])))
        quality = int(rng.integers(max(1, lo), min(100, hi) + 1))
        ok, encoded = cv2.imencode(".jpg", scene, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if ok:
            decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if decoded is not None:
                scene = decoded

    boxes = _project_board_boxes(board_boxes, transform, width, height)
    return scene, boxes, Path(annotation.image_path)


def _render_crabs_on_plane(
    template_sets: Mapping[str, Sequence[CrabTemplate]],
    config: PlaneProjectedDatasetConfig,
    rng: np.random.Generator,
    board_size: int,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, str, tuple[float, float, float, float]]]]:
    overlay = np.zeros((board_size, board_size, 3), dtype=np.uint8)
    overlay_alpha = np.zeros((board_size, board_size), dtype=np.uint8)
    board_boxes: list[tuple[int, str, tuple[float, float, float, float]]] = []
    if rng.random() < float(config.empty_fraction):
        return overlay, overlay_alpha, board_boxes

    layout_profile = _sample_layout_profile(config, rng)  # type: ignore[arg-type]
    object_count = _sample_object_count(config, layout_profile, rng)  # type: ignore[arg-type]
    class_ids = _sample_class_ids(object_count, config, rng)  # type: ignore[arg-type]
    placed_boxes: list[tuple[float, float, float, float]] = []
    min_spacing_px = float(board_size) * max(0.0, float(config.crab_spacing_fraction))
    max_iou = max(0.0, float(config.max_crab_iou))
    even_anchors = _sample_even_anchor_centers(board_size, object_count, config, rng)  # type: ignore[arg-type]

    for object_index, class_id in enumerate(class_ids):
        class_name = CRAB_CLASS_NAMES[int(class_id)]
        templates = template_sets[class_name]
        if not templates:
            continue
        template = templates[int(rng.integers(0, len(templates)))]
        for appearance_attempt in range(10):
            target_long = int(board_size * _sample_crab_scale(config, layout_profile, rng))  # type: ignore[arg-type]
            rotation = float(rng.uniform(0.0, 360.0))
            patch, alpha, label_mask = _render_crab_patch(
                template,
                target_long,
                rotation,
                rng,
                color_jitter_strength=config.crab_color_jitter_strength,
                paper_pad_fraction_range=config.paper_pad_fraction_range,
                paper_radius_fraction_range=config.paper_radius_fraction_range,
                paper_alpha_scale=config.paper_alpha_scale,
            )
            ph, pw = patch.shape[:2]
            if ph >= board_size or pw >= board_size:
                continue
            local_box = _mask_to_box(label_mask, pw, ph)
            if local_box is None:
                continue
            for position_attempt in range(64):
                use_anchor = (
                    even_anchors[object_index]
                    if object_index < len(even_anchors) and appearance_attempt < 4 and position_attempt < 18
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
                _paste_overlay(overlay, overlay_alpha, patch, alpha, x, y)
                board_boxes.append((int(class_id), class_name, box))
                placed_boxes.append(box)
                break
            else:
                continue
            break
    return overlay, overlay_alpha, board_boxes


def _paste_overlay(
    overlay: np.ndarray,
    overlay_alpha: np.ndarray,
    patch: np.ndarray,
    alpha: np.ndarray,
    x: int,
    y: int,
) -> None:
    ph, pw = patch.shape[:2]
    h, w = overlay.shape[:2]
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(w, x + pw)
    y1 = min(h, y + ph)
    if x0 >= x1 or y0 >= y1:
        return
    px0 = x0 - x
    py0 = y0 - y
    roi = overlay[y0:y1, x0:x1]
    patch_roi = patch[py0 : py0 + (y1 - y0), px0 : px0 + (x1 - x0)]
    alpha_roi = alpha[py0 : py0 + (y1 - y0), px0 : px0 + (x1 - x0)]
    overlay[y0:y1, x0:x1] = _alpha_blend(roi, patch_roi, alpha_roi)
    overlay_alpha[y0:y1, x0:x1] = np.maximum(overlay_alpha[y0:y1, x0:x1], alpha_roi)


def _prepare_dataset_dirs(output_dir: Path) -> dict[str, dict[str, Path]]:
    split_dirs: dict[str, dict[str, Path]] = {}
    for split in ("train", "val"):
        image_dir = output_dir / "images" / split
        label_dir = output_dir / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        split_dirs[split] = {"images": image_dir, "labels": label_dir}
    return split_dirs


def _write_data_yaml(path: Path, output_dir: Path) -> None:
    names = "\n".join(f"  {idx}: {name}" for idx, name in enumerate(CRAB_CLASS_NAMES))
    path.write_text(
        f"path: {output_dir.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        f"{names}\n",
        encoding="utf-8",
    )


def _write_manifest(
    output_dir: Path,
    config: PlaneProjectedDatasetConfig,
    class_counts: Mapping[str, int],
    train_images: int,
    val_images: int,
) -> None:
    payload = {
        "generator": "plane_projected_crab_dataset",
        "image_count": int(config.image_count),
        "seed": int(config.seed),
        "classes": list(CRAB_CLASS_NAMES),
        "class_counts": dict(class_counts),
        "train_images": int(train_images),
        "val_images": int(val_images),
        "board_size": int(config.board_size),
        "annotation_count": len(config.annotations),
        "source_images": [str(annotation.image_path) for annotation in config.annotations],
        "template_paths": {class_name: [str(path) for path in paths] for class_name, paths in config.template_paths.items()},
        "min_crabs": int(config.min_crabs),
        "max_crabs": int(config.max_crabs),
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
        "green_positive_fraction": float(config.green_positive_fraction),
        "empty_fraction": float(config.empty_fraction),
        "jpeg_artifact_fraction": float(config.jpeg_artifact_fraction),
        "jpeg_quality_range": list(config.jpeg_quality_range),
        "crab_color_jitter_strength": float(config.crab_color_jitter_strength),
        "paper_pad_fraction_range": list(config.paper_pad_fraction_range),
        "paper_radius_fraction_range": list(config.paper_radius_fraction_range),
        "paper_alpha_scale": float(config.paper_alpha_scale),
        "scene_retry_count": int(config.scene_retry_count),
    }
    (output_dir / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_preview(output_dir: Path, preview_items: Sequence[tuple[Path, list[YoloBox]]]) -> Path | None:
    if not preview_items:
        return None
    cells = []
    for image_path, boxes in preview_items:
        image = _read_image(image_path)
        if image is None:
            continue
        cells.append(_draw_boxes(image, boxes))
    if not cells:
        return None
    thumb_w = 320
    thumbs = []
    for cell in cells:
        height, width = cell.shape[:2]
        thumb_h = max(1, int(round(height * thumb_w / max(1, width))))
        thumbs.append(cv2.resize(cell, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA))
    max_h = max(thumb.shape[0] for thumb in thumbs)
    cols = min(4, len(thumbs))
    rows = int(math.ceil(len(thumbs) / cols))
    sheet = np.full((rows * max_h, cols * thumb_w, 3), 245, dtype=np.uint8)
    for idx, thumb in enumerate(thumbs):
        row = idx // cols
        col = idx % cols
        sheet[row * max_h : row * max_h + thumb.shape[0], col * thumb_w : (col + 1) * thumb_w] = thumb
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


def _box_long_edge(box: tuple[float, float, float, float]) -> float:
    return max(0.0, float(max(box[2] - box[0], box[3] - box[1])))
