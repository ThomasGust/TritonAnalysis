import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from triton_analysis.crab.synthetic import CRAB_CLASS_NAMES, SyntheticDatasetConfig, generate_synthetic_dataset


pytestmark = pytest.mark.vision


def _write_template(path: Path, color: tuple[int, int, int], *, vertical: bool = False) -> None:
    image = np.full((120, 90, 3), 245, dtype=np.uint8)
    center = (45, 60)
    axes = (24, 36) if vertical else (32, 22)
    cv2.ellipse(image, center, axes, 0, 0, 360, color, thickness=-1, lineType=cv2.LINE_AA)
    cv2.line(image, (18, 38), (5, 20), color, 5, cv2.LINE_AA)
    cv2.line(image, (72, 38), (85, 20), color, 5, cv2.LINE_AA)
    cv2.line(image, (18, 82), (5, 100), color, 5, cv2.LINE_AA)
    cv2.line(image, (72, 82), (85, 100), color, 5, cv2.LINE_AA)
    assert cv2.imwrite(str(path), image)


def test_generate_synthetic_dataset_writes_yolo_structure(tmp_path: Path):
    refs_dir = tmp_path / "refs"
    refs_dir.mkdir()
    refs = {
        "european_green_crab": refs_dir / "green.jpg",
        "native_rock_crab": refs_dir / "rock.jpg",
        "jonah_crab": refs_dir / "jonah.jpg",
    }
    _write_template(refs["european_green_crab"], (45, 55, 50))
    _write_template(refs["native_rock_crab"], (65, 95, 160), vertical=True)
    _write_template(refs["jonah_crab"], (80, 120, 190))

    background = np.full((180, 260, 3), (130, 165, 145), dtype=np.uint8)
    background_path = tmp_path / "pool.jpg"
    assert cv2.imwrite(str(background_path), background)

    config = SyntheticDatasetConfig(
        output_dir=tmp_path / "dataset",
        reference_paths=refs,
        background_paths=[background_path],
        image_count=8,
        image_size=(320, 240),
        seed=11,
        min_crabs=3,
        max_crabs=5,
        crab_long_edge_range=(0.05, 0.38),
        large_crab_fraction=0.25,
        large_crab_long_edge_range=(0.3, 0.44),
        sparse_layout_fraction=0.3,
        full_layout_fraction=0.35,
        even_placement_fraction=0.6,
        even_placement_jitter=0.4,
        max_crab_iou=0.02,
        crab_spacing_fraction=0.015,
        min_crab_box_long_edge_px=12,
        board_long_edge_range=(0.34, 1.9),
        board_roll_range_degrees=(-65.0, 65.0),
        board_tilt_range=(0.2, 0.78),
        board_center_jitter=0.35,
        board_corner_jitter=0.12,
        board_min_visible_fraction=0.9,
        board_min_frame_area_fraction=0.06,
        camera_blur_fraction=0.1,
        camera_blur_sigma_range=(0.05, 0.45),
        jpeg_artifact_fraction=0.15,
        jpeg_quality_range=(85, 100),
        val_fraction=0.25,
        preview_count=4,
    )
    result = generate_synthetic_dataset(config)

    assert result.train_images + result.val_images == 8
    assert result.data_yaml.exists()
    assert result.preview_image and result.preview_image.exists()
    assert (config.output_dir / "classes.txt").read_text(encoding="utf-8").splitlines() == list(CRAB_CLASS_NAMES)
    assert "names:" in result.data_yaml.read_text(encoding="utf-8")
    manifest = json.loads((config.output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["large_crab_fraction"] == 0.25
    assert manifest["large_crab_long_edge_range"] == [0.3, 0.44]
    assert manifest["sparse_layout_fraction"] == 0.3
    assert manifest["full_layout_fraction"] == 0.35
    assert manifest["even_placement_fraction"] == 0.6
    assert manifest["even_placement_jitter"] == 0.4
    assert manifest["max_crab_iou"] == 0.02
    assert manifest["crab_spacing_fraction"] == 0.015
    assert manifest["min_crab_box_long_edge_px"] == 12
    assert manifest["board_long_edge_range"] == [0.34, 1.9]
    assert manifest["board_roll_range_degrees"] == [-65.0, 65.0]
    assert manifest["board_tilt_range"] == [0.2, 0.78]
    assert manifest["board_min_visible_fraction"] == 0.9
    assert manifest["board_min_frame_area_fraction"] == 0.06
    assert manifest["camera_blur_fraction"] == 0.1
    assert manifest["camera_blur_sigma_range"] == [0.05, 0.45]
    assert manifest["jpeg_artifact_fraction"] == 0.15
    assert manifest["jpeg_quality_range"] == [85, 100]

    label_paths = list((config.output_dir / "labels").rglob("*.txt"))
    assert len(label_paths) == 8
    non_empty = [path for path in label_paths if path.read_text(encoding="utf-8").strip()]
    assert non_empty
    for path in non_empty:
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            assert len(parts) == 5
            class_id = int(parts[0])
            assert 0 <= class_id < len(CRAB_CLASS_NAMES)
            values = [float(value) for value in parts[1:]]
            assert all(0.0 <= value <= 1.0 for value in values)
            assert values[2] > 0.0
            assert values[3] > 0.0
