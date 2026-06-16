import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from triton_analysis.crab.plane_dataset import (
    BoardPlaneAnnotation,
    PlaneProjectedDatasetConfig,
    generate_plane_projected_dataset,
    load_board_plane_annotations,
    save_board_plane_annotations,
)
from triton_analysis.crab.synthetic import CRAB_CLASS_NAMES


pytestmark = pytest.mark.vision


def _write_template(path: Path, color: tuple[int, int, int], *, vertical: bool = False) -> None:
    image = np.full((140, 100, 3), 246, dtype=np.uint8)
    axes = (25, 40) if vertical else (38, 24)
    cv2.ellipse(image, (50, 70), axes, 0, 0, 360, color, thickness=-1, lineType=cv2.LINE_AA)
    cv2.line(image, (24, 48), (8, 28), color, 5, cv2.LINE_AA)
    cv2.line(image, (76, 48), (92, 28), color, 5, cv2.LINE_AA)
    cv2.line(image, (24, 92), (8, 112), color, 5, cv2.LINE_AA)
    cv2.line(image, (76, 92), (92, 112), color, 5, cv2.LINE_AA)
    assert cv2.imwrite(str(path), image)


def _write_base_board(path: Path) -> None:
    image = np.full((300, 420, 3), (120, 160, 145), dtype=np.uint8)
    board = np.array([[70, 35], [365, 58], [335, 260], [45, 240]], dtype=np.int32)
    cv2.fillConvexPoly(image, board, (238, 240, 236), lineType=cv2.LINE_AA)
    for y in range(55, 250, 14):
        cv2.line(image, (55, y), (350, y + 18), (215, 218, 214), 1, cv2.LINE_AA)
    assert cv2.imwrite(str(path), image)


def test_plane_annotations_round_trip_relative_to_workspace_data(tmp_path: Path):
    data_root = tmp_path / "Workspace" / "data"
    image = data_root / "base images" / "empty.png"
    image.parent.mkdir(parents=True)
    _write_base_board(image)
    annotation = BoardPlaneAnnotation(
        image_path=image,
        quad_xy=((70, 35), (365, 58), (335, 260), (45, 240)),
        image_size=(420, 300),
        label="empty.png",
    )

    path = data_root / "board_plane_annotations.json"
    save_board_plane_annotations(path, [annotation], image_root=data_root)
    loaded = load_board_plane_annotations(path, image_root=data_root)

    assert len(loaded) == 1
    assert loaded[0].image_path == image
    assert loaded[0].quad_xy == annotation.quad_xy
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["annotations"][0]["image"] == str(Path("base images") / "empty.png")


def test_generate_plane_projected_dataset_writes_yolo_structure(tmp_path: Path):
    refs = tmp_path / "refs"
    refs.mkdir()
    template_paths = {
        "european_green_crab": [refs / "eurogreen_real.png", refs / "European Green Crab Image.jpg"],
        "native_rock_crab": [refs / "rock_real.png"],
        "jonah_crab": [refs / "jonah_real.png"],
    }
    _write_template(template_paths["european_green_crab"][0], (40, 58, 55))
    _write_template(template_paths["european_green_crab"][1], (55, 72, 70), vertical=True)
    _write_template(template_paths["native_rock_crab"][0], (70, 95, 150), vertical=True)
    _write_template(template_paths["jonah_crab"][0], (85, 120, 185))

    base = tmp_path / "base" / "empty_board.jpg"
    base.parent.mkdir()
    _write_base_board(base)
    annotation = BoardPlaneAnnotation(
        image_path=base,
        quad_xy=((70, 35), (365, 58), (335, 260), (45, 240)),
        image_size=(420, 300),
        label=base.name,
    )
    config = PlaneProjectedDatasetConfig(
        output_dir=tmp_path / "dataset",
        annotations=[annotation],
        template_paths=template_paths,
        image_count=6,
        val_fraction=0.34,
        seed=123,
        board_size=420,
        min_crabs=3,
        max_crabs=5,
        crab_long_edge_range=(0.08, 0.22),
        large_crab_fraction=0.1,
        large_crab_long_edge_range=(0.18, 0.28),
        min_crab_box_long_edge_px=8,
        jpeg_artifact_fraction=0.0,
        preview_count=3,
    )

    result = generate_plane_projected_dataset(config)

    assert result.train_images + result.val_images == 6
    assert result.data_yaml.exists()
    assert result.preview_image and result.preview_image.exists()
    assert (config.output_dir / "classes.txt").read_text(encoding="utf-8").splitlines() == list(CRAB_CLASS_NAMES)

    manifest = json.loads((config.output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["generator"] == "plane_projected_crab_dataset"
    assert manifest["annotation_count"] == 1
    assert len(manifest["template_paths"]["european_green_crab"]) == 2
    assert manifest["crab_color_jitter_strength"] == 0.35
    assert manifest["paper_alpha_scale"] == 0.0
    assert manifest["scene_retry_count"] == 3

    label_paths = list((config.output_dir / "labels").rglob("*.txt"))
    image_paths = list((config.output_dir / "images").rglob("*.jpg"))
    assert len(label_paths) == 6
    assert len(image_paths) == 6
    assert any(path.read_text(encoding="utf-8").strip() for path in label_paths)
    for path in label_paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            assert len(parts) == 5
            assert 0 <= int(parts[0]) < len(CRAB_CLASS_NAMES)
            values = [float(value) for value in parts[1:]]
            assert all(0.0 <= value <= 1.0 for value in values)
            assert values[2] > 0.0
            assert values[3] > 0.0
