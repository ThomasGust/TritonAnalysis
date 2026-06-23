"""Tests for crab reference-data extraction and counter reference discovery."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from triton_analysis.crab import counter
from triton_analysis.crab.counter import (
    discover_crab_board_reference_paths,
    discover_crab_classification_reference_paths,
    discover_crab_detector_reference_paths,
    _diversify_detector_references,
    _reference_confidence_key,
    _sort_by_reference_confidence,
)
from triton_analysis.crab.plane_dataset import discover_default_crab_template_paths
from tools.crab_extract_references import ExtractionSummary, extract_from_result_file


pytestmark = pytest.mark.vision


def _write_crab(path: Path, size: tuple[int, int], color: tuple[int, int, int]) -> None:
    width, height = size
    image = np.full((height, width, 3), 235, dtype=np.uint8)
    cv2.ellipse(image, (width // 2, height // 2), (width // 3, height // 4), 0, 0, 360, color, -1, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), image)


def _build_run(tmp_path: Path) -> Path:
    run = tmp_path / "run"
    preprocess = run / "preprocess"
    source = tmp_path / "incoming" / "Arm_Camera_20260101-000000-000.jpg"
    processed = preprocess / "Arm_Camera_20260101-000000-000_auto_homography.png"
    _write_crab(source, (640, 480), (70, 80, 70))
    _write_crab(processed, (240, 180), (70, 80, 70))

    (preprocess / "Arm_Camera_20260101-000000-000_auto_homography_preprocess.json").write_text(
        json.dumps(
            {
                "mode": "auto_homography",
                "source_image": str(source),
                "processed_image": str(processed),
                "source_size": [640, 480],
                "output_size": [240, 180],
                "ordered_points": [[100, 80], [520, 90], [510, 400], [110, 380]],
                "auto_board_confidence": 0.91,
                "board_outline": {"confidence": 0.91},
            }
        ),
        encoding="utf-8",
    )

    candidates = [
        {"label": "european_green_crab", "bbox": [20, 20, 90, 70], "confidence": 0.92,
         "class_scores": {"european_green_crab": 0.92, "native_rock_crab": 0.2, "jonah_crab": 0.1},
         "accepted_as_target": True},
        {"label": "native_rock_crab", "bbox": [120, 30, 190, 95], "confidence": 0.7,
         "class_scores": {"european_green_crab": 0.2, "native_rock_crab": 0.7, "jonah_crab": 0.2},
         "accepted_as_target": False},
        {"label": "jonah_crab", "bbox": [30, 100, 110, 165], "confidence": 0.65,
         "class_scores": {"european_green_crab": 0.1, "native_rock_crab": 0.2, "jonah_crab": 0.65},
         "accepted_as_target": False},
        {"label": "uncertain", "bbox": [150, 110, 215, 170], "confidence": 0.4,
         "class_scores": {"european_green_crab": 0.4, "native_rock_crab": 0.4, "jonah_crab": 0.3},
         "accepted_as_target": False},
    ]
    result_json = run / "Arm_Camera_20260101-000000-000_auto_homography_crab_count.json"
    result_json.write_text(
        json.dumps({"image_path": str(processed), "image_size": [240, 180], "candidates": candidates}),
        encoding="utf-8",
    )
    return result_json


def test_extract_builds_all_three_reference_kinds(tmp_path: Path):
    result_json = _build_run(tmp_path)
    dest = tmp_path / "references"
    summary = ExtractionSummary()
    extract_from_result_file(
        result_json,
        dest,
        min_confidence=0.0,
        accepted_only=False,
        crop_pad_fraction=0.08,
        board_margin_fraction=0.06,
        summary=summary,
    )

    assert summary.classification["european_green_crab"] == 1
    assert summary.classification["native_rock_crab"] == 1
    assert summary.classification["jonah_crab"] == 1
    assert summary.detector == 4  # every single-crab box, class-agnostic
    assert summary.board == 1

    egc = list((dest / "classification" / "european_green_crab").glob("*.png"))
    assert len(egc) == 1
    assert egc[0].name.startswith("conf092")  # confidence-first naming
    # uncertain is not a real species bucket
    assert not (dest / "classification" / "uncertain").exists()
    board_crops = list((dest / "board").glob("*.png"))
    assert len(board_crops) == 1 and board_crops[0].name.startswith("conf091_board")


def test_min_confidence_filters_classification_and_detector(tmp_path: Path):
    result_json = _build_run(tmp_path)
    dest = tmp_path / "references"
    summary = ExtractionSummary()
    extract_from_result_file(
        result_json,
        dest,
        min_confidence=0.8,
        accepted_only=False,
        crop_pad_fraction=0.08,
        board_margin_fraction=0.06,
        summary=summary,
    )
    # Only the 0.92 EGC clears the bar.
    assert summary.detector == 1
    assert summary.classification["european_green_crab"] == 1
    assert summary.classification["native_rock_crab"] == 0


def test_reference_confidence_sort_and_detector_diversity():
    paths = [
        Path("conf060_european_green_crab_a.png"),
        Path("conf096_european_green_crab_b.png"),
        Path("conf093_native_rock_crab_c.png"),
        Path("conf088_jonah_crab_d.png"),
        Path("conf090_european_green_crab_e.png"),
    ]
    ranked = _sort_by_reference_confidence(paths)
    assert _reference_confidence_key(ranked[0]) == 0.96
    assert [_reference_confidence_key(p) for p in ranked] == sorted(
        [_reference_confidence_key(p) for p in paths], reverse=True
    )
    # Round-robin must not return three green crabs before the first rock/jonah.
    diversified = _diversify_detector_references(ranked)
    species_order = [counter._reference_species_key(p) for p in diversified]
    assert species_order[:3] == ["european_green_crab", "native_rock_crab", "jonah_crab"]


def test_discovery_reads_workspace_reference_root(tmp_path: Path, monkeypatch):
    # Isolate from the repo-bundled references so the assertions are deterministic.
    monkeypatch.setattr(counter, "CRAB_REFERENCE_ROOT", tmp_path / "empty_repo_refs")
    workspace = tmp_path / "ws"
    base = workspace / "data" / "crab" / "references"
    _write_crab(base / "classification" / "european_green_crab" / "conf090_x.png", (40, 30), (70, 80, 70))
    _write_crab(base / "detector" / "conf090_european_green_crab_x.png", (40, 30), (70, 80, 70))
    _write_crab(base / "detector" / "conf080_native_rock_crab_y.png", (40, 30), (60, 90, 150))
    _write_crab(base / "board" / "conf088_board_x.png", (80, 60), (200, 200, 200))

    classification = discover_crab_classification_reference_paths(workspace_root=workspace)
    assert len(classification["european_green_crab"]) == 1

    detector = discover_crab_detector_reference_paths(workspace_root=workspace)
    assert len(detector) == 2

    board = discover_crab_board_reference_paths(workspace_root=workspace)
    assert any(p.name == "conf088_board_x.png" for p in board)


def test_reference_root_does_not_pollute_dataset_generator(tmp_path: Path):
    # The synthetic/plane generator must never see counter-only reference crops.
    templates = discover_default_crab_template_paths(tmp_path)
    for paths in templates.values():
        for path in paths:
            assert "references" not in Path(path).parts, path
