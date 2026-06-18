"""OpenAI-backed counting for printed MATE crab-board images."""

from __future__ import annotations

import base64
import csv
import json
import mimetypes
import os
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np

from triton_analysis.crab.plane_dataset import discover_default_crab_template_paths
from triton_analysis.crab.synthetic import CRAB_CLASS_NAMES, IMAGE_EXTENSIONS
from triton_analysis.workspace import fresh_output_subdir, workspace_paths


TARGET_CLASS = "european_green_crab"
UNCERTAIN_CLASS = "uncertain"
REFERENCE_CLASS_LABELS = {
    "european_green_crab": "European green crab",
    "native_rock_crab": "Native rock crab",
    "jonah_crab": "Jonah crab",
}
CANDIDATE_CLASS_LABELS = {
    **REFERENCE_CLASS_LABELS,
    UNCERTAIN_CLASS: "Uncertain crab",
}
CANDIDATE_CLASSES = tuple(CRAB_CLASS_NAMES) + (UNCERTAIN_CLASS,)
NON_TARGET_CLASSES = tuple(class_name for class_name in CRAB_CLASS_NAMES if class_name != TARGET_CLASS)
DEFAULT_MODEL = os.environ.get("TRITON_ANALYSIS_CRAB_MODEL", "gpt-5.5")
REASONING_EFFORTS = ("none", "low", "medium", "high", "xhigh")
REFERENCE_ATLAS_MAX_EXAMPLES_PER_CLASS = 4
DEFAULT_REASONING_EFFORT = os.environ.get("TRITON_ANALYSIS_CRAB_REASONING_EFFORT", "xhigh").strip().lower() or "xhigh"
DEFAULT_HOMOGRAPHY_MODEL = os.environ.get("TRITON_ANALYSIS_CRAB_HOMOGRAPHY_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
DEFAULT_HOMOGRAPHY_REASONING_EFFORT = (
    os.environ.get("TRITON_ANALYSIS_CRAB_HOMOGRAPHY_REASONING_EFFORT", "xhigh").strip().lower() or "xhigh"
)
DEFAULT_DETECTOR_REASONING_EFFORT = (
    os.environ.get("TRITON_ANALYSIS_CRAB_DETECTOR_REASONING_EFFORT", "low").strip().lower() or "low"
)
BOARD_REFERENCE_IMAGE_ENV = "TRITON_ANALYSIS_CRAB_BOARD_REFERENCE_IMAGES"
BOARD_REFERENCE_MAX_IMAGES = 4
try:
    DEFAULT_TARGET_MATCH_THRESHOLD = float(os.environ.get("TRITON_ANALYSIS_CRAB_TARGET_THRESHOLD", "0.85"))
except ValueError:
    DEFAULT_TARGET_MATCH_THRESHOLD = 0.85
try:
    DEFAULT_TARGET_MARGIN_THRESHOLD = float(os.environ.get("TRITON_ANALYSIS_CRAB_TARGET_MARGIN", "0.15"))
except ValueError:
    DEFAULT_TARGET_MARGIN_THRESHOLD = 0.15


@dataclass(frozen=True)
class CrabDetection:
    """One crab candidate in target-image pixel coordinates."""

    label: str
    bbox: tuple[float, float, float, float]
    confidence: float
    target_match_confidence: float = 0.0
    class_scores: Mapping[str, float] | None = None
    closest_non_target: str = ""
    decision_margin: float = 0.0
    accepted_as_target: bool = False
    notes: str = ""


@dataclass(frozen=True)
class CrabCountResult:
    """Structured crab-counter result."""

    image_path: Path
    image_size: tuple[int, int]
    count: int
    detections: tuple[CrabDetection, ...]
    candidates: tuple[CrabDetection, ...]
    model: str
    reasoning_effort: str
    target_confidence_threshold: float
    target_margin_threshold: float
    analysis_seconds: float = 0.0
    summary: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "image_path": str(self.image_path),
            "image_size": list(self.image_size),
            "count": int(self.count),
            "detections": [_detection_to_dict(detection) for detection in self.detections],
            "candidates": [_detection_to_dict(candidate) for candidate in self.candidates],
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "target_confidence_threshold": round(float(self.target_confidence_threshold), 4),
            "target_margin_threshold": round(float(self.target_margin_threshold), 4),
            "analysis_seconds": round(float(self.analysis_seconds), 3),
            "summary": self.summary,
        }


@dataclass(frozen=True)
class CrabCounterOutputs:
    """Files written for one crab-counter run."""

    result: CrabCountResult
    output_dir: Path
    result_json: Path
    annotated_image: Path
    artifact_manifest: Path | None = None


@dataclass(frozen=True)
class CrabBenchmarkOutputs:
    """Files written for a reasoning-effort benchmark run."""

    output_dir: Path
    summary_json: Path
    summary_csv: Path
    runs: tuple[CrabCounterOutputs, ...]
    artifact_manifest: Path | None = None


@dataclass(frozen=True)
class CrabCandidateBox:
    """One detector-stage crab candidate box."""

    candidate_id: int
    bbox: tuple[float, float, float, float]
    confidence: float
    single_crab: bool = True
    notes: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id": int(self.candidate_id),
            "bbox": [round(float(value), 2) for value in self.bbox],
            "confidence": round(float(self.confidence), 4),
            "single_crab": bool(self.single_crab),
            "notes": self.notes,
        }


@dataclass(frozen=True)
class CrabCandidateDetectionResult:
    """Detector-stage output for all printed crab candidates."""

    image_path: Path
    image_size: tuple[int, int]
    candidates: tuple[CrabCandidateBox, ...]
    model: str
    reasoning_effort: str
    analysis_seconds: float = 0.0
    summary: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "image_path": str(self.image_path),
            "image_size": list(self.image_size),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "analysis_seconds": round(float(self.analysis_seconds), 3),
            "summary": self.summary,
        }


@dataclass(frozen=True)
class CrabBoardOutlineResult:
    """OpenAI-selected board corners for one source image."""

    image_path: Path
    image_size: tuple[int, int]
    points: tuple[tuple[float, float], ...]
    confidence: float
    board_visible: bool
    model: str
    reasoning_effort: str
    analysis_seconds: float = 0.0
    notes: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "image_path": str(self.image_path),
            "image_size": list(self.image_size),
            "points": [[round(float(x), 3), round(float(y), 3)] for x, y in self.points],
            "confidence": round(float(self.confidence), 4),
            "board_visible": bool(self.board_visible),
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "analysis_seconds": round(float(self.analysis_seconds), 3),
            "notes": self.notes,
        }


@dataclass(frozen=True)
class CrabPreprocessResult:
    """Files written for one target-image preprocessing operation."""

    mode: str
    source_image: Path
    processed_image: Path
    metadata_json: Path
    source_size: tuple[int, int]
    output_size: tuple[int, int]
    selection_points: tuple[tuple[float, float], ...] = ()
    ordered_points: tuple[tuple[float, float], ...] = ()
    board_confidence: float | None = None
    board_detection_seconds: float | None = None
    board_notes: str = ""
    board_model: str = ""
    board_reasoning_effort: str = ""


@dataclass(frozen=True)
class CrabCounterConfig:
    """Inputs for one OpenAI crab-counter request."""

    image_path: Path
    reference_paths: Mapping[str, Path]
    output_dir: Path
    model: str = DEFAULT_MODEL
    reasoning_effort: str = DEFAULT_REASONING_EFFORT
    target_confidence_threshold: float = DEFAULT_TARGET_MATCH_THRESHOLD
    target_margin_threshold: float = DEFAULT_TARGET_MARGIN_THRESHOLD
    reference_atlas_paths: Mapping[str, Sequence[Path]] | None = None


def discover_counter_reference_paths(workspace_root: str | Path | None = None) -> dict[str, Path | None]:
    """Return the best available reference image for each known crab class."""

    workspace = workspace_paths(workspace_root, create=False)
    discovered = discover_default_crab_template_paths(workspace.root)
    references: dict[str, Path | None] = {}
    for class_name in CRAB_CLASS_NAMES:
        paths = discovered.get(class_name, [])
        references[class_name] = paths[0] if paths else None
    return references


def discover_counter_reference_atlas_paths(workspace_root: str | Path | None = None) -> dict[str, tuple[Path, ...]]:
    """Return all available reference images that can be folded into the model atlas."""

    workspace = workspace_paths(workspace_root, create=False)
    discovered = discover_default_crab_template_paths(workspace.root)
    return {class_name: tuple(paths) for class_name, paths in discovered.items()}


def missing_reference_classes(reference_paths: Mapping[str, Path | None]) -> list[str]:
    """Return classes with no usable reference image."""

    missing: list[str] = []
    for class_name in CRAB_CLASS_NAMES:
        path = reference_paths.get(class_name)
        if path is None or not Path(path).expanduser().is_file():
            missing.append(class_name)
    return missing


def default_output_dir(workspace_root: str | Path | None = None) -> Path:
    """Return a fresh crab-counter output folder."""

    workspace = workspace_paths(workspace_root, create=True)
    return fresh_output_subdir(workspace.results / "crab_counter", "crab_counter", create=True)


def _utc_now_text() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _artifact_stem(stage: str) -> str:
    stem = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in stage.strip().lower())
    return stem.strip("_") or "stage"


def _sanitize_json_value(value: object) -> object:
    if isinstance(value, Path):
        return str(value.expanduser())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _sanitize_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_json_value(item) for item in value]
    if isinstance(value, str):
        if value.startswith("data:") and ";base64," in value:
            header, encoded = value.split(",", 1)
            mime_type = header[5:].split(";", 1)[0] or "application/octet-stream"
            return {
                "omitted": "base64_image_data",
                "mime_type": mime_type,
                "encoded_length": len(encoded),
            }
        return value
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return str(value)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_sanitize_json_value(payload), indent=2) + "\n", encoding="utf-8")


def _record_stage_outputs(
    output_dir: Path,
    stage: str,
    *,
    context: Mapping[str, object] | None = None,
    files: Mapping[str, str | Path | None] | None = None,
) -> Path:
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "run_manifest.json"
    manifest: dict[str, object] = {}
    if manifest_path.is_file():
        try:
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            loaded = {}
        if isinstance(loaded, dict):
            manifest = loaded
    manifest.setdefault("schema_version", 1)
    manifest["output_dir"] = str(output_dir)
    manifest["updated_at"] = _utc_now_text()
    stages = manifest.get("stages")
    if not isinstance(stages, list):
        stages = []
    stage_record: dict[str, object] = {
        "stage": stage,
        "recorded_at": _utc_now_text(),
    }
    if context:
        stage_record["context"] = _sanitize_json_value(dict(context))
    if files:
        stage_record["files"] = {
            name: str(Path(path).expanduser())
            for name, path in files.items()
            if path is not None
        }
    stages = [item for item in stages if not (isinstance(item, Mapping) and item.get("stage") == stage)]
    stages.append(stage_record)
    manifest["stages"] = stages
    _write_json(manifest_path, manifest)
    return manifest_path


def _write_openai_stage_artifacts(
    output_dir: Path,
    stage: str,
    *,
    request_kwargs: Mapping[str, object],
    response: Any,
    context: Mapping[str, object] | None = None,
) -> Path:
    output_dir = Path(output_dir).expanduser()
    artifacts_dir = output_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    stem = _artifact_stem(stage)
    request_path = artifacts_dir / f"{stem}_request.json"
    response_text_path = artifacts_dir / f"{stem}_response.txt"
    response_json_path = artifacts_dir / f"{stem}_response.json"

    _write_json(
        request_path,
        {
            "schema_version": 1,
            "stage": stage,
            "recorded_at": _utc_now_text(),
            "context": dict(context or {}),
            "request": dict(request_kwargs),
        },
    )
    response_text = _response_text(response)
    response_text_path.write_text(response_text.rstrip() + "\n", encoding="utf-8")
    files: dict[str, str | Path | None] = {
        "request_json": request_path,
        "response_text": response_text_path,
    }
    try:
        response_payload = json.loads(response_text)
    except json.JSONDecodeError:
        response_payload = None
    if isinstance(response_payload, Mapping):
        _write_json(response_json_path, response_payload)
        files["response_json"] = response_json_path
    return _record_stage_outputs(output_dir, stage, context=context, files=files)


def discover_crab_board_reference_paths(workspace_root: str | Path | None = None) -> tuple[Path, ...]:
    """Return optional board-only pool reference images for the homography request."""

    candidates: list[Path] = []
    for raw in os.environ.get(BOARD_REFERENCE_IMAGE_ENV, "").split(os.pathsep):
        candidates.extend(_discover_image_paths(Path(raw).expanduser()) if raw.strip() else [])
    workspace = workspace_paths(workspace_root, create=False)
    for rel in (
        Path("data") / "crab board references",
        Path("data") / "board references",
        Path("data") / "blank boards",
    ):
        candidates.extend(_discover_image_paths(workspace.root / rel))
    return tuple(_dedupe_paths(candidates)[:BOARD_REFERENCE_MAX_IMAGES])


def write_reference_atlas(
    reference_paths: Mapping[str, Path | str | None],
    output_path: str | Path,
    *,
    atlas_paths: Mapping[str, Sequence[Path]] | None = None,
) -> Path:
    """Write the generated crab reference atlas used by OpenAI requests."""

    normalized_refs = _normalize_reference_paths(reference_paths)
    merged_refs = _merge_reference_atlas_paths(normalized_refs, atlas_paths)
    missing = [class_name for class_name in CRAB_CLASS_NAMES if not merged_refs.get(class_name)]
    if missing:
        labels = ", ".join(REFERENCE_CLASS_LABELS.get(name, name) for name in missing)
        raise ValueError(f"missing crab reference image(s): {labels}")
    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_image(output, _build_reference_atlas(merged_refs))
    return output


def preprocess_crab_target_image(
    image_path: str | Path,
    output_dir: str | Path,
    *,
    mode: str,
    crop_rect: Sequence[float] | None = None,
    homography_points: Sequence[Sequence[float]] | None = None,
) -> CrabPreprocessResult:
    """Write a cropped or rectified target image for crab analysis."""

    source_path = Path(image_path).expanduser()
    image = _read_image(source_path)
    if image is None:
        raise OSError(f"could not read image: {source_path}")
    height, width = image.shape[:2]
    output_root = Path(output_dir).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)
    mode_name = _normalize_preprocess_mode(mode)
    stem = source_path.stem

    if mode_name == "manual_crop":
        if crop_rect is None:
            raise ValueError("manual crop requires a crop rectangle")
        x0, y0, x1, y1 = _clamp_bbox(crop_rect, (width, height))
        x0_i, y0_i = int(round(x0)), int(round(y0))
        x1_i, y1_i = int(round(x1)), int(round(y1))
        if x1_i - x0_i < 8 or y1_i - y0_i < 8:
            raise ValueError("manual crop is too small")
        processed = image[y0_i:y1_i, x0_i:x1_i].copy()
        source_to_processed = np.array([[1.0, 0.0, -x0_i], [0.0, 1.0, -y0_i], [0.0, 0.0, 1.0]], dtype=np.float64)
        processed_to_source = np.array([[1.0, 0.0, x0_i], [0.0, 1.0, y0_i], [0.0, 0.0, 1.0]], dtype=np.float64)
        output_path = output_root / f"{stem}_manual_crop.png"
        metadata: dict[str, object] = {
            "mode": mode_name,
            "crop_bbox": [x0_i, y0_i, x1_i, y1_i],
        }
    elif mode_name in {"manual_homography", "auto_homography"}:
        if homography_points is None or len(homography_points) != 4:
            raise ValueError(f"{mode_name.replace('_', ' ')} requires four points")
        points = np.array([[float(point[0]), float(point[1])] for point in homography_points], dtype=np.float32)
        ordered = _order_quad_points(points)
        out_w, out_h = _homography_output_size(ordered)
        destination = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)
        source_to_processed = cv2.getPerspectiveTransform(ordered.astype(np.float32), destination)
        processed_to_source = np.linalg.inv(source_to_processed)
        processed = cv2.warpPerspective(image, source_to_processed, (out_w, out_h), flags=cv2.INTER_LINEAR)
        output_path = output_root / f"{stem}_{mode_name}.png"
        point_key = "clicked_points" if mode_name == "manual_homography" else "detected_points"
        metadata = {
            "mode": mode_name,
            point_key: [[round(float(x), 3), round(float(y), 3)] for x, y in points],
            "ordered_points": [[round(float(x), 3), round(float(y), 3)] for x, y in ordered],
        }
    else:
        raise ValueError(f"unsupported preprocessing mode: {mode}")

    _write_image(output_path, processed)
    out_h, out_w = processed.shape[:2]
    metadata.update(
        {
            "source_image": str(source_path),
            "processed_image": str(output_path),
            "source_size": [width, height],
            "output_size": [out_w, out_h],
            "source_to_processed_matrix": _matrix_to_lists(source_to_processed),
            "processed_to_source_matrix": _matrix_to_lists(processed_to_source),
        }
    )
    metadata_json = output_root / f"{stem}_{mode_name}_preprocess.json"
    metadata_json.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    _record_stage_outputs(
        output_root,
        f"{mode_name}_preprocess",
        context=metadata,
        files={
            "processed_image": output_path,
            "metadata_json": metadata_json,
        },
    )
    return CrabPreprocessResult(
        mode=mode_name,
        source_image=source_path,
        processed_image=output_path,
        metadata_json=metadata_json,
        source_size=(width, height),
        output_size=(out_w, out_h),
        selection_points=tuple((float(x), float(y)) for x, y in points) if mode_name.endswith("homography") else (),
        ordered_points=tuple((float(x), float(y)) for x, y in ordered) if mode_name.endswith("homography") else (),
    )


def detect_crab_board_homography(
    image_path: str | Path,
    *,
    model: str = DEFAULT_HOMOGRAPHY_MODEL,
    reasoning_effort: str = DEFAULT_HOMOGRAPHY_REASONING_EFFORT,
    board_reference_paths: Sequence[str | Path] | None = None,
    client: Any | None = None,
    artifact_root: str | Path | None = None,
) -> CrabBoardOutlineResult:
    """Ask OpenAI for the board outline corners in source-image coordinates."""

    source_path = Path(image_path).expanduser()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    image_size = image_dimensions(source_path)
    normalized_effort = _normalize_reasoning_effort(reasoning_effort)
    selected_model = str(model or DEFAULT_HOMOGRAPHY_MODEL)
    start = time.perf_counter()
    references = tuple(Path(path).expanduser() for path in board_reference_paths or ())
    response = _create_openai_homography_response(
        image_path=source_path,
        image_size=image_size,
        model=selected_model,
        reasoning_effort=normalized_effort,
        board_reference_paths=references,
        client=client,
        artifact_root=Path(artifact_root).expanduser() if artifact_root is not None else None,
        artifact_context={
            "image_path": source_path,
            "image_size": list(image_size),
            "board_reference_paths": list(references),
        },
    )
    result = _parse_board_outline_response(
        response,
        image_path=source_path,
        image_size=image_size,
        model=selected_model,
        reasoning_effort=normalized_effort,
    )
    return replace(result, analysis_seconds=time.perf_counter() - start)


def auto_preprocess_crab_target_image(
    image_path: str | Path,
    output_dir: str | Path,
    *,
    model: str = DEFAULT_HOMOGRAPHY_MODEL,
    reasoning_effort: str = DEFAULT_HOMOGRAPHY_REASONING_EFFORT,
    board_reference_paths: Sequence[str | Path] | None = None,
    client: Any | None = None,
    artifact_root: str | Path | None = None,
) -> CrabPreprocessResult:
    """Detect the crab board outline with OpenAI and write a locally rectified target image."""

    outline = detect_crab_board_homography(
        image_path,
        model=model,
        reasoning_effort=reasoning_effort,
        board_reference_paths=board_reference_paths,
        client=client,
        artifact_root=artifact_root if artifact_root is not None else output_dir,
    )
    result = preprocess_crab_target_image(
        image_path,
        output_dir,
        mode="auto_homography",
        homography_points=outline.points,
    )
    metadata = json.loads(result.metadata_json.read_text(encoding="utf-8"))
    metadata["board_outline"] = outline.to_dict()
    metadata["auto_board_confidence"] = round(float(outline.confidence), 4)
    metadata["auto_board_detection_seconds"] = round(float(outline.analysis_seconds), 3)
    metadata["auto_board_notes"] = outline.notes
    result.metadata_json.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return replace(
        result,
        board_confidence=outline.confidence,
        board_detection_seconds=outline.analysis_seconds,
        board_notes=outline.notes,
        board_model=outline.model,
        board_reasoning_effort=outline.reasoning_effort,
    )


def analyze_crab_image(config: CrabCounterConfig, *, client: Any | None = None) -> CrabCounterOutputs:
    """Analyze one target image and write JSON plus an annotated copy."""

    image_path = Path(config.image_path).expanduser()
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    image_size = image_dimensions(image_path)
    normalized_refs = _normalize_reference_paths(config.reference_paths)
    missing = missing_reference_classes(normalized_refs)
    if missing:
        labels = ", ".join(REFERENCE_CLASS_LABELS.get(name, name) for name in missing)
        raise ValueError(f"missing crab reference image(s): {labels}")
    reasoning_effort = _normalize_reasoning_effort(config.reasoning_effort)
    target_confidence_threshold = _clamp01(config.target_confidence_threshold)
    target_margin_threshold = _clamp01(config.target_margin_threshold)
    atlas_refs = _merge_reference_atlas_paths(normalized_refs, config.reference_atlas_paths)
    output_dir = Path(config.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    analysis_start = time.perf_counter()
    response = _create_openai_response(
        image_path=image_path,
        image_size=image_size,
        reference_paths={name: path for name, path in normalized_refs.items() if path is not None},
        reference_atlas_paths=atlas_refs,
        model=str(config.model or DEFAULT_MODEL),
        reasoning_effort=reasoning_effort,
        target_confidence_threshold=target_confidence_threshold,
        target_margin_threshold=target_margin_threshold,
        client=client,
        artifact_root=output_dir,
        artifact_context={
            "image_path": image_path,
            "image_size": list(image_size),
            "reference_paths": normalized_refs,
            "reference_atlas_paths": atlas_refs,
            "target_confidence_threshold": target_confidence_threshold,
            "target_margin_threshold": target_margin_threshold,
        },
    )
    result = _parse_response(
        response,
        image_path=image_path,
        image_size=image_size,
        model=str(config.model or DEFAULT_MODEL),
        reasoning_effort=reasoning_effort,
        target_confidence_threshold=target_confidence_threshold,
        target_margin_threshold=target_margin_threshold,
    )
    result = replace(result, analysis_seconds=time.perf_counter() - analysis_start)
    return _write_crab_counter_outputs(result, output_dir=output_dir, annotated_image_source=image_path)


def detect_crab_candidates(
    image_path: str | Path,
    output_dir: str | Path,
    *,
    model: str = DEFAULT_MODEL,
    reasoning_effort: str = DEFAULT_DETECTOR_REASONING_EFFORT,
    client: Any | None = None,
    artifact_root: str | Path | None = None,
) -> tuple[CrabCandidateDetectionResult, Path]:
    """Detect all visible printed crab candidate boxes without species classification."""

    target_path = Path(image_path).expanduser()
    if not target_path.is_file():
        raise FileNotFoundError(target_path)
    image_size = image_dimensions(target_path)
    selected_model = str(model or DEFAULT_MODEL)
    selected_effort = _normalize_reasoning_effort(reasoning_effort)
    output_root = Path(output_dir).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)
    artifact_parent = Path(artifact_root).expanduser() if artifact_root is not None else output_root
    start = time.perf_counter()
    response = _create_openai_candidate_detection_response(
        image_path=target_path,
        image_size=image_size,
        model=selected_model,
        reasoning_effort=selected_effort,
        client=client,
        artifact_root=artifact_parent,
        artifact_context={
            "image_path": target_path,
            "image_size": list(image_size),
        },
    )
    result = _parse_candidate_detection_response(
        response,
        image_path=target_path,
        image_size=image_size,
        model=selected_model,
        reasoning_effort=selected_effort,
    )
    result = replace(result, analysis_seconds=time.perf_counter() - start)

    detection_json = output_root / f"{target_path.stem}_candidate_boxes.json"
    detection_json.write_text(json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8")
    _record_stage_outputs(
        artifact_parent,
        "candidate_detection_outputs",
        context={
            "image_path": target_path,
            "candidate_count": len(result.candidates),
            "analysis_seconds": round(float(result.analysis_seconds), 3),
        },
        files={
            "detector_json": detection_json,
        },
    )
    return result, detection_json


def analyze_crab_image_pipeline(
    config: CrabCounterConfig,
    *,
    client: Any | None = None,
    progress_callback: Any | None = None,
) -> CrabCounterOutputs:
    """Analyze one target image using detector and crop-classifier stages."""

    output_dir = Path(config.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline_start = time.perf_counter()
    detection_result, detection_json = _run_candidate_detection_stage(config, output_dir, client, progress_callback)
    contact_sheet = output_dir / f"{detection_result.image_path.stem}_candidate_contact_sheet.png"
    _write_candidate_contact_sheet(detection_result.image_path, detection_result.candidates, contact_sheet)
    _record_stage_outputs(
        output_dir,
        "candidate_detection_outputs",
        context={
            "image_path": detection_result.image_path,
            "candidate_count": len(detection_result.candidates),
            "analysis_seconds": round(float(detection_result.analysis_seconds), 3),
        },
        files={
            "detector_json": detection_json,
            "contact_sheet": contact_sheet,
        },
    )

    if not detection_result.candidates:
        result = _empty_count_result(
            detection_result.image_path,
            detection_result.image_size,
            model=str(config.model or DEFAULT_MODEL),
            reasoning_effort=config.reasoning_effort,
            target_confidence_threshold=config.target_confidence_threshold,
            target_margin_threshold=config.target_margin_threshold,
            summary="Detector found no printed crab candidates.",
        )
        result = replace(result, analysis_seconds=time.perf_counter() - pipeline_start)
    else:
        result = _classify_detected_crab_candidates(
            config,
            detection_result,
            contact_sheet,
            client=client,
            progress_callback=progress_callback,
        )
        result = replace(result, analysis_seconds=time.perf_counter() - pipeline_start)

    return _write_crab_counter_outputs(
        result,
        output_dir=output_dir,
        annotated_image_source=detection_result.image_path,
        extra_json={
            "pipeline": {
                "mode": "detect_then_classify_crops",
                "detector_json": str(detection_json),
                "contact_sheet": str(contact_sheet),
                "detector_candidate_count": len(detection_result.candidates),
                "detector_seconds": round(float(detection_result.analysis_seconds), 3),
            }
        },
    )


def benchmark_crab_image_pipeline(
    config: CrabCounterConfig,
    *,
    efforts: Iterable[str] | None = None,
    client: Any | None = None,
    progress_callback: Any | None = None,
) -> CrabBenchmarkOutputs:
    """Benchmark crop-classifier reasoning efforts after one shared detector stage."""

    selected_efforts = tuple(_normalize_reasoning_effort(effort) for effort in (efforts or REASONING_EFFORTS))
    if not selected_efforts:
        raise ValueError("at least one reasoning effort is required")

    output_dir = Path(config.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    detection_result, detection_json = _run_candidate_detection_stage(config, output_dir, client, progress_callback)
    contact_sheet = output_dir / f"{detection_result.image_path.stem}_candidate_contact_sheet.png"
    _write_candidate_contact_sheet(detection_result.image_path, detection_result.candidates, contact_sheet)
    _record_stage_outputs(
        output_dir,
        "candidate_detection_outputs",
        context={
            "image_path": detection_result.image_path,
            "candidate_count": len(detection_result.candidates),
            "analysis_seconds": round(float(detection_result.analysis_seconds), 3),
        },
        files={
            "detector_json": detection_json,
            "contact_sheet": contact_sheet,
        },
    )

    runs: list[CrabCounterOutputs] = []
    total = len(selected_efforts)
    for index, effort in enumerate(selected_efforts, start=1):
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "effort_started",
                    "effort": effort,
                    "index": index,
                    "total": total,
                    "completed": index - 1,
                }
            )
        effort_start = time.perf_counter()
        effort_config = replace(config, reasoning_effort=effort, output_dir=output_dir / effort)
        if detection_result.candidates:
            result = _classify_detected_crab_candidates(
                effort_config,
                detection_result,
                contact_sheet,
                client=client,
                progress_callback=None,
            )
        else:
            result = _empty_count_result(
                detection_result.image_path,
                detection_result.image_size,
                model=str(config.model or DEFAULT_MODEL),
                reasoning_effort=effort,
                target_confidence_threshold=config.target_confidence_threshold,
                target_margin_threshold=config.target_margin_threshold,
                summary="Detector found no printed crab candidates.",
            )
        result = replace(result, analysis_seconds=time.perf_counter() - effort_start)
        run = _write_crab_counter_outputs(
            result,
            output_dir=output_dir / effort,
            annotated_image_source=detection_result.image_path,
            extra_json={
                "pipeline": {
                    "mode": "detect_then_classify_crops",
                    "detector_json": str(detection_json),
                    "contact_sheet": str(contact_sheet),
                    "detector_candidate_count": len(detection_result.candidates),
                    "detector_seconds": round(float(detection_result.analysis_seconds), 3),
                }
            },
        )
        runs.append(run)
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "effort_finished",
                    "effort": effort,
                    "index": index,
                    "total": total,
                    "completed": index,
                    "analysis_seconds": run.result.analysis_seconds,
                }
            )

    summary = {
        "image_path": str(Path(config.image_path).expanduser()),
        "model": str(config.model or DEFAULT_MODEL),
        "artifact_manifest": str(output_dir / "run_manifest.json"),
        "pipeline": {
            "mode": "detect_then_classify_crops",
            "detector_json": str(detection_json),
            "contact_sheet": str(contact_sheet),
            "detector_candidate_count": len(detection_result.candidates),
            "detector_seconds": round(float(detection_result.analysis_seconds), 3),
        },
        "target_confidence_threshold": _clamp01(config.target_confidence_threshold),
        "target_margin_threshold": _clamp01(config.target_margin_threshold),
        "runs": [_benchmark_row(run) for run in runs],
    }
    summary_json = output_dir / "reasoning_effort_benchmark.json"
    summary_csv = output_dir / "reasoning_effort_benchmark.csv"
    summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _write_benchmark_csv(summary_csv, runs)
    manifest_path = _record_stage_outputs(
        output_dir,
        "benchmark_summary",
        context={
            "image_path": Path(config.image_path).expanduser(),
            "model": str(config.model or DEFAULT_MODEL),
            "efforts": list(selected_efforts),
            "flow": "detect_then_classify_crops",
        },
        files={
            "summary_json": summary_json,
            "summary_csv": summary_csv,
        },
    )
    return CrabBenchmarkOutputs(
        output_dir=output_dir,
        summary_json=summary_json,
        summary_csv=summary_csv,
        runs=tuple(runs),
        artifact_manifest=manifest_path,
    )


def benchmark_crab_image(
    config: CrabCounterConfig,
    *,
    efforts: Iterable[str] | None = None,
    client: Any | None = None,
    progress_callback: Any | None = None,
) -> CrabBenchmarkOutputs:
    """Run the same image/reference setup across reasoning efforts and save a summary."""

    selected_efforts = tuple(_normalize_reasoning_effort(effort) for effort in (efforts or REASONING_EFFORTS))
    if not selected_efforts:
        raise ValueError("at least one reasoning effort is required")

    output_dir = Path(config.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    runs: list[CrabCounterOutputs] = []
    total = len(selected_efforts)
    for index, effort in enumerate(selected_efforts, start=1):
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "effort_started",
                    "effort": effort,
                    "index": index,
                    "total": total,
                    "completed": index - 1,
                }
            )
        run = analyze_crab_image(
            CrabCounterConfig(
                image_path=config.image_path,
                reference_paths=config.reference_paths,
                output_dir=output_dir / effort,
                model=config.model,
                reasoning_effort=effort,
                target_confidence_threshold=config.target_confidence_threshold,
                target_margin_threshold=config.target_margin_threshold,
                reference_atlas_paths=config.reference_atlas_paths,
            ),
            client=client,
        )
        runs.append(run)
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "effort_finished",
                    "effort": effort,
                    "index": index,
                    "total": total,
                    "completed": index,
                    "analysis_seconds": run.result.analysis_seconds,
                }
            )

    summary = {
        "image_path": str(Path(config.image_path).expanduser()),
        "model": str(config.model or DEFAULT_MODEL),
        "artifact_manifest": str(output_dir / "run_manifest.json"),
        "target_confidence_threshold": _clamp01(config.target_confidence_threshold),
        "target_margin_threshold": _clamp01(config.target_margin_threshold),
        "runs": [_benchmark_row(run) for run in runs],
    }
    summary_json = output_dir / "reasoning_effort_benchmark.json"
    summary_csv = output_dir / "reasoning_effort_benchmark.csv"
    summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _write_benchmark_csv(summary_csv, runs)
    manifest_path = _record_stage_outputs(
        output_dir,
        "benchmark_summary",
        context={
            "image_path": Path(config.image_path).expanduser(),
            "model": str(config.model or DEFAULT_MODEL),
            "efforts": list(selected_efforts),
            "flow": "single_request",
        },
        files={
            "summary_json": summary_json,
            "summary_csv": summary_csv,
        },
    )
    return CrabBenchmarkOutputs(
        output_dir=output_dir,
        summary_json=summary_json,
        summary_csv=summary_csv,
        runs=tuple(runs),
        artifact_manifest=manifest_path,
    )


def draw_crab_count_result(image_path: str | Path, result: CrabCountResult, output_path: str | Path) -> Path:
    """Draw European green crab boxes and a count label onto an image."""

    source = _read_image(Path(image_path).expanduser())
    if source is None:
        raise OSError(f"could not read image: {image_path}")
    height, width = source.shape[:2]
    out = source.copy()
    green = (70, 245, 95)
    black = (0, 0, 0)
    for index, detection in enumerate(result.detections, start=1):
        x0, y0, x1, y1 = _clamp_bbox(detection.bbox, (width, height))
        p0 = (int(round(x0)), int(round(y0)))
        p1 = (int(round(x1)), int(round(y1)))
        cv2.rectangle(out, p0, p1, black, 5, lineType=cv2.LINE_AA)
        cv2.rectangle(out, p0, p1, green, 2, lineType=cv2.LINE_AA)
        label = f"EGC {index}"
        label_origin = (p0[0], max(24, p0[1] - 8))
        _draw_label(out, label, label_origin, green, black, scale=0.58, thickness=1)

    count_text = f"European green crabs: {result.count}"
    count_origin = _choose_count_label_origin(
        count_text,
        (width, height),
        [detection.bbox for detection in result.detections],
        scale=0.78,
        thickness=2,
    )
    _draw_label(out, count_text, count_origin, green, black, scale=0.78, thickness=2)
    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_image(output, out)
    return output


def transform_crab_count_result(
    result: CrabCountResult,
    processed_to_source_matrix: Sequence[Sequence[float]],
    *,
    source_image_path: str | Path,
    source_size: tuple[int, int],
) -> CrabCountResult:
    """Map result boxes from a processed target image back into source-image coordinates."""

    matrix = np.array(processed_to_source_matrix, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError("processed_to_source_matrix must be 3x3")
    transformed_candidates = tuple(
        replace(detection, bbox=_transform_bbox(detection.bbox, matrix, source_size))
        for detection in result.candidates
    )
    transformed_detections = tuple(
        replace(detection, bbox=_transform_bbox(detection.bbox, matrix, source_size))
        for detection in result.detections
    )
    return replace(
        result,
        image_path=Path(source_image_path).expanduser(),
        image_size=source_size,
        detections=transformed_detections,
        candidates=transformed_candidates,
    )


def image_dimensions(path: str | Path) -> tuple[int, int]:
    """Return image dimensions as ``(width, height)``."""

    image = _read_image(Path(path).expanduser())
    if image is None:
        raise OSError(f"could not read image: {path}")
    height, width = image.shape[:2]
    return int(width), int(height)


def result_from_payload(
    payload: Mapping[str, Any],
    *,
    image_path: str | Path,
    image_size: tuple[int, int],
    model: str,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    target_confidence_threshold: float = DEFAULT_TARGET_MATCH_THRESHOLD,
    target_margin_threshold: float = DEFAULT_TARGET_MARGIN_THRESHOLD,
) -> CrabCountResult:
    """Validate a model JSON payload into a crab-count result."""

    candidates: list[CrabDetection] = []
    width, height = image_size
    raw_candidates = payload.get("candidates")
    if raw_candidates is None:
        raw_candidates = payload.get("detections", [])
    for raw in raw_candidates:
        if not isinstance(raw, Mapping):
            continue
        label = _normalize_candidate_label(raw.get("label") or raw.get("class") or TARGET_CLASS)
        if label not in CANDIDATE_CLASSES:
            continue
        bbox_values = raw.get("bbox")
        if not isinstance(bbox_values, Sequence) or len(bbox_values) != 4:
            continue
        try:
            bbox = tuple(float(value) for value in bbox_values)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        x0, y0, x1, y1 = _clamp_bbox(bbox, (width, height))
        if x1 <= x0 or y1 <= y0:
            continue
        confidence = _clamp01(raw.get("confidence", 0.0))
        class_scores = _normalize_class_scores(raw.get("class_scores"))
        target_match_confidence = _clamp01(
            raw.get(
                "target_match_confidence",
                class_scores.get(TARGET_CLASS, confidence if label == TARGET_CLASS else 0.0),
            )
        )
        if class_scores.get(TARGET_CLASS, 0.0) == 0.0 and target_match_confidence > 0.0:
            class_scores[TARGET_CLASS] = target_match_confidence
        if class_scores.get(label, 0.0) == 0.0 and label in CRAB_CLASS_NAMES:
            class_scores[label] = max(confidence, class_scores.get(label, 0.0))
        closest_non_target = _normalize_candidate_label(raw.get("closest_non_target") or "")
        if closest_non_target not in NON_TARGET_CLASSES:
            closest_non_target = _closest_non_target(class_scores)
        next_best_score = max((class_scores.get(name, 0.0) for name in NON_TARGET_CLASSES), default=0.0)
        decision_margin = max(-1.0, min(1.0, target_match_confidence - next_best_score))
        notes = str(raw.get("notes") or "")
        candidates.append(
            CrabDetection(
                label=label,
                bbox=(x0, y0, x1, y1),
                confidence=confidence,
                target_match_confidence=target_match_confidence,
                class_scores=class_scores,
                closest_non_target=closest_non_target,
                decision_margin=decision_margin,
                accepted_as_target=bool(raw.get("accepted_as_target", label == TARGET_CLASS)),
                notes=notes,
            )
        )

    candidates.sort(key=lambda detection: (detection.bbox[1], detection.bbox[0]))
    target_confidence_threshold = _clamp01(target_confidence_threshold)
    target_margin_threshold = _clamp01(target_margin_threshold)
    detections: list[CrabDetection] = []
    updated_candidates: list[CrabDetection] = []
    for candidate in candidates:
        accepted_as_target = (
            candidate.accepted_as_target
            and candidate.label == TARGET_CLASS
            and candidate.target_match_confidence >= target_confidence_threshold
            and candidate.decision_margin >= target_margin_threshold
        )
        updated = replace(candidate, accepted_as_target=accepted_as_target)
        updated_candidates.append(updated)
        if accepted_as_target:
            detections.append(updated)
    detections.sort(key=lambda detection: (detection.bbox[1], detection.bbox[0]))
    summary = str(payload.get("summary") or "")
    return CrabCountResult(
        image_path=Path(image_path).expanduser(),
        image_size=image_size,
        count=len(detections),
        detections=tuple(detections),
        candidates=tuple(updated_candidates),
        model=str(model),
        reasoning_effort=_normalize_reasoning_effort(reasoning_effort),
        target_confidence_threshold=target_confidence_threshold,
        target_margin_threshold=target_margin_threshold,
        summary=summary,
    )


def _run_candidate_detection_stage(
    config: CrabCounterConfig,
    output_dir: Path,
    client: Any | None,
    progress_callback: Any | None,
) -> tuple[CrabCandidateDetectionResult, Path]:
    if progress_callback is not None:
        progress_callback({"event": "candidate_detection_started", "effort": DEFAULT_DETECTOR_REASONING_EFFORT})
    result, detection_json = detect_crab_candidates(
        config.image_path,
        output_dir / "pipeline",
        model=str(config.model or DEFAULT_MODEL),
        reasoning_effort=DEFAULT_DETECTOR_REASONING_EFFORT,
        client=client,
        artifact_root=output_dir,
    )
    if progress_callback is not None:
        progress_callback(
            {
                "event": "candidate_detection_finished",
                "count": len(result.candidates),
                "analysis_seconds": result.analysis_seconds,
                "detection_json": str(detection_json),
                "detection_result": result,
            }
        )
    return result, detection_json


def _classify_detected_crab_candidates(
    config: CrabCounterConfig,
    detection_result: CrabCandidateDetectionResult,
    contact_sheet: Path,
    *,
    client: Any | None,
    progress_callback: Any | None,
) -> CrabCountResult:
    normalized_refs = _normalize_reference_paths(config.reference_paths)
    missing = missing_reference_classes(normalized_refs)
    if missing:
        labels = ", ".join(REFERENCE_CLASS_LABELS.get(name, name) for name in missing)
        raise ValueError(f"missing crab reference image(s): {labels}")
    atlas_refs = _merge_reference_atlas_paths(normalized_refs, config.reference_atlas_paths)
    reasoning_effort = _normalize_reasoning_effort(config.reasoning_effort)
    target_confidence_threshold = _clamp01(config.target_confidence_threshold)
    target_margin_threshold = _clamp01(config.target_margin_threshold)
    output_dir = Path(config.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    if progress_callback is not None:
        progress_callback(
            {
                "event": "candidate_classification_started",
                "effort": reasoning_effort,
                "candidate_count": len(detection_result.candidates),
            }
        )
    start = time.perf_counter()
    response = _create_openai_candidate_classification_response(
        image_size=detection_result.image_size,
        candidates=detection_result.candidates,
        contact_sheet=contact_sheet,
        reference_atlas_paths=atlas_refs,
        model=str(config.model or DEFAULT_MODEL),
        reasoning_effort=reasoning_effort,
        target_confidence_threshold=target_confidence_threshold,
        target_margin_threshold=target_margin_threshold,
        client=client,
        artifact_root=output_dir,
        artifact_context={
            "image_path": detection_result.image_path,
            "image_size": list(detection_result.image_size),
            "candidate_count": len(detection_result.candidates),
            "candidates": [candidate.to_dict() for candidate in detection_result.candidates],
            "contact_sheet": contact_sheet,
            "reference_paths": normalized_refs,
            "reference_atlas_paths": atlas_refs,
            "target_confidence_threshold": target_confidence_threshold,
            "target_margin_threshold": target_margin_threshold,
        },
    )
    result = _parse_candidate_classification_response(
        response,
        detection_result=detection_result,
        model=str(config.model or DEFAULT_MODEL),
        reasoning_effort=reasoning_effort,
        target_confidence_threshold=target_confidence_threshold,
        target_margin_threshold=target_margin_threshold,
    )
    result = replace(result, analysis_seconds=time.perf_counter() - start)
    if progress_callback is not None:
        progress_callback(
            {
                "event": "candidate_classification_finished",
                "effort": reasoning_effort,
                "candidate_count": len(detection_result.candidates),
                "analysis_seconds": result.analysis_seconds,
            }
        )
    return result


def _empty_count_result(
    image_path: str | Path,
    image_size: tuple[int, int],
    *,
    model: str,
    reasoning_effort: str,
    target_confidence_threshold: float,
    target_margin_threshold: float,
    summary: str,
) -> CrabCountResult:
    return CrabCountResult(
        image_path=Path(image_path).expanduser(),
        image_size=image_size,
        count=0,
        detections=(),
        candidates=(),
        model=str(model),
        reasoning_effort=_normalize_reasoning_effort(reasoning_effort),
        target_confidence_threshold=_clamp01(target_confidence_threshold),
        target_margin_threshold=_clamp01(target_margin_threshold),
        summary=summary,
    )


def _write_crab_counter_outputs(
    result: CrabCountResult,
    *,
    output_dir: Path,
    annotated_image_source: str | Path,
    extra_json: Mapping[str, object] | None = None,
) -> CrabCounterOutputs:
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = Path(result.image_path).expanduser()
    result_json = output_dir / f"{image_path.stem}_crab_count.json"
    annotated_image = output_dir / f"{image_path.stem}_crab_count_annotated.png"
    manifest_path = output_dir / "run_manifest.json"
    payload = result.to_dict()
    payload["artifact_manifest"] = str(manifest_path)
    if extra_json:
        payload.update(dict(extra_json))
    result_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    draw_crab_count_result(annotated_image_source, result, annotated_image)
    _record_stage_outputs(
        output_dir,
        "final_outputs",
        context={
            "image_path": image_path,
            "count": int(result.count),
            "candidate_count": len(result.candidates),
            "accepted_count": len(result.detections),
            "analysis_seconds": round(float(result.analysis_seconds), 3),
        },
        files={
            "result_json": result_json,
            "annotated_image": annotated_image,
        },
    )
    return CrabCounterOutputs(
        result=result,
        output_dir=output_dir,
        result_json=result_json,
        annotated_image=annotated_image,
        artifact_manifest=manifest_path,
    )


def _write_candidate_contact_sheet(
    image_path: str | Path,
    candidates: Sequence[CrabCandidateBox],
    output_path: str | Path,
) -> Path:
    image = _read_image(Path(image_path).expanduser())
    if image is None:
        raise OSError(f"could not read image: {image_path}")
    height, width = image.shape[:2]
    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    if not candidates:
        sheet = np.full((180, 360, 3), 245, dtype=np.uint8)
        cv2.putText(sheet, "No crab candidates detected", (22, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 40, 40), 2, cv2.LINE_AA)
        _write_image(output, sheet)
        return output

    columns = min(4, max(1, len(candidates)))
    rows = int(np.ceil(len(candidates) / float(columns)))
    cell_w = 220
    cell_h = 230
    header_h = 34
    sheet = np.full((header_h + rows * cell_h, columns * cell_w, 3), 246, dtype=np.uint8)
    cv2.putText(sheet, "Crab candidate crops", (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (35, 35, 35), 2, cv2.LINE_AA)
    for index, candidate in enumerate(candidates):
        row = index // columns
        col = index % columns
        x0 = col * cell_w
        y0 = header_h + row * cell_h
        cv2.rectangle(sheet, (x0, y0), (x0 + cell_w - 1, y0 + cell_h - 1), (205, 205, 205), 1)
        crop_bbox = _expanded_bbox(candidate.bbox, (width, height), fraction=0.18, min_pad=8.0)
        bx0, by0, bx1, by1 = (int(round(value)) for value in crop_bbox)
        crop = image[by0:by1, bx0:bx1].copy()
        if crop.size == 0:
            continue
        fitted = _fit_image_to_box(crop, cell_w - 24, cell_h - 58)
        fh, fw = fitted.shape[:2]
        px = x0 + (cell_w - fw) // 2
        py = y0 + 36 + (cell_h - 58 - fh) // 2
        sheet[py : py + fh, px : px + fw] = fitted
        label = f"Candidate {candidate.candidate_id}"
        cv2.putText(sheet, label, (x0 + 10, y0 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (20, 20, 20), 2, cv2.LINE_AA)
        bbox_text = f"[{candidate.bbox[0]:.0f},{candidate.bbox[1]:.0f},{candidate.bbox[2]:.0f},{candidate.bbox[3]:.0f}]"
        cv2.putText(sheet, bbox_text, (x0 + 10, y0 + cell_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (70, 70, 70), 1, cv2.LINE_AA)
    _write_image(output, sheet)
    return output


def _parse_candidate_detection_response(
    response: Any,
    *,
    image_path: Path,
    image_size: tuple[int, int],
    model: str,
    reasoning_effort: str,
) -> CrabCandidateDetectionResult:
    text = _response_text(response)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"OpenAI detector response was not valid JSON: {text[:500]}") from exc
    if not isinstance(payload, Mapping):
        raise RuntimeError("OpenAI detector response JSON was not an object.")
    raw_candidates = payload.get("candidates", [])
    candidates: list[CrabCandidateBox] = []
    width, height = image_size
    for raw in raw_candidates if isinstance(raw_candidates, Sequence) else []:
        if not isinstance(raw, Mapping):
            continue
        bbox_values = raw.get("bbox")
        if not isinstance(bbox_values, Sequence) or len(bbox_values) != 4:
            continue
        try:
            bbox = tuple(float(value) for value in bbox_values)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        x0, y0, x1, y1 = _clamp_bbox(bbox, (width, height))
        if x1 - x0 < 3.0 or y1 - y0 < 3.0:
            continue
        single_crab = bool(raw.get("single_crab", True))
        if not single_crab:
            continue
        candidates.append(
            CrabCandidateBox(
                candidate_id=0,
                bbox=(x0, y0, x1, y1),
                confidence=_clamp01(raw.get("confidence", 0.0)),
                single_crab=single_crab,
                notes=str(raw.get("notes") or ""),
            )
        )
    candidates.sort(key=lambda candidate: (candidate.bbox[1], candidate.bbox[0]))
    numbered = tuple(replace(candidate, candidate_id=index) for index, candidate in enumerate(candidates, start=1))
    return CrabCandidateDetectionResult(
        image_path=image_path,
        image_size=image_size,
        candidates=numbered,
        model=str(model),
        reasoning_effort=_normalize_reasoning_effort(reasoning_effort),
        summary=str(payload.get("summary") or ""),
    )


def _parse_candidate_classification_response(
    response: Any,
    *,
    detection_result: CrabCandidateDetectionResult,
    model: str,
    reasoning_effort: str,
    target_confidence_threshold: float,
    target_margin_threshold: float,
) -> CrabCountResult:
    text = _response_text(response)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"OpenAI crop-classifier response was not valid JSON: {text[:500]}") from exc
    if not isinstance(payload, Mapping):
        raise RuntimeError("OpenAI crop-classifier response JSON was not an object.")
    raw_items = payload.get("classifications", [])
    by_id: dict[int, Mapping[str, object]] = {}
    for raw in raw_items if isinstance(raw_items, Sequence) else []:
        if not isinstance(raw, Mapping):
            continue
        try:
            candidate_id = int(raw.get("candidate_id", 0))
        except (TypeError, ValueError):
            continue
        by_id[candidate_id] = raw

    payload_candidates: list[dict[str, object]] = []
    for candidate in detection_result.candidates:
        raw = by_id.get(candidate.candidate_id, {})
        payload_candidates.append(
            {
                "label": raw.get("label", UNCERTAIN_CLASS),
                "bbox": list(candidate.bbox),
                "confidence": raw.get("confidence", 0.0),
                "target_match_confidence": raw.get("target_match_confidence", 0.0),
                "class_scores": raw.get("class_scores", {}),
                "closest_non_target": raw.get("closest_non_target", ""),
                "decision_margin": raw.get("decision_margin", 0.0),
                "accepted_as_target": bool(raw.get("accepted_as_target", False))
                and bool(raw.get("visible_cues_sufficient", True)),
                "notes": raw.get("notes", "missing crop classification"),
            }
        )
    return result_from_payload(
        {
            "candidates": payload_candidates,
            "summary": str(payload.get("summary") or ""),
        },
        image_path=detection_result.image_path,
        image_size=detection_result.image_size,
        model=model,
        reasoning_effort=reasoning_effort,
        target_confidence_threshold=target_confidence_threshold,
        target_margin_threshold=target_margin_threshold,
    )


def _normalize_reference_paths(reference_paths: Mapping[str, Path | str | None]) -> dict[str, Path | None]:
    normalized: dict[str, Path | None] = {}
    for class_name in CRAB_CLASS_NAMES:
        value = reference_paths.get(class_name)
        normalized[class_name] = Path(value).expanduser() if value else None
    return normalized


def _merge_reference_atlas_paths(
    primary_paths: Mapping[str, Path | None],
    atlas_paths: Mapping[str, Sequence[Path]] | None,
) -> dict[str, tuple[Path, ...]]:
    merged: dict[str, list[Path]] = {class_name: [] for class_name in CRAB_CLASS_NAMES}
    for class_name, path in primary_paths.items():
        if path is not None:
            merged[class_name].append(Path(path).expanduser())
    for class_name in CRAB_CLASS_NAMES:
        for path in (atlas_paths or {}).get(class_name, ()):
            merged[class_name].append(Path(path).expanduser())
    return {class_name: tuple(_dedupe_paths(paths)) for class_name, paths in merged.items()}


def _detection_to_dict(detection: CrabDetection) -> dict[str, object]:
    return {
        "label": detection.label,
        "bbox": [round(float(value), 2) for value in detection.bbox],
        "confidence": round(float(detection.confidence), 4),
        "target_match_confidence": round(float(detection.target_match_confidence), 4),
        "class_scores": {
            class_name: round(float((detection.class_scores or {}).get(class_name, 0.0)), 4)
            for class_name in CRAB_CLASS_NAMES
        },
        "closest_non_target": detection.closest_non_target,
        "decision_margin": round(float(detection.decision_margin), 4),
        "accepted_as_target": bool(detection.accepted_as_target),
        "notes": detection.notes,
    }


def _benchmark_row(run: CrabCounterOutputs) -> dict[str, object]:
    result = run.result
    row: dict[str, object] = {
        "reasoning_effort": result.reasoning_effort,
        "analysis_seconds": round(float(result.analysis_seconds), 3),
        "count": int(result.count),
        "candidate_count": len(result.candidates),
        "result_json": str(run.result_json),
        "annotated_image": str(run.annotated_image),
        "summary": result.summary,
    }
    if run.artifact_manifest is not None:
        row["artifact_manifest"] = str(run.artifact_manifest)
    return row


def _write_benchmark_csv(path: Path, runs: Sequence[CrabCounterOutputs]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "reasoning_effort",
        "analysis_seconds",
        "count",
        "candidate_count",
        "result_json",
        "annotated_image",
        "artifact_manifest",
        "summary",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for run in runs:
            writer.writerow(_benchmark_row(run))


def _normalize_candidate_label(value: object) -> str:
    label = str(value or "").strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "green_crab": TARGET_CLASS,
        "european_green": TARGET_CLASS,
        "egc": TARGET_CLASS,
        "rock_crab": "native_rock_crab",
        "native_rock": "native_rock_crab",
        "jonah": "jonah_crab",
        "unknown": UNCERTAIN_CLASS,
        "ambiguous": UNCERTAIN_CLASS,
    }
    return aliases.get(label, label)


def _normalize_class_scores(value: object) -> dict[str, float]:
    scores = {class_name: 0.0 for class_name in CRAB_CLASS_NAMES}
    if not isinstance(value, Mapping):
        return scores
    for raw_name, raw_score in value.items():
        class_name = _normalize_candidate_label(raw_name)
        if class_name in scores:
            scores[class_name] = _clamp01(raw_score)
    return scores


def _closest_non_target(class_scores: Mapping[str, float]) -> str:
    return max(NON_TARGET_CLASSES, key=lambda class_name: float(class_scores.get(class_name, 0.0)))


def _normalize_reasoning_effort(value: object) -> str:
    effort = str(value or DEFAULT_REASONING_EFFORT).strip().lower()
    if effort not in REASONING_EFFORTS:
        allowed = ", ".join(REASONING_EFFORTS)
        raise ValueError(f"unsupported reasoning effort '{effort}'. Use one of: {allowed}")
    return effort


def _clamp01(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = 0.0
    return max(0.0, min(1.0, number))


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
        if key.is_file():
            seen.add(key)
            out.append(path.expanduser())
    return out


def _discover_image_paths(path: Path) -> list[Path]:
    expanded = path.expanduser()
    if expanded.is_file() and expanded.suffix.lower() in IMAGE_EXTENSIONS:
        return [expanded]
    if not expanded.is_dir():
        return []
    return sorted(
        (
            child
            for child in expanded.rglob("*")
            if child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda child: str(child).lower(),
    )


def _normalize_preprocess_mode(value: object) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "auto": "auto_homography",
        "auto_homography": "auto_homography",
        "autohomography": "auto_homography",
        "auto_rectify": "auto_homography",
        "autorectify": "auto_homography",
        "intelligent_homography": "auto_homography",
        "crop": "manual_crop",
        "manualcrop": "manual_crop",
        "homography": "manual_homography",
        "rectify": "manual_homography",
        "manualhomography": "manual_homography",
    }
    return aliases.get(text, text)


def _order_quad_points(points: np.ndarray) -> np.ndarray:
    if points.shape != (4, 2):
        raise ValueError("expected four 2D points")
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).reshape(-1)
    top_left = points[int(np.argmin(sums))]
    bottom_right = points[int(np.argmax(sums))]
    top_right = points[int(np.argmin(diffs))]
    bottom_left = points[int(np.argmax(diffs))]
    ordered = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)
    if abs(float(cv2.contourArea(ordered))) < 100.0:
        raise ValueError("homography points are nearly collinear")
    return ordered


def _homography_output_size(points: np.ndarray) -> tuple[int, int]:
    top_left, top_right, bottom_right, bottom_left = points
    width_a = np.linalg.norm(bottom_right - bottom_left)
    width_b = np.linalg.norm(top_right - top_left)
    height_a = np.linalg.norm(top_right - bottom_right)
    height_b = np.linalg.norm(top_left - bottom_left)
    width = int(round(max(width_a, width_b)))
    height = int(round(max(height_a, height_b)))
    if width < 32 or height < 32:
        raise ValueError("homography output would be too small")
    return width, height


def _matrix_to_lists(matrix: np.ndarray) -> list[list[float]]:
    return [[round(float(value), 6) for value in row] for row in matrix.tolist()]


def _create_openai_response(
    *,
    image_path: Path,
    image_size: tuple[int, int],
    reference_paths: Mapping[str, Path],
    reference_atlas_paths: Mapping[str, Sequence[Path]],
    model: str,
    reasoning_effort: str,
    target_confidence_threshold: float,
    target_margin_threshold: float,
    client: Any | None,
    artifact_root: Path | None = None,
    artifact_context: Mapping[str, object] | None = None,
) -> Any:
    if client is None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - exercised through GUI messaging
            raise RuntimeError("OpenAI Python package is not installed. Run setup_windows.ps1 again.") from exc
        client = OpenAI()
    if not hasattr(client, "responses"):
        raise RuntimeError("OpenAI client does not expose the Responses API. Install openai>=2.0.0.")

    width, height = image_size
    prompt = _build_prompt(
        width=width,
        height=height,
        target_confidence_threshold=target_confidence_threshold,
        target_margin_threshold=target_margin_threshold,
    )
    content: list[dict[str, object]] = [{"type": "input_text", "text": prompt}]
    content.append(
        {
            "type": "input_text",
            "text": "Reference atlas: rows are labeled with the crab class; columns are example appearances.",
        }
    )
    content.append({"type": "input_image", "image_url": _reference_atlas_data_url(reference_atlas_paths), "detail": "high"})
    content.append({"type": "input_text", "text": "Target frame to analyze:"})
    content.append({"type": "input_image", "image_url": _image_data_url(image_path), "detail": "high"})

    request_kwargs: dict[str, object] = {
        "model": model,
        "reasoning": {"effort": reasoning_effort},
        "prompt_cache_key": "triton_analysis_crab_counter_v2",
        "input": [
            {
                "role": "user",
                "content": content,
            }
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "crab_counter_result",
                "strict": True,
                "schema": _result_json_schema(),
            },
            "verbosity": "low",
        },
    }
    response = client.responses.create(**request_kwargs)
    if artifact_root is not None:
        _write_openai_stage_artifacts(
            artifact_root,
            "single_request_count",
            request_kwargs=request_kwargs,
            response=response,
            context=artifact_context,
        )
    return response


def _create_openai_candidate_detection_response(
    *,
    image_path: Path,
    image_size: tuple[int, int],
    model: str,
    reasoning_effort: str,
    client: Any | None,
    artifact_root: Path | None = None,
    artifact_context: Mapping[str, object] | None = None,
) -> Any:
    if client is None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - exercised through GUI messaging
            raise RuntimeError("OpenAI Python package is not installed. Run setup_windows.ps1 again.") from exc
        client = OpenAI()
    if not hasattr(client, "responses"):
        raise RuntimeError("OpenAI client does not expose the Responses API. Install openai>=2.0.0.")

    width, height = image_size
    content: list[dict[str, object]] = [
        {"type": "input_text", "text": _build_candidate_detection_prompt(width=width, height=height)},
        {"type": "input_image", "image_url": _image_data_url(image_path), "detail": "high"},
    ]
    request_kwargs: dict[str, object] = {
        "model": model,
        "reasoning": {"effort": reasoning_effort},
        "prompt_cache_key": "triton_analysis_crab_candidate_detector_v3",
        "input": [
            {
                "role": "user",
                "content": content,
            }
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "crab_candidate_detector",
                "strict": True,
                "schema": _candidate_detection_json_schema(),
            },
            "verbosity": "low",
        },
    }
    response = client.responses.create(**request_kwargs)
    if artifact_root is not None:
        _write_openai_stage_artifacts(
            artifact_root,
            "candidate_detection",
            request_kwargs=request_kwargs,
            response=response,
            context=artifact_context,
        )
    return response


def _create_openai_homography_response(
    *,
    image_path: Path,
    image_size: tuple[int, int],
    model: str,
    reasoning_effort: str,
    board_reference_paths: Sequence[Path],
    client: Any | None,
    artifact_root: Path | None = None,
    artifact_context: Mapping[str, object] | None = None,
) -> Any:
    if client is None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - exercised through GUI messaging
            raise RuntimeError("OpenAI Python package is not installed. Run setup_windows.ps1 again.") from exc
        client = OpenAI()
    if not hasattr(client, "responses"):
        raise RuntimeError("OpenAI client does not expose the Responses API. Install openai>=2.0.0.")

    width, height = image_size
    content: list[dict[str, object]] = [
        {"type": "input_text", "text": _build_homography_prompt(width=width, height=height)},
    ]
    references = _dedupe_paths(tuple(Path(path).expanduser() for path in board_reference_paths))[:BOARD_REFERENCE_MAX_IMAGES]
    if references:
        content.append(
            {
                "type": "input_text",
                "text": (
                    "Board appearance references only: these show the same kind of white board under pool lighting. "
                    "Use them to recognize the board material and caustics, but do not return coordinates from them."
                ),
            }
        )
        for reference_path in references:
            content.append({"type": "input_image", "image_url": _image_data_url(reference_path), "detail": "low"})
    content.append({"type": "input_text", "text": "Target image for board-corner coordinates:"})
    content.append({"type": "input_image", "image_url": _image_data_url(image_path), "detail": "high"})
    request_kwargs: dict[str, object] = {
        "model": model,
        "reasoning": {"effort": reasoning_effort},
        "prompt_cache_key": "triton_analysis_crab_board_homography_v1",
        "input": [
            {
                "role": "user",
                "content": content,
            }
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "crab_board_outline",
                "strict": True,
                "schema": _board_outline_json_schema(),
            },
            "verbosity": "low",
        },
    }
    response = client.responses.create(**request_kwargs)
    if artifact_root is not None:
        _write_openai_stage_artifacts(
            artifact_root,
            "board_homography",
            request_kwargs=request_kwargs,
            response=response,
            context=artifact_context,
        )
    return response


def _create_openai_candidate_classification_response(
    *,
    image_size: tuple[int, int],
    candidates: Sequence[CrabCandidateBox],
    contact_sheet: Path,
    reference_atlas_paths: Mapping[str, Sequence[Path]],
    model: str,
    reasoning_effort: str,
    target_confidence_threshold: float,
    target_margin_threshold: float,
    client: Any | None,
    artifact_root: Path | None = None,
    artifact_context: Mapping[str, object] | None = None,
) -> Any:
    if client is None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - exercised through GUI messaging
            raise RuntimeError("OpenAI Python package is not installed. Run setup_windows.ps1 again.") from exc
        client = OpenAI()
    if not hasattr(client, "responses"):
        raise RuntimeError("OpenAI client does not expose the Responses API. Install openai>=2.0.0.")

    width, height = image_size
    content: list[dict[str, object]] = [
        {
            "type": "input_text",
            "text": _build_candidate_classification_prompt(
                width=width,
                height=height,
                candidates=candidates,
                target_confidence_threshold=target_confidence_threshold,
                target_margin_threshold=target_margin_threshold,
            ),
        },
        {
            "type": "input_text",
            "text": "Reference atlas: rows are labeled with the crab class; columns are example appearances.",
        },
        {"type": "input_image", "image_url": _reference_atlas_data_url(reference_atlas_paths), "detail": "high"},
        {"type": "input_text", "text": "Numbered candidate crop contact sheet to classify:"},
        {"type": "input_image", "image_url": _image_data_url(contact_sheet), "detail": "high"},
    ]
    request_kwargs: dict[str, object] = {
        "model": model,
        "reasoning": {"effort": reasoning_effort},
        "prompt_cache_key": "triton_analysis_crab_candidate_classifier_v1",
        "input": [
            {
                "role": "user",
                "content": content,
            }
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "crab_candidate_classifier",
                "strict": True,
                "schema": _candidate_classification_json_schema(),
            },
            "verbosity": "low",
        },
    }
    response = client.responses.create(**request_kwargs)
    if artifact_root is not None:
        _write_openai_stage_artifacts(
            artifact_root,
            "candidate_classification",
            request_kwargs=request_kwargs,
            response=response,
            context=artifact_context,
        )
    return response


def _build_candidate_detection_prompt(*, width: int, height: int) -> str:
    return (
        "Find every visible printed crab candidate on this MATE crab-board image. This is a detector-only stage: "
        "do not classify species and do not decide which crabs are European green crabs. Return one tight bounding "
        "box for each printed crab image of any class. Include European green crab, native rock crab, Jonah crab, "
        "small crabs, rotated crabs, partially glared crabs, and partly clipped crabs. High recall is more "
        "important than species precision here; if a mark plausibly contains a printed crab, include it with lower "
        "confidence rather than missing it. Ignore board edges, screw heads, glare streaks, shadows, pool floor, "
        "robot parts, and blank white board. Do not duplicate the same crab with multiple boxes. "
        "Never group two printed crabs into one candidate box. Adjacent, touching, overlapping, or partially "
        "occluding crab prints must be split into separate candidate boxes whenever separate bodies, claw sets, "
        "leg fans, or printed outlines can be distinguished. A correct split may have overlapping boxes; overlap "
        "is better than one merged box. If a large region appears to contain multiple crab bodies, do not return "
        "that large region as a candidate. Instead return the tightest one-crab boxes you can. Set single_crab "
        "true only when the bbox contains exactly one printed crab body; if you cannot isolate one crab, set "
        "single_crab false and explain the merge in notes. In crowded clusters, first identify each visible "
        "carapace/body center, then draw one candidate box around each center and its own visible legs/claws. A "
        "bbox is wrong if it encloses two body centers, even when the crabs overlap or one crab's legs cross near "
        "another crab. Before finalizing, review each unusually large or crowded box and replace it with separate "
        "boxes if it contains two carapace centers or two distinct sets of legs/claws. "
        f"Return bbox coordinates in the full target image coordinate system, width={width}, height={height}, using "
        "the order [x1, y1, x2, y2]. Each bbox must tightly cover the visible printed ink of one crab, including "
        "legs and claws when visible, without extra blank board. Numbering can be approximate; the software will "
        "renumber boxes in reading order."
    )


def _candidate_detection_json_schema() -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "image_width": {"type": "integer"},
            "image_height": {"type": "integer"},
            "candidates": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "candidate_id": {"type": "integer"},
                        "bbox": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 4,
                            "maxItems": 4,
                        },
                        "confidence": {"type": "number"},
                        "single_crab": {"type": "boolean"},
                        "notes": {"type": "string"},
                    },
                    "required": ["candidate_id", "bbox", "confidence", "single_crab", "notes"],
                },
            },
            "summary": {"type": "string"},
        },
        "required": ["image_width", "image_height", "candidates", "summary"],
    }


def _build_homography_prompt(*, width: int, height: int) -> str:
    return (
        "Locate the outer boundary of the white rectangular MATE crab board or laminated white sheet in the image. "
        "The board may contain printed crab pictures or may look mostly blank under pool lighting. Return exactly "
        "the four physical board corners in full-image pixel "
        f"coordinates for this image, width={width}, height={height}. Use the order top_left, top_right, "
        "bottom_right, bottom_left as the board appears in the image, not as an unrotated document. The full board "
        "may be partly out of frame, clipped by the image border, covered by caustic light, or seen from a bad angle. "
        "If only part of the board is visible but the sheet boundary is still recognizable, set board_visible true "
        "and return the best four-corner outline for the visible board footprint; for corners outside the frame, use "
        "the closest visible edge or image-border intersection that would make the most stable usable homography. "
        "If a corner is rounded, glared, or partly occluded, estimate the point where the two outer board edges meet. "
        "Ignore the pool floor, grate, robot parts, gripper, shadows, glare streaks, screw heads, crab prints, and "
        "empty background. Do not outline the cluster of crabs; outline the sheet itself. Prefer a tight board "
        "outline over including any surrounding water or hardware. This is only a geometric preprocessing step: do "
        "not classify crabs and do not return crab bounding boxes. If the image is too poor, the board is too small, "
        "or you cannot distinguish the board from the pool floor well enough for a usable transform, set "
        "board_visible false, give low confidence, and still provide your best four-corner estimate."
    )


def _board_outline_json_schema() -> dict[str, object]:
    point_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "x": {"type": "number"},
            "y": {"type": "number"},
        },
        "required": ["x", "y"],
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "image_width": {"type": "integer"},
            "image_height": {"type": "integer"},
            "board_visible": {"type": "boolean"},
            "confidence": {"type": "number"},
            "top_left": point_schema,
            "top_right": point_schema,
            "bottom_right": point_schema,
            "bottom_left": point_schema,
            "notes": {"type": "string"},
        },
        "required": [
            "image_width",
            "image_height",
            "board_visible",
            "confidence",
            "top_left",
            "top_right",
            "bottom_right",
            "bottom_left",
            "notes",
        ],
    }


def _build_candidate_classification_prompt(
    *,
    width: int,
    height: int,
    candidates: Sequence[CrabCandidateBox],
    target_confidence_threshold: float,
    target_margin_threshold: float,
) -> str:
    candidate_lines = "\n".join(
        f"- Candidate {candidate.candidate_id}: bbox [{candidate.bbox[0]:.1f}, {candidate.bbox[1]:.1f}, "
        f"{candidate.bbox[2]:.1f}, {candidate.bbox[3]:.1f}]"
        for candidate in candidates
    )
    return (
        "You are classifying numbered candidate crops from a MATE ROV invasive species board. The reference atlas "
        "is shown before the candidate crop contact sheet. The contact sheet is the final image in the request and "
        "contains only crops already detected by a separate detector stage. Your job is classification only: do not "
        "change bounding boxes, do not add new candidates, and do not remove candidates from the response. Return "
        "exactly one classification object for every candidate id listed below. The original processed image size "
        f"is width={width}, height={height}; candidate boxes are fixed in that original coordinate system:\n"
        f"{candidate_lines}\n"
        "The only printed crab classes are European green crab, native rock crab, and Jonah crab. Species "
        "identification is more important than accepting every possible target. False positives are worse than "
        "false negatives: a native rock crab counted as European green crab is a serious error. Native rock crab "
        "is the main hard negative. Do not count a native rock crab as European green crab even if underwater "
        "lighting, caustics, blur, compression, faded ink, or glare makes it look greenish. Do not use color tint "
        "alone as species evidence because the pool can shift all colors. "
        "Use a reject-first decision process for every numbered crop. First compare the crop to native_rock_crab "
        "and Jonah crab in the atlas, looking for any non-target match in silhouette, leg/claw layout, body "
        "proportions, edge contour, and internal markings. Then compare to European green crab. Only after the "
        "hard-negative check may you assign european_green_crab. If any visible cue matches native_rock_crab as "
        "well as or better than European green crab, label it native_rock_crab or uncertain. Use uncertain only "
        "when the visible class cues are genuinely insufficient or the best target and non-target explanations are "
        "nearly tied. Do not choose uncertain merely because the crop is small, enlarged, slightly blurred, or "
        "caustic-lit. Confidence is a species-match confidence conditional on the visible crop, not an image-quality "
        "score. A small crop with clear European green crab silhouette, leg/claw layout, and compact carapace can "
        "still receive target_match_confidence above the threshold. Assign european_green_crab when at least two "
        "independent non-color visual cues support that class and no visible cue matches a non-target reference as "
        "well or better. "
        "For each candidate, assign class_scores for all three classes. Do not artificially suppress the native "
        "rock crab score to make an uncertain crop pass. closest_non_target must be the better of native_rock_crab "
        "and jonah_crab. decision_margin must be class_scores.european_green_crab minus the larger non-target "
        "score. Fill egc_supporting_cues with the visible non-color cues supporting European green crab, and "
        "non_target_supporting_cues with visible cues supporting native rock crab or Jonah crab. Set "
        "visible_cues_sufficient true when the crop has enough visible shape/leg/body evidence to classify even if "
        "the crop is small or blurry. In notes, use at most 12 words naming the strongest cue and closest rejected "
        "class. Set "
        f"target_match_confidence >= {target_confidence_threshold:.2f} for clear European green crab matches "
        "after the native-rock veto; set it below that threshold for genuinely ambiguous, insufficiently visible, "
        "or non-target crops. Set accepted_as_target true only when label is european_green_crab, "
        f"target_match_confidence >= {target_confidence_threshold:.2f}, decision_margin >= "
        f"{target_margin_threshold:.2f}, visible_cues_sufficient is true, and the crop survives the native-rock "
        "hard-negative check."
    )


def _candidate_classification_json_schema() -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "classifications": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "candidate_id": {"type": "integer"},
                        "label": {"type": "string", "enum": list(CANDIDATE_CLASSES)},
                        "confidence": {"type": "number"},
                        "target_match_confidence": {"type": "number"},
                        "class_scores": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                class_name: {"type": "number"}
                                for class_name in CRAB_CLASS_NAMES
                            },
                            "required": list(CRAB_CLASS_NAMES),
                        },
                        "closest_non_target": {"type": "string", "enum": list(NON_TARGET_CLASSES)},
                        "decision_margin": {"type": "number"},
                        "egc_supporting_cues": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "non_target_supporting_cues": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "visible_cues_sufficient": {"type": "boolean"},
                        "accepted_as_target": {"type": "boolean"},
                        "notes": {"type": "string"},
                    },
                    "required": [
                        "candidate_id",
                        "label",
                        "confidence",
                        "target_match_confidence",
                        "class_scores",
                        "closest_non_target",
                        "decision_margin",
                        "egc_supporting_cues",
                        "non_target_supporting_cues",
                        "visible_cues_sufficient",
                        "accepted_as_target",
                        "notes",
                    ],
                },
            },
            "summary": {"type": "string"},
        },
        "required": ["classifications", "summary"],
    }


def _build_prompt(*, width: int, height: int, target_confidence_threshold: float, target_margin_threshold: float) -> str:
    return (
        "You are counting the MATE ROV invasive species board. The reference atlas is shown before the target "
        "frame so it can be cached across repeated runs. The target frame is the final image in the request. "
        "The atlas contains the only printed crab classes that may appear on the board: European green crab, "
        "native rock crab, and Jonah crab. First locate every visible printed crab candidate in the target image. "
        "Then compare each candidate directly against the atlas examples and classify it as european_green_crab, "
        "native_rock_crab, jonah_crab, or uncertain. "
        "Species identification is more important than finding every possible target. False positives are worse "
        "than false negatives: a native rock crab counted as European green crab is a serious error. Native rock "
        "crab and Jonah crab are hard negatives, and native_rock_crab is the main hard negative. Do not count a "
        "native rock crab as European green crab even if underwater lighting, caustics, blur, compression, faded "
        "ink, or glare makes it look greenish. Do not use color tint alone as species evidence because the pool can "
        "shift all colors. Ignore pool tiles, glare, fasteners, the gripper, shadows, and paper edges. "
        "Use a reject-first decision process for every candidate. First compare the candidate to native_rock_crab "
        "and Jonah crab in the atlas, looking for any non-target match in silhouette, leg/claw layout, body "
        "proportions, edge contour, and internal markings. Then compare to European green crab. Only after the "
        "hard-negative check may you assign european_green_crab. If the candidate has any visible cue that matches "
        "native_rock_crab as well as or better than European green crab, label it native_rock_crab or uncertain. "
        "If the candidate is ambiguous between European green crab and native rock crab, label it uncertain or "
        "native_rock_crab rather than european_green_crab. "
        "Allow real target prints to be rotated, scaled, blurred, partly glared over, or color shifted, but require "
        "the visible silhouette, leg/claw layout, body proportions, and internal markings to match the European "
        "green crab reference better than both non-target references before assigning european_green_crab. Assign "
        "european_green_crab only when at least two independent non-color visual cues support that class and no "
        "visible cue matches a non-target reference as well or better. A greenish or dark body is not one of the "
        "two required cues. Treat small, blurry, caustic-covered, or partly hidden candidates conservatively: if "
        "you cannot explain why it is not native_rock_crab, do not accept it as a target. "
        "For each candidate, assign class_scores for all three classes. The class_scores should express relative "
        "visual support from the atlas; they do not need to add to 1.0. Do not artificially suppress the native "
        "rock crab score to make an uncertain candidate pass. If native_rock_crab is plausible, its score should "
        "remain high enough to lower the decision_margin. closest_non_target must be the better of native_rock_crab "
        "and jonah_crab. decision_margin must be class_scores.european_green_crab minus the larger non-target "
        "score. In notes, use at most 12 words naming the strongest cue and closest rejected class. Set "
        "target_match_confidence >= "
        f"{target_confidence_threshold:.2f} only for clear European green crab matches after the native-rock veto; "
        f"set it below {target_confidence_threshold:.2f} for likely-but-not-clear, ambiguous, small/blurred, "
        "glare-covered, or non-target candidates. Set accepted_as_target true only when label is "
        "european_green_crab, target_match_confidence >= "
        f"{target_confidence_threshold:.2f}, decision_margin >= {target_margin_threshold:.2f}, and the candidate "
        "survives the native-rock hard-negative check. "
        "Return bounding boxes in pixel coordinates for the full "
        f"target image, whose size is width={width}, height={height}. Each bbox must tightly cover the visible "
        "printed ink of one crab, including legs and claws when visible, and must use the order [x1, y1, x2, y2]. "
        "For small crabs, zoom mentally and center the box on the visible printed crab. Do not include surrounding "
        "white board, glare streaks, shadows, screw heads, adjacent crab prints, or empty water just to make the box "
        "larger. Do not crop off visible legs or claws, and do not shift the box toward a shadow or reflection. "
        "Before finalizing the JSON, review each bbox: its center should fall on the crab print, not on blank board "
        "or a shadow, and the box should be shifted or resized if the printed crab is off center inside it. "
        "One printed crab should have exactly one candidate box. "
        "If only part of a crab is visible, box only the visible printed portion and say that in notes. The count "
        "must equal only candidates where accepted_as_target is true."
    )


def _result_json_schema() -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "image_width": {"type": "integer"},
            "image_height": {"type": "integer"},
            "count": {"type": "integer"},
            "candidates": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "label": {"type": "string", "enum": list(CANDIDATE_CLASSES)},
                        "bbox": {
                            "type": "array",
                            "items": {"type": "number"},
                        },
                        "confidence": {"type": "number"},
                        "target_match_confidence": {"type": "number"},
                        "class_scores": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                class_name: {"type": "number"}
                                for class_name in CRAB_CLASS_NAMES
                            },
                            "required": list(CRAB_CLASS_NAMES),
                        },
                        "closest_non_target": {"type": "string", "enum": list(NON_TARGET_CLASSES)},
                        "decision_margin": {"type": "number"},
                        "accepted_as_target": {"type": "boolean"},
                        "notes": {"type": "string"},
                    },
                    "required": [
                        "label",
                        "bbox",
                        "confidence",
                        "target_match_confidence",
                        "class_scores",
                        "closest_non_target",
                        "decision_margin",
                        "accepted_as_target",
                        "notes",
                    ],
                },
            },
            "summary": {"type": "string"},
        },
        "required": ["image_width", "image_height", "count", "candidates", "summary"],
    }


def _parse_response(
    response: Any,
    *,
    image_path: Path,
    image_size: tuple[int, int],
    model: str,
    reasoning_effort: str,
    target_confidence_threshold: float,
    target_margin_threshold: float,
) -> CrabCountResult:
    text = _response_text(response)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"OpenAI response was not valid JSON: {text[:500]}") from exc
    if not isinstance(payload, Mapping):
        raise RuntimeError("OpenAI response JSON was not an object.")
    return result_from_payload(
        payload,
        image_path=image_path,
        image_size=image_size,
        model=model,
        reasoning_effort=reasoning_effort,
        target_confidence_threshold=target_confidence_threshold,
        target_margin_threshold=target_margin_threshold,
    )


def _parse_board_outline_response(
    response: Any,
    *,
    image_path: Path,
    image_size: tuple[int, int],
    model: str,
    reasoning_effort: str,
) -> CrabBoardOutlineResult:
    text = _response_text(response)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"OpenAI board-outline response was not valid JSON: {text[:500]}") from exc
    if not isinstance(payload, Mapping):
        raise RuntimeError("OpenAI board-outline response JSON was not an object.")

    width, height = image_size
    names = ("top_left", "top_right", "bottom_right", "bottom_left")
    raw_points: list[tuple[float, float]] = []
    for name in names:
        point = payload.get(name)
        if not isinstance(point, Mapping):
            raise RuntimeError(f"OpenAI board-outline response missing {name}.")
        try:
            x = float(point["x"])
            y = float(point["y"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"OpenAI board-outline response had invalid {name}.") from exc
        raw_points.append((max(0.0, min(float(width), x)), max(0.0, min(float(height), y))))

    ordered = _order_quad_points(np.array(raw_points, dtype=np.float32))
    board_visible = bool(payload.get("board_visible", True))
    confidence = _clamp01(payload.get("confidence", 0.0))
    if not board_visible:
        raise RuntimeError("OpenAI did not find a usable crab board outline in the image.")
    if confidence < 0.2:
        raise RuntimeError(f"OpenAI board-outline confidence was too low ({confidence:.2f}).")
    return CrabBoardOutlineResult(
        image_path=image_path,
        image_size=image_size,
        points=tuple((float(x), float(y)) for x, y in ordered),
        confidence=confidence,
        board_visible=board_visible,
        model=str(model),
        reasoning_effort=_normalize_reasoning_effort(reasoning_effort),
        notes=str(payload.get("notes") or ""),
    )


def _response_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text
    if isinstance(response, Mapping):
        value = response.get("output_text")
        if isinstance(value, str) and value.strip():
            return value
    output = getattr(response, "output", None)
    if output is None and isinstance(response, Mapping):
        output = response.get("output")
    chunks: list[str] = []
    for item in output or []:
        content = getattr(item, "content", None)
        if content is None and isinstance(item, Mapping):
            content = item.get("content")
        for part in content or []:
            text = getattr(part, "text", None)
            if text is None and isinstance(part, Mapping):
                text = part.get("text")
            if isinstance(text, str):
                chunks.append(text)
    if chunks:
        return "\n".join(chunks)
    raise RuntimeError("OpenAI response did not contain output text.")


def _image_data_url(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix not in IMAGE_EXTENSIONS:
        raise ValueError(f"unsupported image type for crab counter: {path}")
    mime = mimetypes.guess_type(str(path))[0] or "image/jpeg"
    data = path.read_bytes()
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _reference_atlas_data_url(reference_atlas_paths: Mapping[str, Sequence[Path]]) -> str:
    atlas = _build_reference_atlas(reference_atlas_paths)
    ok, encoded = cv2.imencode(".png", atlas)
    if not ok:
        raise OSError("could not encode crab reference atlas")
    data = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def _build_reference_atlas(reference_atlas_paths: Mapping[str, Sequence[Path]]) -> np.ndarray:
    max_examples = max(
        1,
        min(
            REFERENCE_ATLAS_MAX_EXAMPLES_PER_CLASS,
            max((len(reference_atlas_paths.get(class_name, ())) for class_name in CRAB_CLASS_NAMES), default=1),
        ),
    )
    label_width = 210
    cell_width = 190
    row_height = 178
    header_height = 38
    atlas = np.full(
        (header_height + row_height * len(CRAB_CLASS_NAMES), label_width + cell_width * max_examples, 3),
        245,
        dtype=np.uint8,
    )
    grid_color = (210, 210, 210)
    text_color = (35, 35, 35)
    cv2.putText(atlas, "Crab reference atlas", (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.72, text_color, 2, cv2.LINE_AA)
    for index in range(max_examples):
        x = label_width + index * cell_width + 10
        cv2.putText(atlas, f"Example {index + 1}", (x, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

    for row_index, class_name in enumerate(CRAB_CLASS_NAMES):
        y0 = header_height + row_index * row_height
        y1 = y0 + row_height
        cv2.rectangle(atlas, (0, y0), (atlas.shape[1] - 1, y1), grid_color, 1)
        label = REFERENCE_CLASS_LABELS.get(class_name, class_name)
        cv2.putText(atlas, label, (12, y0 + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, 2, cv2.LINE_AA)
        cv2.putText(atlas, class_name, (12, y0 + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.42, text_color, 1, cv2.LINE_AA)
        for col_index, path in enumerate(reference_atlas_paths.get(class_name, ())[:max_examples]):
            x0 = label_width + col_index * cell_width
            x1 = x0 + cell_width
            cv2.rectangle(atlas, (x0, y0), (x1, y1), grid_color, 1)
            image = _read_image(Path(path).expanduser())
            if image is None:
                continue
            fitted = _fit_image_to_box(image, cell_width - 24, row_height - 48)
            fy, fx = fitted.shape[:2]
            px = x0 + (cell_width - fx) // 2
            py = y0 + 12 + (row_height - 48 - fy) // 2
            atlas[py : py + fy, px : px + fx] = fitted
            source_label = Path(path).stem[:24]
            cv2.putText(atlas, source_label, (x0 + 8, y1 - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, text_color, 1, cv2.LINE_AA)
    return atlas


def _fit_image_to_box(image: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
    height, width = image.shape[:2]
    if width <= 0 or height <= 0:
        return image
    scale = min(max_width / float(width), max_height / float(height))
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    return cv2.resize(image, (new_width, new_height), interpolation=interpolation)


def _read_image(path: Path) -> np.ndarray | None:
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _write_image(path: Path, image: np.ndarray) -> None:
    suffix = path.suffix or ".png"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise OSError(f"could not encode image: {path}")
    encoded.tofile(str(path))


def _clamp_bbox(
    bbox: Sequence[float],
    image_size: tuple[int, int],
) -> tuple[float, float, float, float]:
    width, height = image_size
    x0, y0, x1, y1 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    return (
        max(0.0, min(float(width), x0)),
        max(0.0, min(float(height), y0)),
        max(0.0, min(float(width), x1)),
        max(0.0, min(float(height), y1)),
    )


def _transform_bbox(
    bbox: Sequence[float],
    matrix: np.ndarray,
    image_size: tuple[int, int],
) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = _clamp_bbox(bbox, (10**9, 10**9))
    corners = np.array([[[x0, y0]], [[x1, y0]], [[x1, y1]], [[x0, y1]]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(corners, matrix.astype(np.float64)).reshape(-1, 2)
    min_x = float(np.min(transformed[:, 0]))
    max_x = float(np.max(transformed[:, 0]))
    min_y = float(np.min(transformed[:, 1]))
    max_y = float(np.max(transformed[:, 1]))
    return _clamp_bbox((min_x, min_y, max_x, max_y), image_size)


def _expanded_bbox(
    bbox: Sequence[float],
    image_size: tuple[int, int],
    *,
    fraction: float,
    min_pad: float,
) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = _clamp_bbox(bbox, image_size)
    pad = max(float(min_pad), max(x1 - x0, y1 - y0) * float(fraction))
    return _clamp_bbox((x0 - pad, y0 - pad, x1 + pad, y1 + pad), image_size)


def _draw_label(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    foreground: tuple[int, int, int],
    outline: tuple[int, int, int],
    *,
    scale: float = 0.7,
    thickness: int = 2,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = origin
    x = max(0, min(image.shape[1] - text_w - 8, x))
    y = max(text_h + 8, min(image.shape[0] - baseline - 4, y))
    cv2.putText(image, text, (x, y), font, scale, outline, thickness + 2, cv2.LINE_AA)
    cv2.putText(image, text, (x, y), font, scale, foreground, thickness, cv2.LINE_AA)


def _choose_count_label_origin(
    text: str,
    image_size: tuple[int, int],
    detection_bboxes: Sequence[Sequence[float]],
    *,
    scale: float,
    thickness: int,
) -> tuple[int, int]:
    font = cv2.FONT_HERSHEY_SIMPLEX
    width, height = image_size
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    margin = 18
    candidates = (
        (margin, margin + text_h),
        (max(margin, width - margin - text_w), margin + text_h),
        (margin, max(margin + text_h, height - margin - baseline)),
        (max(margin, width - margin - text_w), max(margin + text_h, height - margin - baseline)),
    )

    def label_bbox(origin: tuple[int, int]) -> tuple[float, float, float, float]:
        x, y = origin
        return (
            float(max(0, min(width - text_w - 8, x))),
            float(max(0, y - text_h - 6)),
            float(min(width, x + text_w + 8)),
            float(min(height, y + baseline + 6)),
        )

    def overlap_area(a: tuple[float, float, float, float], b: Sequence[float]) -> float:
        ax0, ay0, ax1, ay1 = a
        bx0, by0, bx1, by1 = _clamp_bbox(b, image_size)
        overlap_w = max(0.0, min(ax1, bx1) - max(ax0, bx0))
        overlap_h = max(0.0, min(ay1, by1) - max(ay0, by0))
        return overlap_w * overlap_h

    return min(candidates, key=lambda origin: sum(overlap_area(label_bbox(origin), bbox) for bbox in detection_bboxes))
