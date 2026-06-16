"""OpenAI-backed counting for printed MATE crab-board images."""

from __future__ import annotations

import base64
import csv
import json
import mimetypes
import os
import time
from dataclasses import dataclass, replace
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
DEFAULT_REASONING_EFFORT = os.environ.get("TRITON_ANALYSIS_CRAB_REASONING_EFFORT", "high").strip().lower() or "high"
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


@dataclass(frozen=True)
class CrabBenchmarkOutputs:
    """Files written for a reasoning-effort benchmark run."""

    output_dir: Path
    summary_json: Path
    summary_csv: Path
    runs: tuple[CrabCounterOutputs, ...]


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

    output_dir = Path(config.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    result_json = output_dir / f"{image_path.stem}_crab_count.json"
    annotated_image = output_dir / f"{image_path.stem}_crab_count_annotated.png"
    result_json.write_text(json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8")
    draw_crab_count_result(image_path, result, annotated_image)
    return CrabCounterOutputs(
        result=result,
        output_dir=output_dir,
        result_json=result_json,
        annotated_image=annotated_image,
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
        "target_confidence_threshold": _clamp01(config.target_confidence_threshold),
        "target_margin_threshold": _clamp01(config.target_margin_threshold),
        "runs": [_benchmark_row(run) for run in runs],
    }
    summary_json = output_dir / "reasoning_effort_benchmark.json"
    summary_csv = output_dir / "reasoning_effort_benchmark.csv"
    summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _write_benchmark_csv(summary_csv, runs)
    return CrabBenchmarkOutputs(
        output_dir=output_dir,
        summary_json=summary_json,
        summary_csv=summary_csv,
        runs=tuple(runs),
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
        cv2.rectangle(out, p0, p1, black, 8, lineType=cv2.LINE_AA)
        cv2.rectangle(out, p0, p1, green, 3, lineType=cv2.LINE_AA)
        label = f"EGC {index}"
        label_origin = (p0[0], max(24, p0[1] - 8))
        _draw_label(out, label, label_origin, green, black)

    _draw_label(out, f"European green crabs: {result.count}", (18, 34), green, black, scale=0.85, thickness=2)
    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_image(output, out)
    return output


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
    return {
        "reasoning_effort": result.reasoning_effort,
        "analysis_seconds": round(float(result.analysis_seconds), 3),
        "count": int(result.count),
        "candidate_count": len(result.candidates),
        "result_json": str(run.result_json),
        "annotated_image": str(run.annotated_image),
        "summary": result.summary,
    }


def _write_benchmark_csv(path: Path, runs: Sequence[CrabCounterOutputs]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "reasoning_effort",
        "analysis_seconds",
        "count",
        "candidate_count",
        "result_json",
        "annotated_image",
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

    return client.responses.create(
        model=model,
        reasoning={"effort": reasoning_effort},
        prompt_cache_key="triton_analysis_crab_counter_v2",
        input=[
            {
                "role": "user",
                "content": content,
            }
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "crab_counter_result",
                "strict": True,
                "schema": _result_json_schema(),
            },
            "verbosity": "low",
        },
    )


def _build_prompt(*, width: int, height: int, target_confidence_threshold: float, target_margin_threshold: float) -> str:
    return (
        "You are counting the MATE ROV invasive species board. The reference atlas is shown before the target "
        "frame so it can be cached across repeated runs. The target frame is the final image in the request. "
        "The atlas contains the only printed crab classes that may appear on the board: European green crab, "
        "native rock crab, and Jonah crab. First locate every visible printed crab candidate in the target image. "
        "Then compare each candidate directly against the atlas examples and classify it as european_green_crab, "
        "native_rock_crab, jonah_crab, or uncertain. "
        "Species identification is more important than finding every possible target. Native rock crab and Jonah "
        "crab are hard negatives. Do not count a native rock crab as European green crab even if underwater lighting, "
        "blur, compression, faded ink, or glare makes it look greenish. Do not use color tint alone as species "
        "evidence because the pool can shift all colors. If a candidate is ambiguous between European green crab "
        "and native rock crab, label it uncertain or native_rock_crab rather than european_green_crab. Ignore pool "
        "tiles, glare, fasteners, the gripper, shadows, and paper edges. "
        "Allow real target prints to be rotated, scaled, blurred, partly glared over, or color shifted, but require "
        "the visible silhouette, leg/claw layout, body proportions, and internal markings to match the European "
        "green crab reference better than both non-target references before assigning european_green_crab. Assign "
        "european_green_crab only when at least two independent visual cues support that class and no visible cue "
        "matches a non-target reference as well or better. For each candidate, assign class_scores for all three "
        "classes. The class_scores should sum loosely to your relative visual support; they do not need to add to "
        "1.0. closest_non_target must be the better of native_rock_crab and jonah_crab. decision_margin must be "
        "class_scores.european_green_crab minus the larger non-target score. In notes, use at most 12 words naming "
        "the strongest cue and closest rejected class. Set target_match_confidence >= "
        f"{target_confidence_threshold:.2f} only for clear European green crab matches; set it below "
        f"{target_confidence_threshold:.2f} for likely-but-not-clear, ambiguous, or non-target candidates. Set "
        "accepted_as_target true only when label is european_green_crab, target_match_confidence >= "
        f"{target_confidence_threshold:.2f}, and decision_margin >= {target_margin_threshold:.2f}. "
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


def _draw_label(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    foreground: tuple[int, int, int],
    background: tuple[int, int, int],
    *,
    scale: float = 0.7,
    thickness: int = 2,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = origin
    x = max(0, min(image.shape[1] - text_w - 8, x))
    y = max(text_h + 8, min(image.shape[0] - baseline - 4, y))
    cv2.rectangle(image, (x - 4, y - text_h - 8), (x + text_w + 4, y + baseline + 4), background, -1)
    cv2.putText(image, text, (x, y), font, scale, foreground, thickness, cv2.LINE_AA)
