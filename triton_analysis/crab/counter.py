"""OpenAI-backed counting for printed MATE crab-board images."""

from __future__ import annotations

import base64
import json
import mimetypes
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from triton_analysis.crab.plane_dataset import discover_default_crab_template_paths
from triton_analysis.crab.synthetic import CRAB_CLASS_NAMES, IMAGE_EXTENSIONS
from triton_analysis.workspace import fresh_output_subdir, workspace_paths


TARGET_CLASS = "european_green_crab"
REFERENCE_CLASS_LABELS = {
    "european_green_crab": "European green crab",
    "native_rock_crab": "Native rock crab",
    "jonah_crab": "Jonah crab",
}
DEFAULT_MODEL = os.environ.get("TRITON_ANALYSIS_CRAB_MODEL", "gpt-5.5")


@dataclass(frozen=True)
class CrabDetection:
    """One European green crab detection in target-image pixel coordinates."""

    label: str
    bbox: tuple[float, float, float, float]
    confidence: float
    notes: str = ""


@dataclass(frozen=True)
class CrabCountResult:
    """Structured crab-counter result."""

    image_path: Path
    image_size: tuple[int, int]
    count: int
    detections: tuple[CrabDetection, ...]
    model: str
    summary: str = ""

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["image_path"] = str(self.image_path)
        data["image_size"] = list(self.image_size)
        data["detections"] = [
            {
                "label": detection.label,
                "bbox": [round(float(value), 2) for value in detection.bbox],
                "confidence": round(float(detection.confidence), 4),
                "notes": detection.notes,
            }
            for detection in self.detections
        ]
        return data


@dataclass(frozen=True)
class CrabCounterOutputs:
    """Files written for one crab-counter run."""

    result: CrabCountResult
    output_dir: Path
    result_json: Path
    annotated_image: Path


@dataclass(frozen=True)
class CrabCounterConfig:
    """Inputs for one OpenAI crab-counter request."""

    image_path: Path
    reference_paths: Mapping[str, Path]
    output_dir: Path
    model: str = DEFAULT_MODEL


def discover_counter_reference_paths(workspace_root: str | Path | None = None) -> dict[str, Path | None]:
    """Return the best available reference image for each known crab class."""

    workspace = workspace_paths(workspace_root, create=False)
    discovered = discover_default_crab_template_paths(workspace.root)
    references: dict[str, Path | None] = {}
    for class_name in CRAB_CLASS_NAMES:
        paths = discovered.get(class_name, [])
        references[class_name] = paths[0] if paths else None
    return references


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

    response = _create_openai_response(
        image_path=image_path,
        image_size=image_size,
        reference_paths={name: path for name, path in normalized_refs.items() if path is not None},
        model=str(config.model or DEFAULT_MODEL),
        client=client,
    )
    result = _parse_response(response, image_path=image_path, image_size=image_size, model=str(config.model or DEFAULT_MODEL))

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
) -> CrabCountResult:
    """Validate a model JSON payload into a crab-count result."""

    detections: list[CrabDetection] = []
    width, height = image_size
    for raw in payload.get("detections", []):
        if not isinstance(raw, Mapping):
            continue
        label = str(raw.get("label") or TARGET_CLASS).strip().lower()
        if label != TARGET_CLASS:
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
        try:
            confidence = float(raw.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))
        notes = str(raw.get("notes") or "")
        detections.append(CrabDetection(label=TARGET_CLASS, bbox=(x0, y0, x1, y1), confidence=confidence, notes=notes))

    detections.sort(key=lambda detection: (detection.bbox[1], detection.bbox[0]))
    summary = str(payload.get("summary") or "")
    return CrabCountResult(
        image_path=Path(image_path).expanduser(),
        image_size=image_size,
        count=len(detections),
        detections=tuple(detections),
        model=str(model),
        summary=summary,
    )


def _normalize_reference_paths(reference_paths: Mapping[str, Path | str | None]) -> dict[str, Path | None]:
    normalized: dict[str, Path | None] = {}
    for class_name in CRAB_CLASS_NAMES:
        value = reference_paths.get(class_name)
        normalized[class_name] = Path(value).expanduser() if value else None
    return normalized


def _create_openai_response(
    *,
    image_path: Path,
    image_size: tuple[int, int],
    reference_paths: Mapping[str, Path],
    model: str,
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
    prompt = _build_prompt(width=width, height=height)
    content: list[dict[str, object]] = [{"type": "input_text", "text": prompt}]
    content.append({"type": "input_image", "image_url": _image_data_url(image_path), "detail": "high"})
    for class_name in CRAB_CLASS_NAMES:
        reference_path = reference_paths[class_name]
        label = REFERENCE_CLASS_LABELS.get(class_name, class_name)
        content.append({"type": "input_text", "text": f"Reference image: {label}."})
        content.append({"type": "input_image", "image_url": _image_data_url(reference_path), "detail": "high"})

    return client.responses.create(
        model=model,
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
            }
        },
    )


def _build_prompt(*, width: int, height: int) -> str:
    return (
        "You are counting the MATE ROV invasive species board. The first image is the target frame. "
        "The following three images are the only crab print types that may appear on the board: European green crab, "
        "native rock crab, and Jonah crab. Identify and box only European green crab print instances in the target. "
        "Ignore native rock crab, Jonah crab, pool tiles, glare, fasteners, the gripper, shadows, and paper edges. "
        "Use the reference images as exact visual definitions, but allow the target crabs to be rotated, scaled, "
        "blurred, partially glared over, or color shifted by underwater lighting. Return bounding boxes in pixel "
        f"coordinates for the full target image, whose size is width={width}, height={height}. "
        "Each bbox must tightly cover the visible printed crab, including legs and claws when visible, and must use "
        "the order [x1, y1, x2, y2]. If a European green crab is small but visible, include it. "
        "Do not include any non-European crab in detections."
    )


def _result_json_schema() -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "image_width": {"type": "integer"},
            "image_height": {"type": "integer"},
            "count": {"type": "integer"},
            "detections": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "label": {"type": "string", "enum": [TARGET_CLASS]},
                        "bbox": {
                            "type": "array",
                            "items": {"type": "number"},
                        },
                        "confidence": {"type": "number"},
                        "notes": {"type": "string"},
                    },
                    "required": ["label", "bbox", "confidence", "notes"],
                },
            },
            "summary": {"type": "string"},
        },
        "required": ["image_width", "image_height", "count", "detections", "summary"],
    }


def _parse_response(response: Any, *, image_path: Path, image_size: tuple[int, int], model: str) -> CrabCountResult:
    text = _response_text(response)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"OpenAI response was not valid JSON: {text[:500]}") from exc
    if not isinstance(payload, Mapping):
        raise RuntimeError("OpenAI response JSON was not an object.")
    return result_from_payload(payload, image_path=image_path, image_size=image_size, model=model)


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
