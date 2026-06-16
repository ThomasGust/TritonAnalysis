import json
from pathlib import Path

import cv2
import numpy as np

from triton_analysis.crab.counter import (
    CrabCounterConfig,
    analyze_crab_image,
    benchmark_crab_image,
    draw_crab_count_result,
    result_from_payload,
)


def _write_image(path: Path, color: tuple[int, int, int] = (40, 80, 120), size: tuple[int, int] = (120, 90)) -> None:
    image = np.full((size[1], size[0], 3), color, dtype=np.uint8)
    ok = cv2.imwrite(str(path), image)
    assert ok


class _FakeResponses:
    def __init__(self):
        self.kwargs = None
        self.calls = []

    def create(self, **kwargs):
        self.kwargs = kwargs
        self.calls.append(kwargs)
        return type(
            "FakeResponse",
            (),
            {
                "output_text": json.dumps(
                    {
                        "image_width": 120,
                        "image_height": 90,
                        "count": 2,
                        "candidates": [
                            {
                                "label": "european_green_crab",
                                "bbox": [10, 12, 42, 46],
                                "confidence": 0.91,
                                "target_match_confidence": 0.93,
                                "notes": "left green crab",
                            },
                            {
                                "label": "native_rock_crab",
                                "bbox": [44, 14, 62, 39],
                                "confidence": 0.94,
                                "target_match_confidence": 0.12,
                                "notes": "hard negative",
                            },
                            {
                                "label": "european_green_crab",
                                "bbox": [70, 20, 110, 62],
                                "confidence": 0.87,
                                "target_match_confidence": 0.88,
                                "notes": "right green crab",
                            },
                        ],
                        "summary": "Three candidates found; two are European green crabs.",
                    }
                )
            },
        )()


class _FakeClient:
    def __init__(self):
        self.responses = _FakeResponses()


def test_result_from_payload_filters_and_clamps_to_green_crabs(tmp_path: Path):
    payload = {
        "count": 3,
        "candidates": [
            {
                "label": "native_rock_crab",
                "bbox": [0, 0, 50, 50],
                "confidence": 1.0,
                "target_match_confidence": 0.1,
                "notes": "",
            },
            {
                "label": "european_green_crab",
                "bbox": [100, 80, 20, -5],
                "confidence": 1.4,
                "target_match_confidence": 0.9,
                "notes": "reversed",
            },
            {
                "label": "european_green_crab",
                "bbox": [3, 4, 20, 22],
                "confidence": 0.6,
                "target_match_confidence": 0.6,
                "notes": "below threshold",
            },
            {
                "label": "uncertain",
                "bbox": [30, 10, 45, 30],
                "confidence": 0.8,
                "target_match_confidence": 0.5,
                "notes": "ambiguous",
            },
        ],
        "summary": "Mixed crab labels.",
    }

    result = result_from_payload(
        payload,
        image_path=tmp_path / "target.png",
        image_size=(90, 70),
        model="test-model",
    )

    assert result.count == 1
    assert len(result.candidates) == 4
    assert [detection.label for detection in result.detections] == ["european_green_crab"]
    assert result.detections[0].bbox == (20.0, 0.0, 90.0, 70.0)
    assert result.detections[0].confidence == 1.0
    assert result.detections[0].target_match_confidence == 0.9


def test_draw_crab_count_result_writes_annotated_image(tmp_path: Path):
    target = tmp_path / "target.png"
    _write_image(target)
    result = result_from_payload(
        {
            "candidates": [
                {
                    "label": "european_green_crab",
                    "bbox": [10, 10, 60, 45],
                    "confidence": 0.9,
                    "target_match_confidence": 0.9,
                    "notes": "",
                }
            ],
            "summary": "One.",
        },
        image_path=target,
        image_size=(120, 90),
        model="test-model",
    )

    output = draw_crab_count_result(target, result, tmp_path / "annotated.png")

    assert output.exists()
    annotated = cv2.imread(str(output), cv2.IMREAD_COLOR)
    assert annotated is not None
    assert annotated.shape[:2] == (90, 120)


def test_analyze_crab_image_uses_responses_api_and_writes_outputs(tmp_path: Path):
    target = tmp_path / "target.png"
    refs = {
        "european_green_crab": tmp_path / "green.png",
        "native_rock_crab": tmp_path / "rock.png",
        "jonah_crab": tmp_path / "jonah.png",
    }
    _write_image(target, (30, 80, 120))
    _write_image(refs["european_green_crab"], (20, 60, 50))
    _write_image(refs["native_rock_crab"], (80, 110, 160))
    _write_image(refs["jonah_crab"], (90, 130, 190))
    fake_client = _FakeClient()

    outputs = analyze_crab_image(
        CrabCounterConfig(
            image_path=target,
            reference_paths=refs,
            output_dir=tmp_path / "out",
            model="test-vision-model",
        ),
        client=fake_client,
    )

    assert outputs.result.count == 2
    assert len(outputs.result.candidates) == 3
    assert outputs.result.target_confidence_threshold == 0.85
    assert outputs.result.analysis_seconds >= 0.0
    assert outputs.result_json.exists()
    assert outputs.annotated_image.exists()
    assert fake_client.responses.kwargs["model"] == "test-vision-model"
    assert fake_client.responses.kwargs["reasoning"] == {"effort": "high"}
    content = fake_client.responses.kwargs["input"][0]["content"]
    assert content[0]["type"] == "input_text"
    assert "European green crab" in content[0]["text"]
    assert "hard negatives" in content[0]["text"]
    assert "Species identification is more important" in content[0]["text"]
    assert "0.85" in content[0]["text"]
    assert sum(1 for item in content if item["type"] == "input_image") == 4
    assert fake_client.responses.kwargs["text"]["format"]["type"] == "json_schema"
    schema = fake_client.responses.kwargs["text"]["format"]["schema"]
    assert "candidates" in schema["properties"]
    candidate_schema = schema["properties"]["candidates"]["items"]["properties"]
    assert "native_rock_crab" in candidate_schema["label"]["enum"]
    assert "target_match_confidence" in candidate_schema
    written = json.loads(outputs.result_json.read_text(encoding="utf-8"))
    assert written["analysis_seconds"] >= 0.0


def test_benchmark_crab_image_runs_each_reasoning_effort_and_writes_summary(tmp_path: Path):
    target = tmp_path / "target.png"
    refs = {
        "european_green_crab": tmp_path / "green.png",
        "native_rock_crab": tmp_path / "rock.png",
        "jonah_crab": tmp_path / "jonah.png",
    }
    _write_image(target, (30, 80, 120))
    _write_image(refs["european_green_crab"], (20, 60, 50))
    _write_image(refs["native_rock_crab"], (80, 110, 160))
    _write_image(refs["jonah_crab"], (90, 130, 190))
    fake_client = _FakeClient()

    outputs = benchmark_crab_image(
        CrabCounterConfig(
            image_path=target,
            reference_paths=refs,
            output_dir=tmp_path / "bench",
            model="test-vision-model",
        ),
        efforts=("low", "high"),
        client=fake_client,
    )

    assert len(outputs.runs) == 2
    assert outputs.summary_json.exists()
    assert outputs.summary_csv.exists()
    assert [call["reasoning"]["effort"] for call in fake_client.responses.calls] == ["low", "high"]
    assert [run.result.reasoning_effort for run in outputs.runs] == ["low", "high"]
    assert all(run.result.analysis_seconds >= 0.0 for run in outputs.runs)
    summary = json.loads(outputs.summary_json.read_text(encoding="utf-8"))
    assert [run["reasoning_effort"] for run in summary["runs"]] == ["low", "high"]
    assert all("analysis_seconds" in run for run in summary["runs"])
