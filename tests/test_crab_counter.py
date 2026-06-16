import json
from pathlib import Path

import cv2
import numpy as np

from triton_analysis.crab.counter import (
    CrabCounterConfig,
    analyze_crab_image,
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

    def create(self, **kwargs):
        self.kwargs = kwargs
        return type(
            "FakeResponse",
            (),
            {
                "output_text": json.dumps(
                    {
                        "image_width": 120,
                        "image_height": 90,
                        "count": 2,
                        "detections": [
                            {
                                "label": "european_green_crab",
                                "bbox": [10, 12, 42, 46],
                                "confidence": 0.91,
                                "notes": "left green crab",
                            },
                            {
                                "label": "european_green_crab",
                                "bbox": [70, 20, 110, 62],
                                "confidence": 0.87,
                                "notes": "right green crab",
                            },
                        ],
                        "summary": "Two European green crabs found.",
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
        "detections": [
            {"label": "native_rock_crab", "bbox": [0, 0, 50, 50], "confidence": 1.0, "notes": ""},
            {"label": "european_green_crab", "bbox": [100, 80, 20, -5], "confidence": 1.4, "notes": "reversed"},
            {"label": "european_green_crab", "bbox": [3, 4, 20, 22], "confidence": 0.6, "notes": ""},
        ],
        "summary": "Mixed crab labels.",
    }

    result = result_from_payload(
        payload,
        image_path=tmp_path / "target.png",
        image_size=(90, 70),
        model="test-model",
    )

    assert result.count == 2
    assert [detection.label for detection in result.detections] == ["european_green_crab", "european_green_crab"]
    assert result.detections[0].bbox == (20.0, 0.0, 90.0, 70.0)
    assert result.detections[0].confidence == 1.0
    assert result.detections[1].bbox == (3.0, 4.0, 20.0, 22.0)


def test_draw_crab_count_result_writes_annotated_image(tmp_path: Path):
    target = tmp_path / "target.png"
    _write_image(target)
    result = result_from_payload(
        {
            "detections": [
                {"label": "european_green_crab", "bbox": [10, 10, 60, 45], "confidence": 0.9, "notes": ""}
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
    assert outputs.result_json.exists()
    assert outputs.annotated_image.exists()
    assert fake_client.responses.kwargs["model"] == "test-vision-model"
    content = fake_client.responses.kwargs["input"][0]["content"]
    assert content[0]["type"] == "input_text"
    assert "European green crab" in content[0]["text"]
    assert sum(1 for item in content if item["type"] == "input_image") == 4
    assert fake_client.responses.kwargs["text"]["format"]["type"] == "json_schema"
