import json
from pathlib import Path

import cv2
import numpy as np

from triton_analysis.crab.counter import (
    CrabCounterConfig,
    analyze_crab_image,
    analyze_crab_image_pipeline,
    auto_preprocess_crab_target_image,
    benchmark_crab_image,
    benchmark_crab_image_pipeline,
    detect_crab_board_homography,
    discover_crab_board_reference_paths,
    draw_crab_count_result,
    preprocess_crab_target_image,
    result_from_payload,
    transform_crab_count_result,
    write_reference_atlas,
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
                                "class_scores": {
                                    "european_green_crab": 0.93,
                                    "native_rock_crab": 0.12,
                                    "jonah_crab": 0.08,
                                },
                                "closest_non_target": "native_rock_crab",
                                "decision_margin": 0.81,
                                "accepted_as_target": True,
                                "notes": "left green crab",
                            },
                            {
                                "label": "native_rock_crab",
                                "bbox": [44, 14, 62, 39],
                                "confidence": 0.94,
                                "target_match_confidence": 0.12,
                                "class_scores": {
                                    "european_green_crab": 0.12,
                                    "native_rock_crab": 0.94,
                                    "jonah_crab": 0.25,
                                },
                                "closest_non_target": "native_rock_crab",
                                "decision_margin": -0.82,
                                "accepted_as_target": False,
                                "notes": "hard negative",
                            },
                            {
                                "label": "european_green_crab",
                                "bbox": [70, 20, 110, 62],
                                "confidence": 0.87,
                                "target_match_confidence": 0.88,
                                "class_scores": {
                                    "european_green_crab": 0.88,
                                    "native_rock_crab": 0.18,
                                    "jonah_crab": 0.10,
                                },
                                "closest_non_target": "native_rock_crab",
                                "decision_margin": 0.70,
                                "accepted_as_target": True,
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


class _FakePipelineResponses:
    def __init__(self):
        self.calls = []
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        self.calls.append(kwargs)
        schema_name = kwargs["text"]["format"]["name"]
        if schema_name == "crab_candidate_detector":
            payload = {
                "image_width": 120,
                "image_height": 90,
                "candidates": [
                    {"candidate_id": 1, "bbox": [10, 12, 42, 46], "confidence": 0.95, "single_crab": True, "notes": "printed crab"},
                    {"candidate_id": 2, "bbox": [44, 14, 62, 39], "confidence": 0.93, "single_crab": True, "notes": "printed crab"},
                    {"candidate_id": 3, "bbox": [70, 20, 110, 62], "confidence": 0.91, "single_crab": True, "notes": "printed crab"},
                    {
                        "candidate_id": 4,
                        "bbox": [8, 10, 64, 48],
                        "confidence": 0.6,
                        "single_crab": False,
                        "notes": "merged region containing two adjacent printed crabs",
                    },
                ],
                "summary": "Three single-crab candidates plus one rejected merged region.",
            }
        elif schema_name == "crab_candidate_classifier":
            payload = {
                "classifications": [
                    {
                        "candidate_id": 1,
                        "label": "european_green_crab",
                        "confidence": 0.91,
                        "target_match_confidence": 0.93,
                        "class_scores": {
                            "european_green_crab": 0.93,
                            "native_rock_crab": 0.12,
                            "jonah_crab": 0.08,
                        },
                        "closest_non_target": "native_rock_crab",
                        "decision_margin": 0.81,
                        "egc_supporting_cues": ["compact carapace", "thin lateral legs"],
                        "non_target_supporting_cues": [],
                        "visible_cues_sufficient": True,
                        "accepted_as_target": True,
                        "notes": "green cue vs rock",
                    },
                    {
                        "candidate_id": 2,
                        "label": "native_rock_crab",
                        "confidence": 0.94,
                        "target_match_confidence": 0.16,
                        "class_scores": {
                            "european_green_crab": 0.16,
                            "native_rock_crab": 0.94,
                            "jonah_crab": 0.15,
                        },
                        "closest_non_target": "native_rock_crab",
                        "decision_margin": -0.78,
                        "egc_supporting_cues": [],
                        "non_target_supporting_cues": ["native rock silhouette"],
                        "visible_cues_sufficient": True,
                        "accepted_as_target": False,
                        "notes": "rock silhouette",
                    },
                    {
                        "candidate_id": 3,
                        "label": "european_green_crab",
                        "confidence": 0.9,
                        "target_match_confidence": 0.91,
                        "class_scores": {
                            "european_green_crab": 0.91,
                            "native_rock_crab": 0.13,
                            "jonah_crab": 0.08,
                        },
                        "closest_non_target": "native_rock_crab",
                        "decision_margin": 0.78,
                        "egc_supporting_cues": ["leg layout", "compact body"],
                        "non_target_supporting_cues": [],
                        "visible_cues_sufficient": True,
                        "accepted_as_target": True,
                        "notes": "leg layout vs rock",
                    },
                ],
                "summary": "Two candidates are European green crabs.",
            }
        else:
            raise AssertionError(f"unexpected schema name {schema_name}")
        return type("FakePipelineResponse", (), {"output_text": json.dumps(payload)})()


class _FakePipelineClient:
    def __init__(self):
        self.responses = _FakePipelineResponses()


class _FakeBoardResponses:
    def __init__(self):
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return type(
            "FakeBoardResponse",
            (),
            {
                "output_text": json.dumps(
                    {
                        "image_width": 220,
                        "image_height": 140,
                        "board_visible": True,
                        "confidence": 0.94,
                        "top_left": {"x": 25, "y": 18},
                        "top_right": {"x": 182, "y": 20},
                        "bottom_right": {"x": 174, "y": 110},
                        "bottom_left": {"x": 32, "y": 114},
                        "notes": "clear board outline",
                    }
                )
            },
        )()


class _FakeBoardClient:
    def __init__(self):
        self.responses = _FakeBoardResponses()


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
                "label": "european_green_crab",
                "bbox": [50, 10, 70, 30],
                "confidence": 0.92,
                "target_match_confidence": 0.9,
                "class_scores": {
                    "european_green_crab": 0.9,
                    "native_rock_crab": 0.82,
                    "jonah_crab": 0.2,
                },
                "closest_non_target": "native_rock_crab",
                "decision_margin": 0.08,
                "accepted_as_target": True,
                "notes": "too close to rock crab",
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
    assert len(result.candidates) == 5
    assert [detection.label for detection in result.detections] == ["european_green_crab"]
    assert result.detections[0].bbox == (20.0, 0.0, 90.0, 70.0)
    assert result.detections[0].confidence == 1.0
    assert result.detections[0].target_match_confidence == 0.9
    assert result.detections[0].accepted_as_target is True
    assert result.candidates[3].accepted_as_target is False


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
    assert outputs.result.target_margin_threshold == 0.15
    assert outputs.result.analysis_seconds >= 0.0
    assert outputs.result_json.exists()
    assert outputs.annotated_image.exists()
    assert outputs.artifact_manifest == outputs.output_dir / "run_manifest.json"
    assert outputs.artifact_manifest.exists()
    assert fake_client.responses.kwargs["model"] == "test-vision-model"
    assert fake_client.responses.kwargs["reasoning"] == {"effort": "xhigh"}
    content = fake_client.responses.kwargs["input"][0]["content"]
    assert content[0]["type"] == "input_text"
    assert "European green crab" in content[0]["text"]
    assert "hard negatives" in content[0]["text"]
    assert "Species identification is more important" in content[0]["text"]
    assert "0.85" in content[0]["text"]
    assert "class_scores" in content[0]["text"]
    assert "Reference atlas" in content[1]["text"]
    assert "Target frame" in content[3]["text"]
    assert sum(1 for item in content if item["type"] == "input_image") == 2
    assert fake_client.responses.kwargs["prompt_cache_key"] == "triton_analysis_crab_counter_v2"
    assert fake_client.responses.kwargs["text"]["format"]["type"] == "json_schema"
    assert fake_client.responses.kwargs["text"]["verbosity"] == "low"
    schema = fake_client.responses.kwargs["text"]["format"]["schema"]
    assert "candidates" in schema["properties"]
    candidate_schema = schema["properties"]["candidates"]["items"]["properties"]
    assert "native_rock_crab" in candidate_schema["label"]["enum"]
    assert "target_match_confidence" in candidate_schema
    assert "class_scores" in candidate_schema
    assert "decision_margin" in candidate_schema
    assert "accepted_as_target" in candidate_schema
    written = json.loads(outputs.result_json.read_text(encoding="utf-8"))
    assert written["analysis_seconds"] >= 0.0
    assert written["detections"][0]["accepted_as_target"] is True
    assert written["artifact_manifest"] == str(outputs.artifact_manifest)
    manifest = json.loads(outputs.artifact_manifest.read_text(encoding="utf-8"))
    stages = {stage["stage"] for stage in manifest["stages"]}
    assert {"single_request_count", "final_outputs"} <= stages
    request_summary = json.loads((outputs.output_dir / "artifacts" / "single_request_count_request.json").read_text(encoding="utf-8"))
    saved_content = request_summary["request"]["input"][0]["content"]
    assert saved_content[2]["image_url"]["omitted"] == "base64_image_data"
    assert saved_content[4]["image_url"]["omitted"] == "base64_image_data"
    response_payload = json.loads((outputs.output_dir / "artifacts" / "single_request_count_response.json").read_text(encoding="utf-8"))
    assert response_payload["count"] == 2


def test_analyze_crab_image_pipeline_detects_then_classifies_crops(tmp_path: Path):
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
    fake_client = _FakePipelineClient()

    outputs = analyze_crab_image_pipeline(
        CrabCounterConfig(
            image_path=target,
            reference_paths=refs,
            output_dir=tmp_path / "pipeline",
            model="test-vision-model",
            reasoning_effort="high",
        ),
        client=fake_client,
    )

    assert outputs.result.count == 2
    assert len(outputs.result.candidates) == 3
    assert outputs.result.detections[0].bbox == (10.0, 12.0, 42.0, 46.0)
    assert outputs.result.analysis_seconds >= 0.0
    assert outputs.result_json.exists()
    assert outputs.annotated_image.exists()
    assert outputs.artifact_manifest == outputs.output_dir / "run_manifest.json"
    assert outputs.artifact_manifest.exists()
    detection_json = outputs.output_dir / "pipeline" / "target_candidate_boxes.json"
    contact_sheet = outputs.output_dir / "target_candidate_contact_sheet.png"
    assert detection_json.exists()
    assert contact_sheet.exists()
    assert [call["text"]["format"]["name"] for call in fake_client.responses.calls] == [
        "crab_candidate_detector",
        "crab_candidate_classifier",
    ]
    assert fake_client.responses.calls[0]["reasoning"] == {"effort": "low"}
    assert fake_client.responses.calls[1]["reasoning"] == {"effort": "high"}
    detector_call = fake_client.responses.calls[0]
    detector_prompt = detector_call["input"][0]["content"][0]["text"]
    assert "Never group two printed crabs into one candidate box" in detector_prompt
    assert "overlapping boxes" in detector_prompt
    assert "first identify each visible carapace/body center" in detector_prompt
    assert "wrong if it encloses two body centers" in detector_prompt
    detector_schema = detector_call["text"]["format"]["schema"]
    detector_candidate_props = detector_schema["properties"]["candidates"]["items"]["properties"]
    assert "single_crab" in detector_candidate_props
    assert "single_crab" in detector_schema["properties"]["candidates"]["items"]["required"]
    assert detector_candidate_props["bbox"]["minItems"] == 4
    assert detector_call["prompt_cache_key"] == "triton_analysis_crab_candidate_detector_v3"
    classifier_content = fake_client.responses.calls[1]["input"][0]["content"]
    assert "classification only" in classifier_content[0]["text"]
    assert "Confidence is a species-match confidence conditional on the visible crop" in classifier_content[0]["text"]
    assert "Candidate 1" in classifier_content[0]["text"]
    assert sum(1 for item in classifier_content if item["type"] == "input_image") == 2
    classifier_schema = fake_client.responses.calls[1]["text"]["format"]["schema"]
    classification_props = classifier_schema["properties"]["classifications"]["items"]["properties"]
    assert "egc_supporting_cues" in classification_props
    assert "non_target_supporting_cues" in classification_props
    assert "visible_cues_sufficient" in classification_props
    written = json.loads(outputs.result_json.read_text(encoding="utf-8"))
    assert written["pipeline"]["mode"] == "detect_then_classify_crops"
    assert written["pipeline"]["detector_candidate_count"] == 3
    assert written["artifact_manifest"] == str(outputs.artifact_manifest)
    manifest = json.loads(outputs.artifact_manifest.read_text(encoding="utf-8"))
    stages = {stage["stage"] for stage in manifest["stages"]}
    assert {"candidate_detection", "candidate_detection_outputs", "candidate_classification", "final_outputs"} <= stages
    detector_request = json.loads((outputs.output_dir / "artifacts" / "candidate_detection_request.json").read_text(encoding="utf-8"))
    assert detector_request["request"]["input"][0]["content"][1]["image_url"]["omitted"] == "base64_image_data"
    classifier_response = json.loads(
        (outputs.output_dir / "artifacts" / "candidate_classification_response.json").read_text(encoding="utf-8")
    )
    assert len(classifier_response["classifications"]) == 3
    detection_stage = next(stage for stage in manifest["stages"] if stage["stage"] == "candidate_detection_outputs")
    assert detection_stage["files"]["contact_sheet"] == str(contact_sheet)


def test_benchmark_crab_image_pipeline_reuses_detector_stage(tmp_path: Path):
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
    fake_client = _FakePipelineClient()

    outputs = benchmark_crab_image_pipeline(
        CrabCounterConfig(
            image_path=target,
            reference_paths=refs,
            output_dir=tmp_path / "pipeline_bench",
            model="test-vision-model",
        ),
        efforts=("low", "high"),
        client=fake_client,
    )

    assert len(outputs.runs) == 2
    assert [call["text"]["format"]["name"] for call in fake_client.responses.calls] == [
        "crab_candidate_detector",
        "crab_candidate_classifier",
        "crab_candidate_classifier",
    ]
    assert [call["reasoning"]["effort"] for call in fake_client.responses.calls] == ["low", "low", "high"]
    summary = json.loads(outputs.summary_json.read_text(encoding="utf-8"))
    assert summary["pipeline"]["detector_candidate_count"] == 3
    assert [run["reasoning_effort"] for run in summary["runs"]] == ["low", "high"]


def test_write_reference_atlas_writes_preview_image(tmp_path: Path):
    refs = {
        "european_green_crab": tmp_path / "green.png",
        "native_rock_crab": tmp_path / "rock.png",
        "jonah_crab": tmp_path / "jonah.png",
    }
    extra = {
        "european_green_crab": [tmp_path / "green_pool.png"],
        "native_rock_crab": [tmp_path / "rock_pool.png"],
        "jonah_crab": [tmp_path / "jonah_pool.png"],
    }
    _write_image(refs["european_green_crab"], (20, 60, 50))
    _write_image(refs["native_rock_crab"], (80, 110, 160))
    _write_image(refs["jonah_crab"], (90, 130, 190))
    _write_image(extra["european_green_crab"][0], (35, 55, 55))
    _write_image(extra["native_rock_crab"][0], (75, 100, 145))
    _write_image(extra["jonah_crab"][0], (95, 125, 180))

    output = write_reference_atlas(refs, tmp_path / "atlas.png", atlas_paths=extra)

    assert output.exists()
    atlas = cv2.imread(str(output), cv2.IMREAD_COLOR)
    assert atlas is not None
    assert atlas.shape[0] > 100
    assert atlas.shape[1] > 300


def test_preprocess_crab_target_image_writes_manual_crop_and_metadata(tmp_path: Path):
    target = tmp_path / "target.png"
    _write_image(target, (30, 80, 120), size=(200, 120))

    result = preprocess_crab_target_image(
        target,
        tmp_path / "preprocess",
        mode="manual_crop",
        crop_rect=(20, 15, 150, 95),
    )

    assert result.mode == "manual_crop"
    assert result.processed_image.exists()
    assert result.metadata_json.exists()
    cropped = cv2.imread(str(result.processed_image), cv2.IMREAD_COLOR)
    assert cropped is not None
    assert cropped.shape[:2] == (80, 130)
    metadata = json.loads(result.metadata_json.read_text(encoding="utf-8"))
    assert metadata["crop_bbox"] == [20, 15, 150, 95]
    assert metadata["source_size"] == [200, 120]
    assert metadata["output_size"] == [130, 80]


def test_transform_crab_count_result_maps_processed_boxes_to_source_coordinates(tmp_path: Path):
    processed_result = result_from_payload(
        {
            "candidates": [
                {
                    "label": "european_green_crab",
                    "bbox": [10, 12, 42, 46],
                    "confidence": 0.9,
                    "target_match_confidence": 0.9,
                    "class_scores": {
                        "european_green_crab": 0.9,
                        "native_rock_crab": 0.1,
                        "jonah_crab": 0.1,
                    },
                    "closest_non_target": "native_rock_crab",
                    "decision_margin": 0.8,
                    "accepted_as_target": True,
                    "notes": "",
                }
            ],
            "summary": "One.",
        },
        image_path=tmp_path / "processed.png",
        image_size=(130, 80),
        model="test-model",
    )

    source_result = transform_crab_count_result(
        processed_result,
        [[1, 0, 20], [0, 1, 15], [0, 0, 1]],
        source_image_path=tmp_path / "source.png",
        source_size=(200, 120),
    )

    assert source_result.image_path == tmp_path / "source.png"
    assert source_result.image_size == (200, 120)
    assert source_result.detections[0].bbox == (30.0, 27.0, 62.0, 61.0)
    assert source_result.candidates[0].bbox == (30.0, 27.0, 62.0, 61.0)


def test_preprocess_crab_target_image_writes_manual_homography_and_metadata(tmp_path: Path):
    target = tmp_path / "target.png"
    _write_image(target, (30, 80, 120), size=(220, 140))

    result = preprocess_crab_target_image(
        target,
        tmp_path / "preprocess",
        mode="manual_homography",
        homography_points=((25, 20), (180, 18), (170, 110), (35, 115)),
    )

    assert result.mode == "manual_homography"
    assert result.processed_image.exists()
    assert result.metadata_json.exists()
    warped = cv2.imread(str(result.processed_image), cv2.IMREAD_COLOR)
    assert warped is not None
    assert warped.shape[0] >= 80
    assert warped.shape[1] >= 130
    metadata = json.loads(result.metadata_json.read_text(encoding="utf-8"))
    assert len(metadata["clicked_points"]) == 4
    assert len(metadata["ordered_points"]) == 4
    assert len(metadata["source_to_processed_matrix"]) == 3
    assert len(metadata["processed_to_source_matrix"]) == 3


def test_detect_crab_board_homography_uses_fast_outline_prompt_and_schema(tmp_path: Path):
    target = tmp_path / "target.png"
    _write_image(target, (30, 80, 120), size=(220, 140))
    fake_client = _FakeBoardClient()

    result = detect_crab_board_homography(
        target,
        model="test-outline-model",
        reasoning_effort="xhigh",
        client=fake_client,
    )

    assert result.image_size == (220, 140)
    assert result.confidence == 0.94
    assert result.board_visible is True
    assert len(result.points) == 4
    assert result.analysis_seconds >= 0.0
    kwargs = fake_client.responses.kwargs
    assert kwargs["model"] == "test-outline-model"
    assert kwargs["reasoning"] == {"effort": "xhigh"}
    assert kwargs["prompt_cache_key"] == "triton_analysis_crab_board_homography_v1"
    content = kwargs["input"][0]["content"]
    assert "Locate the outer boundary" in content[0]["text"]
    assert "do not return crab bounding boxes" in content[0]["text"]
    assert sum(1 for item in content if item["type"] == "input_image") == 1
    assert kwargs["text"]["format"]["name"] == "crab_board_outline"
    schema = kwargs["text"]["format"]["schema"]
    assert "top_left" in schema["properties"]
    assert "bottom_right" in schema["required"]
    assert kwargs["text"]["verbosity"] == "low"


def test_detect_crab_board_homography_can_include_board_appearance_references(tmp_path: Path):
    target = tmp_path / "target.png"
    reference = tmp_path / "blank_board_pool.png"
    _write_image(target, (30, 80, 120), size=(220, 140))
    _write_image(reference, (210, 210, 205), size=(220, 140))
    fake_client = _FakeBoardClient()

    detect_crab_board_homography(
        target,
        model="test-outline-model",
        reasoning_effort="xhigh",
        board_reference_paths=(reference,),
        client=fake_client,
    )

    content = fake_client.responses.kwargs["input"][0]["content"]
    assert "Board appearance references only" in content[1]["text"]
    images = [item for item in content if item["type"] == "input_image"]
    assert len(images) == 2
    assert images[0]["detail"] == "low"
    assert images[-1]["detail"] == "high"
    assert "Target image for board-corner coordinates" in content[-2]["text"]


def test_discover_crab_board_reference_paths_reads_workspace_and_env(tmp_path: Path, monkeypatch):
    workspace_root = tmp_path / "Workspace"
    workspace_ref_dir = workspace_root / "data" / "crab board references"
    env_ref_dir = tmp_path / "env_refs"
    workspace_ref_dir.mkdir(parents=True)
    env_ref_dir.mkdir()
    workspace_ref = workspace_ref_dir / "workspace_board.png"
    env_ref = env_ref_dir / "env_board.jpg"
    _write_image(workspace_ref)
    _write_image(env_ref)
    monkeypatch.setenv("TRITON_ANALYSIS_CRAB_BOARD_REFERENCE_IMAGES", str(env_ref_dir))

    paths = discover_crab_board_reference_paths(workspace_root)

    assert env_ref in paths
    assert workspace_ref in paths


def test_auto_preprocess_crab_target_image_writes_detected_homography_metadata(tmp_path: Path):
    target = tmp_path / "target.png"
    _write_image(target, (30, 80, 120), size=(220, 140))
    fake_client = _FakeBoardClient()

    result = auto_preprocess_crab_target_image(
        target,
        tmp_path / "preprocess",
        model="test-outline-model",
        reasoning_effort="xhigh",
        client=fake_client,
    )

    assert result.mode == "auto_homography"
    assert result.processed_image.exists()
    assert result.metadata_json.exists()
    assert result.board_confidence == 0.94
    assert result.board_reasoning_effort == "xhigh"
    assert len(result.ordered_points) == 4
    warped = cv2.imread(str(result.processed_image), cv2.IMREAD_COLOR)
    assert warped is not None
    assert warped.shape[0] >= 80
    assert warped.shape[1] >= 130
    metadata = json.loads(result.metadata_json.read_text(encoding="utf-8"))
    assert metadata["mode"] == "auto_homography"
    assert len(metadata["detected_points"]) == 4
    assert metadata["board_outline"]["confidence"] == 0.94
    assert metadata["auto_board_detection_seconds"] >= 0.0
    manifest = json.loads((tmp_path / "preprocess" / "run_manifest.json").read_text(encoding="utf-8"))
    stages = {stage["stage"] for stage in manifest["stages"]}
    assert {"board_homography", "auto_homography_preprocess"} <= stages
    request_summary = json.loads(
        (tmp_path / "preprocess" / "artifacts" / "board_homography_request.json").read_text(encoding="utf-8")
    )
    assert request_summary["request"]["input"][0]["content"][-1]["image_url"]["omitted"] == "base64_image_data"


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
    assert outputs.artifact_manifest == outputs.output_dir / "run_manifest.json"
    assert outputs.artifact_manifest.exists()
    assert [call["reasoning"]["effort"] for call in fake_client.responses.calls] == ["low", "high"]
    assert [run.result.reasoning_effort for run in outputs.runs] == ["low", "high"]
    assert all(run.result.analysis_seconds >= 0.0 for run in outputs.runs)
    summary = json.loads(outputs.summary_json.read_text(encoding="utf-8"))
    assert summary["artifact_manifest"] == str(outputs.artifact_manifest)
    assert [run["reasoning_effort"] for run in summary["runs"]] == ["low", "high"]
    assert all("analysis_seconds" in run for run in summary["runs"])
