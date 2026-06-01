import argparse

import pytest

from analysis_workspace import set_active_workspace_root
from tools.crab_yolo_predict import (
    YoloPrediction,
    _boxes_text,
    _deduplicate_predictions,
    _parse_crop_scales,
    build_parser,
    latest_trained_weights,
)


def test_crab_yolo_prediction_deduplicates_overlapping_boxes():
    predictions = [
        YoloPrediction(0, 0.91, (100.0, 100.0, 220.0, 220.0)),
        YoloPrediction(0, 0.84, (108.0, 104.0, 226.0, 218.0)),
        YoloPrediction(0, 0.76, (320.0, 100.0, 430.0, 230.0)),
    ]

    kept = _deduplicate_predictions(predictions)

    assert kept == [predictions[0], predictions[2]]


def test_crab_yolo_boxes_text_is_stable_csv_payload():
    predictions = [
        YoloPrediction(0, 0.9123, (10.2, 20.6, 100.4, 140.5)),
        YoloPrediction(0, 0.7, (200.0, 210.0, 260.0, 300.0)),
    ]

    assert _boxes_text(predictions) == "0:0.912:10:21:100:140;0:0.700:200:210:260:300"


def test_crab_yolo_prediction_default_has_no_detection_cap():
    args = build_parser().parse_args(["image.png"])

    assert args.max_detections == 0


def test_crab_yolo_prediction_default_confidence_keeps_low_confidence_steep_crabs_visible():
    args = build_parser().parse_args(["image.png"])

    assert args.conf == 0.20


def test_crab_yolo_prediction_defaults_to_multiscale_board_crops():
    args = build_parser().parse_args(["image.png"])

    assert _parse_crop_scales(args.board_crop_scales, single_scale=args.board_crop_scale) == (1.15, 1.55, 2.05)


def test_crab_yolo_prediction_legacy_single_crop_scale_overrides_multiscale():
    args = build_parser().parse_args(["image.png", "--board-crop-scale", "1.35", "--board-crop-scales", "1.15,1.55"])

    assert _parse_crop_scales(args.board_crop_scales, single_scale=args.board_crop_scale) == (1.35,)


def test_crab_yolo_prediction_rejects_too_small_crop_scale():
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_crop_scales("0.9,1.4")


def test_crab_yolo_prediction_prefers_promoted_production_weights(tmp_path):
    set_active_workspace_root(tmp_path)
    try:
        promoted = tmp_path / "models" / "crab_yolo" / "production" / "weights" / "best.pt"
        promoted.parent.mkdir(parents=True)
        promoted.write_text("promoted", encoding="utf-8")
        latest = tmp_path / "models" / "crab_yolo" / "newer_experiment" / "weights" / "best.pt"
        latest.parent.mkdir(parents=True)
        latest.write_text("experiment", encoding="utf-8")

        assert latest_trained_weights() == promoted.resolve()
    finally:
        set_active_workspace_root(None)
