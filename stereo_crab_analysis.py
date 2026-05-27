"""Stereo-assisted helpers for the crab detection applet."""

from __future__ import annotations

from typing import Mapping

import cv2
import numpy as np

from crab_detector_cv import (
    DEFAULT_UNWRAP_SIZE,
    axis_aligned_quad_from_box,
    build_crab_mask,
    build_reference_copy_color_mask,
    build_species_counts,
    build_video_sample_quality,
    classify_reference_copy_candidate,
    detect_crabs,
    detection_confidence_score,
    detection_species_signature,
    draw_crab_detections,
    merge_reference_copy_candidates,
    put_readable_text,
    score_video_detection_result,
)
from stereo_depth import (
    RectificationMaps,
    colorize_depth,
    compute_disparity,
    rectification_maps_from_artifact,
    rectify_stereo_images,
    reproject_disparity,
    sample_depth_point,
)


def default_max_depth_for_units(units: str) -> float | None:
    """Return a conservative max depth for filtering dense stereo points."""

    normalized = str(units or "").strip().lower()
    if normalized in {"mm", "millimeter", "millimeters"}:
        return 20000.0
    if normalized in {"cm", "centimeter", "centimeters"}:
        return 2000.0
    if normalized in {"m", "meter", "meters"}:
        return 20.0
    return None


def format_stereo_distance(value: float | None, units: str) -> str:
    """Format one calibration-space distance for operator-facing labels."""

    if value is None or not np.isfinite(float(value)):
        return "-"
    value = float(value)
    normalized = str(units or "").strip().lower()
    magnitude = abs(value)
    if normalized in {"mm", "millimeter", "millimeters"}:
        if magnitude >= 1000.0:
            return f"{value / 1000.0:.2f} m"
        if magnitude >= 10.0:
            return f"{value / 10.0:.1f} cm"
        return f"{value:.1f} mm"
    if normalized in {"cm", "centimeter", "centimeters"}:
        if magnitude >= 100.0:
            return f"{value / 100.0:.2f} m"
        return f"{value:.1f} cm"
    if normalized in {"m", "meter", "meters"}:
        return f"{value:.2f} m"
    suffix = f" {units}" if str(units or "").strip() else ""
    return f"{value:.1f}{suffix}"


def _detection_center(detection: Mapping) -> tuple[float, float]:
    box = detection.get("original_box")
    if box is not None:
        x, y, width, height = [float(value) for value in np.asarray(box).reshape(-1)[:4]]
        return x + width * 0.5, y + height * 0.5

    quad = detection.get("original_quad")
    if quad is not None:
        points = np.asarray(quad, dtype=np.float64).reshape(-1, 2)
        return float(np.mean(points[:, 0])), float(np.mean(points[:, 1]))

    raise ValueError("Detection is missing original image geometry")


def _detection_quad(detection: Mapping) -> np.ndarray:
    quad = detection.get("original_quad")
    if quad is not None:
        return np.round(np.asarray(quad, dtype=np.float32).reshape(-1, 2)).astype(np.int32)

    box = detection.get("original_box")
    if box is None:
        raise ValueError("Detection is missing original image geometry")
    x, y, width, height = [int(round(float(value))) for value in np.asarray(box).reshape(-1)[:4]]
    return np.array(
        [[x, y], [x + width, y], [x + width, y + height], [x, y + height]],
        dtype=np.int32,
    )


def _relaxed_reference_copy_candidate_boxes(image: np.ndarray) -> list[dict]:
    """Extract partial reference-copy candidates for stereo validation."""

    height, width = image.shape[:2]
    image_area = int(height * width)
    masks = [build_crab_mask(image), build_reference_copy_color_mask(image)]
    raw_candidates: list[dict] = []

    min_area = max(250, int(image_area * 0.00012))
    max_area = int(image_area * 0.030)
    min_box_area = max(500, int(image_area * 0.00035))
    min_width = max(14, int(max(height, width) * 0.010))
    min_height = max(12, int(max(height, width) * 0.010))
    max_width = int(width * 0.22)
    max_height = int(height * 0.24)

    for mask in masks:
        component_count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, 8)
        for component_index in range(1, component_count):
            x, y, box_width, box_height, area = [int(value) for value in stats[component_index]]
            if area < min_area or area > max_area:
                continue
            if box_width < min_width or box_height < min_height:
                continue
            if box_width > max_width or box_height > max_height:
                continue
            if box_width * box_height < min_box_area:
                continue

            aspect_ratio = box_width / max(1, box_height)
            fill_ratio = area / float(max(1, box_width * box_height))
            if aspect_ratio > 4.2 or aspect_ratio < 0.16:
                continue
            if fill_ratio < 0.08:
                continue

            raw_candidates.append(
                {
                    "box": np.array([x, y, box_width, box_height], dtype=np.int32),
                    "area": int(area),
                }
            )

    return merge_reference_copy_candidates(raw_candidates)


def _candidate_center(candidate: Mapping) -> tuple[float, float]:
    box = candidate.get("box", candidate.get("original_box"))
    x, y, width, height = [float(value) for value in np.asarray(box).reshape(-1)[:4]]
    return x + width * 0.5, y + height * 0.5


def _candidate_record(image: np.ndarray, candidate: Mapping, index: int) -> dict:
    box = np.asarray(candidate["box"], dtype=np.int32).reshape(4)
    classification = classify_reference_copy_candidate(image, box)
    detection = {
        "index": int(index),
        "unwrapped_box": box.copy(),
        "unwrapped_quad": axis_aligned_quad_from_box(box).astype(np.float32),
        "original_quad": axis_aligned_quad_from_box(box),
        "original_box": box.copy(),
        "area": int(candidate.get("area", int(box[2] * box[3]))),
        "classification": classification,
    }
    detection["candidate_confidence"] = detection_confidence_score(detection)
    return detection


def _best_feature_label(classification: Mapping) -> str | None:
    feature_scores = classification.get("copy_feature_scores") or classification.get("board_feature_scores") or {}
    if not feature_scores:
        return None
    return max(feature_scores.items(), key=lambda item: float(item[1].get("score", 0.0)))[0]


def _combine_stereo_classification(left_classification: Mapping, right_classification: Mapping) -> dict:
    labels = {"european_green", "native_rock", "jonah", "other"}
    combined_scores = {label: 0.0 for label in labels}

    for classification in (left_classification, right_classification):
        label = str(classification.get("label") or "other")
        combined_scores[label if label in labels else "other"] += 3.0
        color_label = str(classification.get("copy_color_label") or "")
        if color_label in labels:
            combined_scores[color_label] += 0.75
        feature_scores = classification.get("copy_feature_scores") or {}
        for feature_label, score in feature_scores.items():
            if feature_label in combined_scores:
                combined_scores[feature_label] += float(score.get("score", 0.0)) * 0.35
                combined_scores[feature_label] += min(12.0, float(score.get("inliers", 0))) * 0.18

    label = max(combined_scores.items(), key=lambda item: (item[1], item[0] == "european_green"))[0]
    return {
        "label": label,
        "is_european_green": label == "european_green",
        "stereo_combined_scores": combined_scores,
        "stereo_left_label": left_classification.get("label", "other"),
        "stereo_right_label": right_classification.get("label", "other"),
        "stereo_left_feature_label": _best_feature_label(left_classification),
        "stereo_right_feature_label": _best_feature_label(right_classification),
        "left_classification": dict(left_classification),
        "right_classification": dict(right_classification),
    }


def _stereo_candidate_match_score(left_detection: Mapping, right_detection: Mapping) -> tuple[float, dict] | None:
    left_center = _candidate_center(left_detection)
    right_center = _candidate_center(right_detection)
    disparity = left_center[0] - right_center[0]
    vertical_error = abs(left_center[1] - right_center[1])
    left_box = np.asarray(left_detection["original_box"], dtype=np.float64)
    right_box = np.asarray(right_detection["original_box"], dtype=np.float64)

    if disparity < 12.0 or disparity > 420.0:
        return None
    y_tolerance = max(18.0, 0.45 * max(left_box[3], right_box[3]))
    if vertical_error > y_tolerance:
        return None

    width_ratio = left_box[2] / max(1.0, right_box[2])
    height_ratio = left_box[3] / max(1.0, right_box[3])
    if width_ratio < 0.35 or width_ratio > 2.85 or height_ratio < 0.30 or height_ratio > 3.20:
        return None

    left_confidence = float(left_detection.get("candidate_confidence", 0.0))
    right_confidence = float(right_detection.get("candidate_confidence", 0.0))
    label_bonus = 0.0
    if left_detection["classification"].get("label") == right_detection["classification"].get("label"):
        label_bonus = 0.25
    score = (
        left_confidence
        + right_confidence
        + label_bonus
        + max(0.0, 1.0 - vertical_error / max(1.0, y_tolerance)) * 0.45
    )
    return score, {
        "right_box": np.asarray(right_detection["original_box"], dtype=np.int32).tolist(),
        "right_quad": np.asarray(right_detection["original_quad"], dtype=np.int32).tolist(),
        "right_confidence": right_confidence,
        "left_confidence": left_confidence,
        "disparity_px": float(disparity),
        "vertical_error_px": float(left_center[1] - right_center[1]),
        "match_score": float(score),
    }


def _cluster_stereo_matches(detections: list[dict], image_shape: tuple[int, int, int]) -> list[dict]:
    if len(detections) <= 2:
        return detections

    height, width = image_shape[:2]
    distance_threshold = max(190.0, 0.20 * max(height, width))
    centers = [np.array(_detection_center(detection), dtype=np.float32) for detection in detections]
    seen = [False] * len(detections)
    components: list[list[int]] = []
    for start_index in range(len(detections)):
        if seen[start_index]:
            continue
        stack = [start_index]
        seen[start_index] = True
        component: list[int] = []
        while stack:
            index = stack.pop()
            component.append(index)
            for neighbor_index in range(len(detections)):
                if seen[neighbor_index]:
                    continue
                if np.linalg.norm(centers[index] - centers[neighbor_index]) <= distance_threshold:
                    seen[neighbor_index] = True
                    stack.append(neighbor_index)
        components.append(component)

    best_component = max(
        components,
        key=lambda component: (
            len(component),
            sum(float(detections[index].get("candidate_confidence", 0.0)) for index in component),
            sum(int(detections[index].get("area", 0)) for index in component),
        ),
    )
    if len(best_component) < 2:
        return detections
    return [detections[index] for index in best_component]


def detect_stereo_reference_copy_crabs(
    left_rect_bgr: np.ndarray,
    right_rect_bgr: np.ndarray,
    *,
    min_match_score: float = 0.95,
    min_matches: int = 2,
) -> dict | None:
    """Detect reference-copy crabs by requiring stereo candidate agreement."""

    left_candidates = [
        _candidate_record(left_rect_bgr, candidate, index)
        for index, candidate in enumerate(_relaxed_reference_copy_candidate_boxes(left_rect_bgr), start=1)
    ]
    right_candidates = [
        _candidate_record(right_rect_bgr, candidate, index)
        for index, candidate in enumerate(_relaxed_reference_copy_candidate_boxes(right_rect_bgr), start=1)
    ]
    if not left_candidates or not right_candidates:
        return None

    possible_matches = []
    for left_detection in left_candidates:
        if float(left_detection.get("candidate_confidence", 0.0)) < 0.25:
            continue
        for right_detection in right_candidates:
            if float(right_detection.get("candidate_confidence", 0.0)) < 0.22:
                continue
            match = _stereo_candidate_match_score(left_detection, right_detection)
            if match is not None:
                score, payload = match
                possible_matches.append((score, left_detection, right_detection, payload))

    matched_left: set[int] = set()
    matched_right: set[int] = set()
    detections: list[dict] = []
    for score, left_detection, right_detection, payload in sorted(possible_matches, key=lambda item: item[0], reverse=True):
        if score < float(min_match_score):
            continue
        left_key = id(left_detection)
        right_key = id(right_detection)
        if left_key in matched_left or right_key in matched_right:
            continue
        matched_left.add(left_key)
        matched_right.add(right_key)
        detection = dict(left_detection)
        detection["classification"] = _combine_stereo_classification(
            left_detection["classification"],
            right_detection["classification"],
        )
        detection["stereo_match"] = payload
        detection["candidate_confidence"] = max(
            float(left_detection.get("candidate_confidence", 0.0)),
            float(right_detection.get("candidate_confidence", 0.0)),
        )
        detections.append(detection)

    detections = _cluster_stereo_matches(detections, left_rect_bgr.shape)
    if len(detections) < int(min_matches):
        return None

    detections = sorted(
        detections,
        key=lambda detection: (
            int(detection["original_box"][1]),
            int(detection["original_box"][0]),
        ),
    )
    candidate_mask = np.zeros(left_rect_bgr.shape[:2], dtype=np.uint8)
    for index, detection in enumerate(detections, start=1):
        detection["index"] = index
        x, y, box_width, box_height = [int(value) for value in detection["original_box"]]
        candidate_mask[y : y + box_height, x : x + box_width] = 255

    species_counts = build_species_counts(detections)
    green_count = int(species_counts.get("european_green", 0))
    return {
        "board_polygon": None,
        "board_polygon_source": "stereo_reference_copy",
        "detector": "stereo_reference_copy",
        "unwrapped_image": left_rect_bgr.copy(),
        "classification_gains": np.ones(3, dtype=np.float32),
        "unwrapped_mask": candidate_mask,
        "transform": np.eye(3, dtype=np.float32),
        "detections": detections,
        "count": len(detections),
        "green_count": green_count,
        "other_count": len(detections) - green_count,
        "species_counts": species_counts,
    }


def better_stereo_detection_result(current: dict | None, candidate: dict | None) -> dict | None:
    """Return the stronger detection result between normal and stereo paths."""

    if candidate is None:
        return current
    if current is None:
        return candidate
    current_score = score_video_detection_result(current)
    candidate_score = score_video_detection_result(candidate)
    current_signature = detection_species_signature(current)
    candidate_signature = detection_species_signature(candidate)
    if candidate["count"] > current["count"] and sum(candidate_signature[:3]) >= sum(current_signature[:3]):
        return candidate
    if candidate_score > current_score and candidate["count"] >= max(2, current["count"] - 1):
        return candidate
    return current


def attach_stereo_depth_to_detections(
    detection_result: dict,
    points_3d: np.ndarray,
    disparity: np.ndarray,
    valid_depth: np.ndarray,
    *,
    units: str = "",
    sample_radius: int = 5,
) -> dict:
    """Attach stereo depth samples to each detection in ``detection_result``."""

    detections = list(detection_result.get("detections") or [])
    depths: list[float] = []
    for detection in detections:
        try:
            center = _detection_center(detection)
            sample = sample_depth_point(
                points_3d,
                disparity,
                valid_depth,
                center,
                radius=sample_radius,
            )
        except Exception as exc:
            detection["stereo_depth"] = {
                "available": False,
                "reason": str(exc),
            }
            continue

        point = np.asarray(sample.point, dtype=np.float64)
        depth_units = float(abs(point[2]))
        depths.append(depth_units)
        detection["stereo_depth"] = {
            "available": True,
            "pixel": [int(sample.pixel[0]), int(sample.pixel[1])],
            "point": [float(point[0]), float(point[1]), float(point[2])],
            "depth_units": depth_units,
            "depth_label": format_stereo_distance(depth_units, units),
            "disparity_px": float(sample.disparity),
            "sample_count": int(sample.sample_count),
        }

    summary = {
        "available_count": int(len(depths)),
        "detection_count": int(len(detections)),
        "median_depth_units": None if not depths else float(np.median(np.asarray(depths, dtype=np.float64))),
        "units": str(units or ""),
        "sample_radius": int(sample_radius),
    }
    summary["median_depth_label"] = format_stereo_distance(summary["median_depth_units"], units)
    detection_result["stereo_depth_summary"] = summary
    return summary


def stereo_depth_summary_text(detection_result: Mapping) -> str:
    """Return a compact summary of stereo depth coverage for crab detections."""

    summary = detection_result.get("stereo_depth_summary") or {}
    available = int(summary.get("available_count") or 0)
    total = int(summary.get("detection_count") or 0)
    if total <= 0:
        return "stereo depth: no detections"
    if available <= 0:
        return f"stereo depth: 0/{total}"
    return f"stereo depth: {available}/{total}, median {summary.get('median_depth_label', '-')}"


def draw_stereo_crab_detections(
    image_bgr: np.ndarray,
    detection_result: Mapping,
    *,
    units: str = "",
) -> np.ndarray:
    """Draw crab detections plus per-crab stereo depth labels."""

    annotated = draw_crab_detections(image_bgr, detection_result)
    for detection in detection_result.get("detections") or []:
        stereo_depth = detection.get("stereo_depth") or {}
        if not stereo_depth.get("available"):
            continue
        quad = _detection_quad(detection)
        label_anchor = quad[np.argmax(quad[:, 1] - quad[:, 0])]
        label = f"Z {stereo_depth.get('depth_label') or format_stereo_distance(stereo_depth.get('depth_units'), units)}"
        put_readable_text(
            annotated,
            label,
            (int(label_anchor[0]), int(label_anchor[1] + 24)),
            0.58,
            2,
            text_color=(70, 255, 255),
        )
    return annotated


def draw_stereo_depth_overlay(
    depth_preview_bgr: np.ndarray,
    detection_result: Mapping | None,
    *,
    units: str = "",
) -> np.ndarray:
    """Draw crab detection outlines on an already colorized depth preview."""

    annotated = depth_preview_bgr.copy()
    if detection_result is None:
        return annotated

    summary = stereo_depth_summary_text(detection_result)
    put_readable_text(annotated, summary, (20, 40), 0.75, 2, text_color=(70, 255, 255))

    for detection in detection_result.get("detections") or []:
        quad = _detection_quad(detection)
        cv2.polylines(annotated, [quad.reshape(-1, 1, 2)], True, (70, 255, 255), 3)
        stereo_depth = detection.get("stereo_depth") or {}
        label = str(detection.get("index", ""))
        if stereo_depth.get("available"):
            label = f"#{label} {stereo_depth.get('depth_label') or format_stereo_distance(stereo_depth.get('depth_units'), units)}"
        anchor = quad[np.argmin(quad[:, 0] + quad[:, 1])]
        put_readable_text(
            annotated,
            label,
            (int(anchor[0]), int(max(24, anchor[1] - 8))),
            0.55,
            2,
            text_color=(70, 255, 255),
        )
    return annotated


def analyze_stereo_crab_pair(
    left_bgr: np.ndarray,
    right_bgr: np.ndarray,
    calibration_artifact: Mapping | None,
    *,
    rectification_maps: RectificationMaps | None = None,
    force_square: bool = True,
    unwrap_size: tuple[int, int] = DEFAULT_UNWRAP_SIZE,
    board_polygon: np.ndarray | None = None,
    min_disparity: int = 0,
    num_disparities: int = 320,
    block_size: int = 7,
    sample_radius: int = 5,
    max_abs_depth: float | None = None,
    compute_depth: bool = True,
) -> dict:
    """Run crab detection on a calibrated stereo pair and attach depth samples."""

    maps = rectification_maps
    if calibration_artifact is not None and maps is None:
        maps = rectification_maps_from_artifact(calibration_artifact)

    if maps is None:
        left_view = left_bgr.copy()
        right_view = right_bgr.copy()
        units = ""
    else:
        left_view, right_view = rectify_stereo_images(left_bgr, right_bgr, maps)
        units = maps.units

    detection_result = detect_crabs(
        left_view,
        force_square=force_square,
        unwrap_size=unwrap_size,
        board_polygon=board_polygon,
    )
    stereo_candidate_result = None
    if board_polygon is None and maps is not None:
        stereo_candidate_result = detect_stereo_reference_copy_crabs(left_view, right_view)
        detection_result = better_stereo_detection_result(detection_result, stereo_candidate_result)

    disparity = None
    disparity_valid = None
    points_3d = None
    valid_depth = None
    depth_preview = None
    annotated_depth = None
    if maps is not None and compute_depth:
        disparity, disparity_valid = compute_disparity(
            left_view,
            right_view,
            min_disparity=min_disparity,
            num_disparities=num_disparities,
            block_size=block_size,
            preprocess="clahe",
            left_right_check=True,
        )
        if max_abs_depth is None:
            max_abs_depth = default_max_depth_for_units(units)
        points_3d, valid_depth = reproject_disparity(
            disparity,
            maps.q,
            disparity_valid,
            max_abs_depth=max_abs_depth,
        )
        depth_preview = colorize_depth(points_3d, valid_depth)
        if detection_result is not None:
            attach_stereo_depth_to_detections(
                detection_result,
                points_3d,
                disparity,
                valid_depth,
                units=units,
                sample_radius=sample_radius,
            )
        annotated_depth = draw_stereo_depth_overlay(depth_preview, detection_result, units=units)

    annotated_left = (
        None
        if detection_result is None
        else draw_stereo_crab_detections(left_view, detection_result, units=units)
    )
    quality = build_video_sample_quality(left_view, detection_result)

    return {
        "left_rectified": left_view,
        "right_rectified": right_view,
        "detection_result": detection_result,
        "stereo_candidate_result": stereo_candidate_result,
        "annotated_left": annotated_left,
        "disparity": disparity,
        "disparity_valid": disparity_valid,
        "points_3d": points_3d,
        "valid_depth": valid_depth,
        "depth_preview": depth_preview,
        "annotated_depth": annotated_depth,
        "units": units,
        "quality": quality,
    }
