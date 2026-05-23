"""Stereo calibration and rectification helpers for saved TritonPilot captures."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np


PointPairs = Sequence[tuple[str | Path, str | Path]]
DEFAULT_CHARUCO_DICTIONARY = "DICT_5X5_1000"
CHARUCO_DICTIONARIES = [
    "DICT_4X4_50",
    "DICT_4X4_100",
    "DICT_5X5_100",
    "DICT_5X5_250",
    "DICT_5X5_1000",
    "DICT_6X6_250",
    "DICT_6X6_1000",
]


@dataclass(frozen=True)
class CheckerboardSpec:
    """OpenCV checkerboard spec using inner-corner counts."""

    columns: int
    rows: int
    square_size: float
    units: str = "cm"


@dataclass(frozen=True)
class CharucoBoardSpec:
    """ChArUco board spec using square counts, not inner-corner counts."""

    squares_x: int
    squares_y: int
    square_size: float
    marker_size: float
    dictionary: str = DEFAULT_CHARUCO_DICTIONARY
    units: str = "cm"


@dataclass(frozen=True)
class StereoObservations:
    """Matched calibration points from multiple stereo image pairs."""

    object_points: list[np.ndarray]
    left_image_points: list[np.ndarray]
    right_image_points: list[np.ndarray]
    image_size: tuple[int, int]
    rejected: list[dict]
    accepted: list[dict] = field(default_factory=list)


def _as_list(array: np.ndarray) -> list:
    return np.asarray(array, dtype=np.float64).tolist()


def _finite_or_none(value: float | int | np.floating) -> float | None:
    number = float(value)
    return number if np.isfinite(number) else None


def _error_summary(values: Iterable[float], *, unit_suffix: str = "") -> dict:
    errors = np.asarray([float(value) for value in values if np.isfinite(value)], dtype=np.float64)
    metric_names = {
        "mean": f"mean{unit_suffix}",
        "median": f"median{unit_suffix}",
        "rms": f"rms{unit_suffix}",
        "max": f"max{unit_suffix}",
    }
    if errors.size == 0:
        return {
            "point_count": 0,
            metric_names["mean"]: None,
            metric_names["median"]: None,
            metric_names["rms"]: None,
            metric_names["max"]: None,
        }
    return {
        "point_count": int(errors.size),
        metric_names["mean"]: float(np.mean(errors)),
        metric_names["median"]: float(np.median(errors)),
        metric_names["rms"]: float(np.sqrt(np.mean(np.square(errors)))),
        metric_names["max"]: float(np.max(errors)),
    }


def _coerce_observation_points(
    object_points: Iterable[np.ndarray],
    left_image_points: Iterable[np.ndarray],
    right_image_points: Iterable[np.ndarray],
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    obj = [np.asarray(points, dtype=np.float32).reshape(-1, 3) for points in object_points]
    left = [np.asarray(points, dtype=np.float32).reshape(-1, 2) for points in left_image_points]
    right = [np.asarray(points, dtype=np.float32).reshape(-1, 2) for points in right_image_points]
    if not obj or len(obj) != len(left) or len(obj) != len(right):
        raise ValueError("Stereo calibration requires matched object, left, and right observations")
    for idx, (obj_points, left_points, right_points) in enumerate(zip(obj, left, right), start=1):
        if len(obj_points) < 4:
            raise ValueError(f"Observation {idx} has too few object points")
        if len(obj_points) != len(left_points) or len(obj_points) != len(right_points):
            raise ValueError(f"Observation {idx} point counts do not match")
    return obj, left, right


def checkerboard_object_points(board: CheckerboardSpec) -> np.ndarray:
    """Return checkerboard object points in the board coordinate frame."""

    if board.columns <= 1 or board.rows <= 1:
        raise ValueError("Checkerboard columns and rows must be inner-corner counts greater than one")
    if board.square_size <= 0:
        raise ValueError("Checkerboard square_size must be positive")
    grid = np.zeros((board.rows * board.columns, 3), np.float32)
    grid[:, :2] = np.mgrid[0 : board.columns, 0 : board.rows].T.reshape(-1, 2)
    grid *= float(board.square_size)
    return grid


def find_checkerboard_corners(image_bgr: np.ndarray, board: CheckerboardSpec) -> np.ndarray | None:
    """Find and refine checkerboard inner corners in one image."""

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr
    pattern_size = (int(board.columns), int(board.rows))
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
    ok, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
    if not ok or corners is None:
        return None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
    refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return refined.reshape(-1, 2).astype(np.float32)


def _read_image(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def collect_checkerboard_observations(
    image_pairs: PointPairs,
    board: CheckerboardSpec,
    *,
    min_pairs: int = 8,
) -> StereoObservations:
    """Detect checkerboard corners in saved left/right image pairs."""

    obj_template = checkerboard_object_points(board)
    object_points: list[np.ndarray] = []
    left_points: list[np.ndarray] = []
    right_points: list[np.ndarray] = []
    rejected: list[dict] = []
    accepted: list[dict] = []
    image_size: tuple[int, int] | None = None

    for index, (left_path, right_path) in enumerate(image_pairs, start=1):
        left_image = _read_image(left_path)
        right_image = _read_image(right_path)
        size = (int(left_image.shape[1]), int(left_image.shape[0]))
        if right_image.shape[:2] != left_image.shape[:2]:
            rejected.append({"index": index, "reason": "left/right image sizes differ"})
            continue
        if image_size is None:
            image_size = size
        elif image_size != size:
            rejected.append({"index": index, "reason": "image size differs from first accepted pair"})
            continue

        left_corners = find_checkerboard_corners(left_image, board)
        right_corners = find_checkerboard_corners(right_image, board)
        if left_corners is None or right_corners is None:
            rejected.append({
                "index": index,
                "reason": "checkerboard not found in both images",
                "left_path": str(left_path),
                "right_path": str(right_path),
            })
            continue

        object_points.append(obj_template.copy())
        left_points.append(left_corners)
        right_points.append(right_corners)
        accepted.append(
            {
                "index": int(index),
                "detector": "checkerboard",
                "point_count": int(len(obj_template)),
                "left_path": str(left_path),
                "right_path": str(right_path),
            }
        )

    if image_size is None:
        raise ValueError("No readable stereo image pairs were accepted")
    if len(object_points) < int(min_pairs):
        raise ValueError(f"Only {len(object_points)} valid stereo pairs found; need at least {int(min_pairs)}")
    return StereoObservations(object_points, left_points, right_points, image_size, rejected, accepted)


def _aruco_dictionary(name: str):
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("OpenCV aruco module is not available; install an OpenCV build with aruco support")
    dictionary_id = getattr(cv2.aruco, str(name), None)
    if dictionary_id is None:
        raise ValueError(f"Unknown aruco dictionary: {name}")
    return cv2.aruco.getPredefinedDictionary(dictionary_id)


def _charuco_board(board: CharucoBoardSpec):
    dictionary = _aruco_dictionary(board.dictionary)
    return cv2.aruco.CharucoBoard(
        (int(board.squares_x), int(board.squares_y)),
        float(board.square_size),
        float(board.marker_size),
        dictionary,
    )


def detect_board_in_image(image_bgr: np.ndarray, board: CheckerboardSpec | CharucoBoardSpec) -> dict:
    """Detect calibration-board image points for preview and diagnostics."""

    if isinstance(board, CharucoBoardSpec):
        cv_board = _charuco_board(board)
        detector = cv2.aruco.CharucoDetector(cv_board)
        corners, ids, marker_corners, marker_ids = detector.detectBoard(image_bgr)
        ids_flat = [] if ids is None else [int(value) for value in ids.reshape(-1)]
        return {
            "kind": "charuco",
            "detected": bool(ids_flat),
            "corner_count": int(len(ids_flat)),
            "marker_count": 0 if marker_ids is None else int(len(marker_ids)),
            "ids": ids_flat,
            "corners": corners,
            "marker_corners": marker_corners,
            "marker_ids": marker_ids,
        }

    corners = find_checkerboard_corners(image_bgr, board)
    return {
        "kind": "checkerboard",
        "detected": corners is not None,
        "corner_count": 0 if corners is None else int(len(corners)),
        "ids": [],
        "corners": None if corners is None else corners.reshape(-1, 1, 2),
        "pattern_size": (int(board.columns), int(board.rows)),
    }


def detect_stereo_board(
    left_image_bgr: np.ndarray,
    right_image_bgr: np.ndarray,
    board: CheckerboardSpec | CharucoBoardSpec,
    *,
    min_corners: int = 8,
) -> dict:
    """Detect and summarize one left/right board observation for preview."""

    left = detect_board_in_image(left_image_bgr, board)
    right = detect_board_in_image(right_image_bgr, board)
    if isinstance(board, CharucoBoardSpec):
        matched_ids = sorted(set(left["ids"]) & set(right["ids"]))
        matched_count = int(len(matched_ids))
        accepted = matched_count >= int(min_corners)
        if accepted:
            reason = "ok"
        elif not left["detected"] or not right["detected"]:
            reason = "board not found in both images"
        else:
            reason = f"only {matched_count} matched ChArUco corners"
        return {
            "kind": "charuco",
            "left": left,
            "right": right,
            "matched_ids": matched_ids,
            "matched_count": matched_count,
            "accepted": bool(accepted),
            "reason": reason,
        }

    matched_count = (
        min(int(left["corner_count"]), int(right["corner_count"]))
        if left["detected"] and right["detected"]
        else 0
    )
    accepted = bool(left["detected"] and right["detected"])
    return {
        "kind": "checkerboard",
        "left": left,
        "right": right,
        "matched_ids": [],
        "matched_count": int(matched_count),
        "accepted": accepted,
        "reason": "ok" if accepted else "checkerboard not found in both images",
    }


def annotate_board_detection(
    image_bgr: np.ndarray,
    detection: dict,
    *,
    matched_ids: Iterable[int] | None = None,
) -> np.ndarray:
    """Draw detected board markers/corners for visual calibration review."""

    annotated = np.ascontiguousarray(image_bgr.copy())
    matched = {int(value) for value in (matched_ids or [])}
    if detection.get("kind") == "charuco":
        marker_corners = detection.get("marker_corners")
        marker_ids = detection.get("marker_ids")
        if marker_corners is not None and marker_ids is not None:
            cv2.aruco.drawDetectedMarkers(annotated, marker_corners, marker_ids, (0, 180, 255))

        corners = detection.get("corners")
        ids = detection.get("ids") or []
        if corners is not None and ids:
            ids_array = np.asarray(ids, dtype=np.int32).reshape(-1, 1)
            cv2.aruco.drawDetectedCornersCharuco(annotated, corners, ids_array, (40, 255, 40))
            for cid, corner in zip(ids, np.asarray(corners, dtype=np.float32).reshape(-1, 2)):
                color = (60, 255, 60) if int(cid) in matched else (0, 180, 255)
                center = (int(round(float(corner[0]))), int(round(float(corner[1]))))
                cv2.circle(annotated, center, 6, color, 2)
                if int(cid) in matched:
                    cv2.putText(
                        annotated,
                        str(int(cid)),
                        (center[0] + 6, center[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.42,
                        color,
                        1,
                        cv2.LINE_AA,
                    )
        return annotated

    corners = detection.get("corners")
    if corners is not None:
        cv2.drawChessboardCorners(
            annotated,
            tuple(detection.get("pattern_size") or (0, 0)),
            corners,
            bool(detection.get("detected")),
        )
    return annotated


def collect_charuco_observations(
    image_pairs: PointPairs,
    board: CharucoBoardSpec,
    *,
    min_corners: int = 8,
    min_pairs: int = 8,
) -> StereoObservations:
    """Detect matched ChArUco corners in saved left/right image pairs."""

    cv_board = _charuco_board(board)
    detector = cv2.aruco.CharucoDetector(cv_board)
    board_points = cv_board.getChessboardCorners().astype(np.float32)
    object_points: list[np.ndarray] = []
    left_points: list[np.ndarray] = []
    right_points: list[np.ndarray] = []
    rejected: list[dict] = []
    accepted: list[dict] = []
    image_size: tuple[int, int] | None = None

    for index, (left_path, right_path) in enumerate(image_pairs, start=1):
        left_image = _read_image(left_path)
        right_image = _read_image(right_path)
        size = (int(left_image.shape[1]), int(left_image.shape[0]))
        if right_image.shape[:2] != left_image.shape[:2]:
            rejected.append({"index": index, "reason": "left/right image sizes differ"})
            continue
        if image_size is None:
            image_size = size
        elif image_size != size:
            rejected.append({"index": index, "reason": "image size differs from first accepted pair"})
            continue

        left_corners, left_ids, _left_marker_corners, _left_marker_ids = detector.detectBoard(left_image)
        right_corners, right_ids, _right_marker_corners, _right_marker_ids = detector.detectBoard(right_image)
        if left_corners is None or right_corners is None or left_ids is None or right_ids is None:
            rejected.append({"index": index, "reason": "charuco board not found in both images"})
            continue

        left_by_id = {int(cid): corner.reshape(2) for cid, corner in zip(left_ids.reshape(-1), left_corners)}
        right_by_id = {int(cid): corner.reshape(2) for cid, corner in zip(right_ids.reshape(-1), right_corners)}
        common_ids = sorted(set(left_by_id) & set(right_by_id))
        if len(common_ids) < int(min_corners):
            rejected.append({"index": index, "reason": f"only {len(common_ids)} matched charuco corners"})
            continue

        object_points.append(np.asarray([board_points[cid] for cid in common_ids], dtype=np.float32))
        left_points.append(np.asarray([left_by_id[cid] for cid in common_ids], dtype=np.float32))
        right_points.append(np.asarray([right_by_id[cid] for cid in common_ids], dtype=np.float32))
        accepted.append(
            {
                "index": int(index),
                "detector": "charuco",
                "point_count": int(len(common_ids)),
                "left_path": str(left_path),
                "right_path": str(right_path),
            }
        )

    if image_size is None:
        raise ValueError("No readable stereo image pairs were accepted")
    if len(object_points) < int(min_pairs):
        raise ValueError(f"Only {len(object_points)} valid stereo pairs found; need at least {int(min_pairs)}")
    return StereoObservations(object_points, left_points, right_points, image_size, rejected, accepted)


def _reprojection_quality(
    object_points: Sequence[np.ndarray],
    image_points: Sequence[np.ndarray],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> dict:
    """Solve each board pose and report per-camera reprojection residuals."""

    all_errors: list[float] = []
    views: list[dict] = []
    for index, (obj, points) in enumerate(zip(object_points, image_points), start=1):
        obj_arr = np.asarray(obj, dtype=np.float32).reshape(-1, 3)
        points_arr = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        ok, rvec, tvec = cv2.solvePnP(obj_arr, points_arr, camera_matrix, dist_coeffs)
        if not ok:
            views.append({"index": index, "point_count": int(len(points_arr)), "rms_px": None, "max_px": None})
            continue

        projected, _jacobian = cv2.projectPoints(obj_arr, rvec, tvec, camera_matrix, dist_coeffs)
        residuals = np.linalg.norm(projected.reshape(-1, 2) - points_arr, axis=1)
        view_summary = _error_summary(residuals, unit_suffix="_px")
        view_summary["index"] = index
        views.append(view_summary)
        all_errors.extend(float(value) for value in residuals)

    summary = _error_summary(all_errors, unit_suffix="_px")
    summary["view_count"] = int(len(views))
    summary["views"] = views
    return summary


def _epipolar_quality(
    left_points: Sequence[np.ndarray],
    right_points: Sequence[np.ndarray],
    fundamental: np.ndarray,
) -> dict:
    """Report symmetric point-to-epipolar-line error for accepted matches."""

    fundamental = np.asarray(fundamental, dtype=np.float64).reshape(3, 3)
    all_errors: list[float] = []
    views: list[dict] = []
    for index, (left, right) in enumerate(zip(left_points, right_points), start=1):
        left_arr = np.asarray(left, dtype=np.float64).reshape(-1, 2)
        right_arr = np.asarray(right, dtype=np.float64).reshape(-1, 2)
        left_h = np.column_stack((left_arr, np.ones(len(left_arr), dtype=np.float64)))
        right_h = np.column_stack((right_arr, np.ones(len(right_arr), dtype=np.float64)))

        right_lines = (fundamental @ left_h.T).T
        left_lines = (fundamental.T @ right_h.T).T
        right_den = np.linalg.norm(right_lines[:, :2], axis=1)
        left_den = np.linalg.norm(left_lines[:, :2], axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            right_error = np.abs(np.sum(right_h * right_lines, axis=1)) / right_den
            left_error = np.abs(np.sum(left_h * left_lines, axis=1)) / left_den
        symmetric = 0.5 * (left_error + right_error)

        view_summary = _error_summary(symmetric, unit_suffix="_px")
        view_summary["index"] = index
        views.append(view_summary)
        all_errors.extend(float(value) for value in symmetric if np.isfinite(value))

    summary = _error_summary(all_errors, unit_suffix="_px")
    summary["view_count"] = int(len(views))
    summary["views"] = views
    return summary


def _coverage_quality(image_points: Sequence[np.ndarray], image_size: tuple[int, int]) -> dict:
    """Measure whether accepted points cover the image instead of only the center."""

    width, height = int(image_size[0]), int(image_size[1])
    if width <= 0 or height <= 0:
        raise ValueError("image_size must contain positive width and height")
    if not image_points:
        return {
            "point_count": 0,
            "x_min_px": None,
            "x_max_px": None,
            "y_min_px": None,
            "y_max_px": None,
            "width_fraction": None,
            "height_fraction": None,
            "area_fraction": None,
            "grid_4x4_fraction": None,
        }

    points = np.concatenate([np.asarray(item, dtype=np.float64).reshape(-1, 2) for item in image_points], axis=0)
    x_min = float(np.min(points[:, 0]))
    x_max = float(np.max(points[:, 0]))
    y_min = float(np.min(points[:, 1]))
    y_max = float(np.max(points[:, 1]))
    clipped_x_min = max(0.0, min(float(width), x_min))
    clipped_x_max = max(0.0, min(float(width), x_max))
    clipped_y_min = max(0.0, min(float(height), y_min))
    clipped_y_max = max(0.0, min(float(height), y_max))
    width_fraction = max(0.0, clipped_x_max - clipped_x_min) / float(width)
    height_fraction = max(0.0, clipped_y_max - clipped_y_min) / float(height)

    grid = 4
    grid_x = np.clip(np.floor(points[:, 0] / float(width) * grid).astype(int), 0, grid - 1)
    grid_y = np.clip(np.floor(points[:, 1] / float(height) * grid).astype(int), 0, grid - 1)
    occupied = set(zip(grid_x.tolist(), grid_y.tolist()))

    return {
        "point_count": int(len(points)),
        "x_min_px": _finite_or_none(x_min),
        "x_max_px": _finite_or_none(x_max),
        "y_min_px": _finite_or_none(y_min),
        "y_max_px": _finite_or_none(y_max),
        "width_fraction": _finite_or_none(width_fraction),
        "height_fraction": _finite_or_none(height_fraction),
        "area_fraction": _finite_or_none(width_fraction * height_fraction),
        "grid_4x4_fraction": float(len(occupied) / float(grid * grid)),
    }


def _build_quality_warnings(artifact: dict, quality: dict) -> list[str]:
    warnings: list[str] = []
    observation_count = int(artifact.get("observation_count") or 0)
    stereo_rms = (artifact.get("rms") or {}).get("stereo")
    epipolar_rms = (quality.get("epipolar") or {}).get("rms_px")
    left_coverage = (quality.get("left_coverage") or {}).get("area_fraction")
    right_coverage = (quality.get("right_coverage") or {}).get("area_fraction")
    left_grid = (quality.get("left_coverage") or {}).get("grid_4x4_fraction")
    right_grid = (quality.get("right_coverage") or {}).get("grid_4x4_fraction")

    if observation_count < 20:
        warnings.append("Accepted fewer than 20 stereo pairs; capture more board poses before trusting measurement.")
    elif observation_count < 30:
        warnings.append("Accepted fewer than 30 stereo pairs; usable, but more varied poses will usually improve stability.")
    if stereo_rms is not None and float(stereo_rms) > 1.0:
        warnings.append("Stereo RMS is above 1 px; inspect blur, lighting, frame sync, and board coverage.")
    if epipolar_rms is not None and float(epipolar_rms) > 1.0:
        warnings.append("Epipolar RMS is above 1 px; rectified left/right features may not align well.")
    if left_coverage is not None and float(left_coverage) < 0.35:
        warnings.append("Left image corner coverage is narrow; include board poses near the edges and corners.")
    if right_coverage is not None and float(right_coverage) < 0.35:
        warnings.append("Right image corner coverage is narrow; include board poses near the edges and corners.")
    if left_grid is not None and float(left_grid) < 0.50:
        warnings.append("Left image points occupy less than half of the 4x4 coverage grid.")
    if right_grid is not None and float(right_grid) < 0.50:
        warnings.append("Right image points occupy less than half of the 4x4 coverage grid.")
    return warnings


def calibrate_stereo_from_observations(
    observations: StereoObservations,
    *,
    rig_id: str,
    pair_name: str,
    board_spec: CheckerboardSpec | CharucoBoardSpec | None = None,
    flags: int | None = None,
    camera_matrix_left: np.ndarray | None = None,
    dist_coeffs_left: np.ndarray | None = None,
    camera_matrix_right: np.ndarray | None = None,
    dist_coeffs_right: np.ndarray | None = None,
) -> dict:
    """Run OpenCV stereo calibration and return a JSON-serializable artifact."""

    obj, left, right = _coerce_observation_points(
        observations.object_points,
        observations.left_image_points,
        observations.right_image_points,
    )
    image_size = (int(observations.image_size[0]), int(observations.image_size[1]))
    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-6)

    if camera_matrix_left is None or dist_coeffs_left is None:
        left_rms, camera_matrix_left, dist_coeffs_left, _rvecs, _tvecs = cv2.calibrateCamera(
            obj,
            left,
            image_size,
            None,
            None,
        )
    else:
        left_rms = None
        camera_matrix_left = np.asarray(camera_matrix_left, dtype=np.float64)
        dist_coeffs_left = np.asarray(dist_coeffs_left, dtype=np.float64)

    if camera_matrix_right is None or dist_coeffs_right is None:
        right_rms, camera_matrix_right, dist_coeffs_right, _rvecs, _tvecs = cv2.calibrateCamera(
            obj,
            right,
            image_size,
            None,
            None,
        )
    else:
        right_rms = None
        camera_matrix_right = np.asarray(camera_matrix_right, dtype=np.float64)
        dist_coeffs_right = np.asarray(dist_coeffs_right, dtype=np.float64)

    stereo_flags = cv2.CALIB_FIX_INTRINSIC if flags is None else int(flags)
    (
        stereo_rms,
        camera_matrix_left,
        dist_coeffs_left,
        camera_matrix_right,
        dist_coeffs_right,
        rotation,
        translation,
        essential,
        fundamental,
    ) = cv2.stereoCalibrate(
        obj,
        left,
        right,
        camera_matrix_left,
        dist_coeffs_left,
        camera_matrix_right,
        dist_coeffs_right,
        image_size,
        criteria=criteria,
        flags=stereo_flags,
    )
    r1, r2, p1, p2, q, roi1, roi2 = cv2.stereoRectify(
        camera_matrix_left,
        dist_coeffs_left,
        camera_matrix_right,
        dist_coeffs_right,
        image_size,
        rotation,
        translation,
        flags=cv2.CALIB_ZERO_DISPARITY,
    )

    quality = {
        "accepted_observations": list(observations.accepted),
        "left_reprojection": _reprojection_quality(obj, left, camera_matrix_left, dist_coeffs_left),
        "right_reprojection": _reprojection_quality(obj, right, camera_matrix_right, dist_coeffs_right),
        "epipolar": _epipolar_quality(left, right, fundamental),
        "left_coverage": _coverage_quality(left, image_size),
        "right_coverage": _coverage_quality(right, image_size),
        "warnings": [],
    }
    artifact = {
        "schema": "tritonanalysis.stereo_calibration",
        "schema_version": 1,
        "created_wall_ts": time.time(),
        "rig_id": str(rig_id),
        "pair_name": str(pair_name),
        "image_size": [image_size[0], image_size[1]],
        "observation_count": len(obj),
        "rejected_observations": list(observations.rejected),
        "board": asdict(board_spec) if board_spec is not None else None,
        "calibration_flags": int(stereo_flags),
        "rms": {
            "left": None if left_rms is None else float(left_rms),
            "right": None if right_rms is None else float(right_rms),
            "stereo": float(stereo_rms),
        },
        "left": {
            "camera_matrix": _as_list(camera_matrix_left),
            "dist_coeffs": _as_list(dist_coeffs_left),
        },
        "right": {
            "camera_matrix": _as_list(camera_matrix_right),
            "dist_coeffs": _as_list(dist_coeffs_right),
        },
        "stereo": {
            "rotation": _as_list(rotation),
            "translation": _as_list(translation.reshape(-1)),
            "baseline": float(np.linalg.norm(translation)),
            "essential": _as_list(essential),
            "fundamental": _as_list(fundamental),
        },
        "rectification": {
            "r1": _as_list(r1),
            "r2": _as_list(r2),
            "p1": _as_list(p1),
            "p2": _as_list(p2),
            "q": _as_list(q),
            "roi1": [int(v) for v in roi1],
            "roi2": [int(v) for v in roi2],
        },
        "quality": quality,
    }
    quality["warnings"] = _build_quality_warnings(artifact, quality)
    return artifact


def manifest_image_pairs(manifest_path: str | Path) -> list[tuple[Path, Path]]:
    """Return absolute left/right image paths from a TritonPilot stereo manifest."""

    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    root = path.parent
    pairs: list[tuple[Path, Path]] = []
    for frame in manifest.get("frames", []):
        left = frame.get("left_path")
        right = frame.get("right_path")
        if left and right:
            pairs.append((root / left, root / right))
    return pairs


def discover_stereo_manifests(sources: Iterable[str | Path]) -> list[Path]:
    """Resolve manifest files from explicit manifest paths or session folders."""

    manifests: list[Path] = []
    seen: set[Path] = set()
    for source in sources:
        path = Path(source)
        candidates: list[Path]
        if path.is_dir():
            direct = path / "manifest.json"
            candidates = [direct] if direct.exists() else sorted(path.rglob("manifest.json"))
        else:
            candidates = [path]
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved not in seen:
                manifests.append(candidate)
                seen.add(resolved)
    return manifests


def _same_stereo_pair(left: dict, right: dict) -> bool:
    for key in ("name", "left", "right", "rig_id"):
        if str(left.get(key, "")) != str(right.get(key, "")):
            return False
    return True


def load_manifest_collection(sources: Iterable[str | Path]) -> tuple[dict, list[tuple[Path, Path]]]:
    """Load one or more TritonPilot stereo manifests as one calibration dataset."""

    manifest_paths = discover_stereo_manifests(sources)
    if not manifest_paths:
        raise ValueError("No stereo manifest files found")

    combined_frames: list[dict] = []
    image_pairs: list[tuple[Path, Path]] = []
    source_records: list[dict] = []
    pair_ref: dict | None = None
    first_manifest: dict | None = None

    for manifest_path in manifest_paths:
        path = Path(manifest_path)
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if first_manifest is None:
            first_manifest = manifest
        pair = dict(manifest.get("pair") or {})
        if pair_ref is None:
            pair_ref = pair
        elif not _same_stereo_pair(pair_ref, pair):
            raise ValueError(f"Manifest uses a different stereo pair and cannot be mixed: {path}")

        root = path.parent
        frames = list(manifest.get("frames") or [])
        source_records.append(
            {
                "path": str(path),
                "session_name": str(manifest.get("session_name") or path.parent.name),
                "frames": len(frames),
            }
        )
        for frame in frames:
            left = frame.get("left_path")
            right = frame.get("right_path")
            if not left or not right:
                continue
            left_path = root / left
            right_path = root / right
            image_pairs.append((left_path, right_path))
            combined = dict(frame)
            combined["index"] = len(combined_frames) + 1
            combined["source_manifest"] = str(path)
            combined["source_session"] = str(manifest.get("session_name") or path.parent.name)
            combined["source_index"] = frame.get("index", len(combined_frames) + 1)
            combined_frames.append(combined)

    if first_manifest is None:
        first_manifest = {}
    collection = {
        "schema": "tritonanalysis.stereo_manifest_collection",
        "schema_version": 1,
        "sources": source_records,
        "source_count": len(source_records),
        "session_name": (
            str(first_manifest.get("session_name") or Path(manifest_paths[0]).parent.name)
            if len(source_records) == 1
            else f"{len(source_records)} stereo sessions"
        ),
        "pair": pair_ref or {},
        "streams": first_manifest.get("streams") or {},
        "frames": combined_frames,
    }
    return collection, image_pairs


def write_calibration_artifact(artifact: dict, path: str | Path) -> Path:
    """Write a stereo calibration artifact to JSON."""

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def read_calibration_artifact(path: str | Path) -> dict:
    """Read a stereo calibration artifact JSON file."""

    return json.loads(Path(path).read_text(encoding="utf-8"))
