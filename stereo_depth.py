"""Stereo rectification, disparity, and 3D measurement helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import cv2
import numpy as np

from stereo_calibration import read_calibration_artifact


@dataclass(frozen=True)
class RectificationMaps:
    """OpenCV remap tables and reprojection metadata for one stereo artifact."""

    left_x: np.ndarray
    left_y: np.ndarray
    right_x: np.ndarray
    right_y: np.ndarray
    q: np.ndarray
    image_size: tuple[int, int]
    units: str


@dataclass(frozen=True)
class DepthSample:
    """One sampled 3D point from a disparity-derived point cloud."""

    pixel: tuple[int, int]
    point: np.ndarray
    disparity: float
    sample_count: int


@dataclass(frozen=True)
class CorrespondenceSample:
    """One 3D point from a manually matched rectified left/right pixel pair."""

    left_pixel: tuple[int, int]
    right_pixel: tuple[int, int]
    point: np.ndarray
    disparity: float
    vertical_error_px: float


def _artifact_array(artifact: Mapping, *keys: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
    value = artifact
    for key in keys:
        value = value[key]
    array = np.asarray(value, dtype=np.float64)
    if shape is not None:
        array = array.reshape(shape)
    return array


def load_depth_calibration(path: str | Path) -> dict:
    """Read and minimally validate a stereo calibration artifact."""

    artifact = read_calibration_artifact(path)
    if artifact.get("schema") != "tritonanalysis.stereo_calibration":
        raise ValueError("Calibration file is not a TritonAnalysis stereo calibration artifact")
    if "rectification" not in artifact or "left" not in artifact or "right" not in artifact:
        raise ValueError("Calibration artifact is missing rectification or camera matrices")
    return artifact


def rectification_maps_from_artifact(artifact: Mapping) -> RectificationMaps:
    """Build OpenCV remap tables from a stereo calibration artifact."""

    image_size_raw = artifact.get("image_size") or []
    if len(image_size_raw) != 2:
        raise ValueError("Calibration artifact is missing image_size")
    image_size = (int(image_size_raw[0]), int(image_size_raw[1]))
    if image_size[0] <= 0 or image_size[1] <= 0:
        raise ValueError("Calibration artifact image_size must be positive")

    left_matrix = _artifact_array(artifact, "left", "camera_matrix", shape=(3, 3))
    right_matrix = _artifact_array(artifact, "right", "camera_matrix", shape=(3, 3))
    left_dist = _artifact_array(artifact, "left", "dist_coeffs").reshape(-1, 1)
    right_dist = _artifact_array(artifact, "right", "dist_coeffs").reshape(-1, 1)
    rectification = artifact["rectification"]
    r1 = np.asarray(rectification["r1"], dtype=np.float64).reshape(3, 3)
    r2 = np.asarray(rectification["r2"], dtype=np.float64).reshape(3, 3)
    p1 = np.asarray(rectification["p1"], dtype=np.float64)
    p2 = np.asarray(rectification["p2"], dtype=np.float64)
    q = np.asarray(rectification["q"], dtype=np.float64).reshape(4, 4)

    left_x, left_y = cv2.initUndistortRectifyMap(
        left_matrix,
        left_dist,
        r1,
        p1,
        image_size,
        cv2.CV_32FC1,
    )
    right_x, right_y = cv2.initUndistortRectifyMap(
        right_matrix,
        right_dist,
        r2,
        p2,
        image_size,
        cv2.CV_32FC1,
    )
    board = artifact.get("board") or {}
    return RectificationMaps(
        left_x=left_x,
        left_y=left_y,
        right_x=right_x,
        right_y=right_y,
        q=q,
        image_size=image_size,
        units=str(board.get("units") or ""),
    )


def _charuco_board_from_artifact(artifact: Mapping):
    board = artifact.get("board") or {}
    required = ("squares_x", "squares_y", "square_size", "marker_size", "dictionary")
    if not all(key in board for key in required):
        return None
    if not hasattr(cv2, "aruco"):
        return None
    dictionary_id = getattr(cv2.aruco, str(board["dictionary"]), None)
    if dictionary_id is None:
        return None
    return cv2.aruco.CharucoBoard(
        (int(board["squares_x"]), int(board["squares_y"])),
        float(board["square_size"]),
        float(board["marker_size"]),
        cv2.aruco.getPredefinedDictionary(dictionary_id),
    )


def analyze_charuco_stereo_geometry(left_bgr: np.ndarray, right_bgr: np.ndarray, artifact: Mapping) -> dict:
    """Validate rectified geometry using detected ChArUco corners when available."""

    cv_board = _charuco_board_from_artifact(artifact)
    if cv_board is None:
        return {"available": False, "reason": "calibration artifact does not describe a ChArUco board"}

    detector = cv2.aruco.CharucoDetector(cv_board)
    left_corners, left_ids, _left_marker_corners, _left_marker_ids = detector.detectBoard(left_bgr)
    right_corners, right_ids, _right_marker_corners, _right_marker_ids = detector.detectBoard(right_bgr)
    if left_corners is None or right_corners is None or left_ids is None or right_ids is None:
        return {"available": False, "reason": "ChArUco corners not detected in both images"}

    left_by_id = {int(cid): corner.reshape(2) for cid, corner in zip(left_ids.reshape(-1), left_corners)}
    right_by_id = {int(cid): corner.reshape(2) for cid, corner in zip(right_ids.reshape(-1), right_corners)}
    common_ids = sorted(set(left_by_id) & set(right_by_id))
    if len(common_ids) < 4:
        return {"available": False, "reason": "too few matched ChArUco corners"}

    left_points = np.asarray([left_by_id[cid] for cid in common_ids], dtype=np.float32).reshape(-1, 1, 2)
    right_points = np.asarray([right_by_id[cid] for cid in common_ids], dtype=np.float32).reshape(-1, 1, 2)
    left_matrix = _artifact_array(artifact, "left", "camera_matrix", shape=(3, 3))
    right_matrix = _artifact_array(artifact, "right", "camera_matrix", shape=(3, 3))
    left_dist = _artifact_array(artifact, "left", "dist_coeffs").reshape(-1, 1)
    right_dist = _artifact_array(artifact, "right", "dist_coeffs").reshape(-1, 1)
    rectification = artifact["rectification"]
    r1 = np.asarray(rectification["r1"], dtype=np.float64).reshape(3, 3)
    r2 = np.asarray(rectification["r2"], dtype=np.float64).reshape(3, 3)
    p1 = np.asarray(rectification["p1"], dtype=np.float64).reshape(3, 4)
    p2 = np.asarray(rectification["p2"], dtype=np.float64).reshape(3, 4)
    q = np.asarray(rectification["q"], dtype=np.float64).reshape(4, 4)

    left_rect = cv2.undistortPoints(left_points, left_matrix, left_dist, R=r1, P=p1).reshape(-1, 2)
    right_rect = cv2.undistortPoints(right_points, right_matrix, right_dist, R=r2, P=p2).reshape(-1, 2)
    disparity = left_rect[:, 0] - right_rect[:, 0]
    vertical_error = left_rect[:, 1] - right_rect[:, 1]

    q_points = []
    for point, disp in zip(left_rect, disparity):
        homogeneous = q @ np.array([point[0], point[1], disp, 1.0], dtype=np.float64)
        q_points.append(homogeneous[:3] / homogeneous[3])
    q_points = np.asarray(q_points, dtype=np.float64)

    board_points = cv_board.getChessboardCorners().astype(np.float64)
    id_to_index = {cid: index for index, cid in enumerate(common_ids)}
    square_size = float((artifact.get("board") or {}).get("square_size") or 0.0)
    edge_lengths = []
    if square_size > 0:
        for first in common_ids:
            for second in common_ids:
                if second <= first:
                    continue
                object_distance = float(np.linalg.norm(board_points[first] - board_points[second]))
                if abs(object_distance - square_size) <= max(1.0e-6, square_size * 1.0e-6):
                    edge_lengths.append(
                        float(np.linalg.norm(q_points[id_to_index[first]] - q_points[id_to_index[second]]))
                    )

    return {
        "available": True,
        "matched_count": int(len(common_ids)),
        "disparity_min": float(np.min(disparity)),
        "disparity_median": float(np.median(disparity)),
        "disparity_max": float(np.max(disparity)),
        "vertical_rms_px": float(np.sqrt(np.mean(np.square(vertical_error)))),
        "vertical_median_abs_px": float(np.median(np.abs(vertical_error))),
        "edge_count": int(len(edge_lengths)),
        "edge_median": None if not edge_lengths else float(np.median(edge_lengths)),
        "edge_p10": None if not edge_lengths else float(np.percentile(edge_lengths, 10)),
        "edge_p90": None if not edge_lengths else float(np.percentile(edge_lengths, 90)),
        "square_size": square_size,
        "units": str((artifact.get("board") or {}).get("units") or ""),
    }


def rectify_stereo_images(
    left_bgr: np.ndarray,
    right_bgr: np.ndarray,
    maps: RectificationMaps,
) -> tuple[np.ndarray, np.ndarray]:
    """Undistort and rectify a left/right image pair."""

    expected_size = (int(maps.image_size[1]), int(maps.image_size[0]))
    if tuple(left_bgr.shape[:2]) != expected_size:
        raise ValueError(
            f"Left image is {left_bgr.shape[1]}x{left_bgr.shape[0]}, "
            f"but calibration expects {maps.image_size[0]}x{maps.image_size[1]}"
        )
    if tuple(right_bgr.shape[:2]) != expected_size:
        raise ValueError(
            f"Right image is {right_bgr.shape[1]}x{right_bgr.shape[0]}, "
            f"but calibration expects {maps.image_size[0]}x{maps.image_size[1]}"
        )
    left_rect = cv2.remap(left_bgr, maps.left_x, maps.left_y, cv2.INTER_LINEAR)
    right_rect = cv2.remap(right_bgr, maps.right_x, maps.right_y, cv2.INTER_LINEAR)
    return left_rect, right_rect


def normalized_num_disparities(value: int) -> int:
    """Return a positive multiple of 16 accepted by StereoSGBM."""

    return max(16, int(np.ceil(max(1, int(value)) / 16.0)) * 16)


def normalized_block_size(value: int) -> int:
    """Return an odd block size accepted by StereoSGBM."""

    block_size = max(3, int(value))
    if block_size % 2 == 0:
        block_size += 1
    return block_size


def compute_disparity(
    left_rect_bgr: np.ndarray,
    right_rect_bgr: np.ndarray,
    *,
    min_disparity: int = 0,
    num_disparities: int = 320,
    block_size: int = 7,
    uniqueness_ratio: int = 8,
    speckle_window_size: int = 80,
    speckle_range: int = 2,
    disp12_max_diff: int = 1,
    preprocess: str = "clahe",
    left_right_check: bool = True,
    left_right_max_diff: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a float32 disparity map and validity mask from rectified images."""

    if left_rect_bgr.shape[:2] != right_rect_bgr.shape[:2]:
        raise ValueError("Rectified left/right image sizes differ")
    left_gray = _prepare_match_image(left_rect_bgr, preprocess=preprocess)
    right_gray = _prepare_match_image(right_rect_bgr, preprocess=preprocess)
    block_size = normalized_block_size(block_size)
    num_disparities = normalized_num_disparities(num_disparities)
    matcher = _make_sgbm_matcher(
        min_disparity=int(min_disparity),
        num_disparities=int(num_disparities),
        block_size=int(block_size),
        uniqueness_ratio=int(uniqueness_ratio),
        speckle_window_size=int(speckle_window_size),
        speckle_range=int(speckle_range),
        disp12_max_diff=int(disp12_max_diff),
    )
    disparity = matcher.compute(left_gray, right_gray).astype(np.float32) / 16.0
    max_disparity = int(min_disparity) + int(num_disparities)
    valid = (
        np.isfinite(disparity)
        & (disparity >= float(min_disparity))
        & (disparity < float(max_disparity))
        & (np.abs(disparity) >= 1.0)
    )
    if left_right_check:
        right_min_disparity = -(int(min_disparity) + int(num_disparities))
        right_matcher = _make_sgbm_matcher(
            min_disparity=right_min_disparity,
            num_disparities=int(num_disparities),
            block_size=int(block_size),
            uniqueness_ratio=int(uniqueness_ratio),
            speckle_window_size=int(speckle_window_size),
            speckle_range=int(speckle_range),
            disp12_max_diff=int(disp12_max_diff),
        )
        right_disparity = right_matcher.compute(right_gray, left_gray).astype(np.float32) / 16.0
        sampled_right = _sample_right_disparity_for_left(disparity, right_disparity, valid)
        valid &= np.isfinite(sampled_right) & (np.abs(disparity + sampled_right) <= float(left_right_max_diff))
    disparity = disparity.astype(np.float32)
    disparity[~valid] = np.nan
    return disparity, valid


def _prepare_match_image(image: np.ndarray, *, preprocess: str) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    gray = np.ascontiguousarray(gray)
    mode = str(preprocess or "none").strip().lower()
    if mode in {"", "none", "raw"}:
        return gray
    if mode == "clahe":
        if gray.dtype != np.uint8:
            gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)
    raise ValueError(f"Unknown stereo disparity preprocess mode: {preprocess}")


def _make_sgbm_matcher(
    *,
    min_disparity: int,
    num_disparities: int,
    block_size: int,
    uniqueness_ratio: int,
    speckle_window_size: int,
    speckle_range: int,
    disp12_max_diff: int,
):
    return cv2.StereoSGBM_create(
        minDisparity=int(min_disparity),
        numDisparities=int(num_disparities),
        blockSize=int(block_size),
        P1=8 * int(block_size) * int(block_size),
        P2=32 * int(block_size) * int(block_size),
        disp12MaxDiff=int(disp12_max_diff),
        uniquenessRatio=int(uniqueness_ratio),
        speckleWindowSize=int(speckle_window_size),
        speckleRange=int(speckle_range),
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )


def _sample_right_disparity_for_left(
    left_disparity: np.ndarray,
    right_disparity: np.ndarray,
    left_valid: np.ndarray,
) -> np.ndarray:
    height, width = left_disparity.shape[:2]
    x_coords = np.broadcast_to(np.arange(width, dtype=np.float32), (height, width))
    y_coords = np.broadcast_to(np.arange(height, dtype=np.int32).reshape(-1, 1), (height, width))
    right_x = np.rint(x_coords - np.nan_to_num(left_disparity, nan=0.0)).astype(np.int32)
    in_bounds = left_valid.astype(bool) & (right_x >= 0) & (right_x < width)
    sampled = np.full(left_disparity.shape, np.nan, dtype=np.float32)
    sampled[in_bounds] = right_disparity[y_coords[in_bounds], right_x[in_bounds]]
    return sampled


def reproject_disparity(
    disparity: np.ndarray,
    q: np.ndarray,
    valid_mask: np.ndarray | None = None,
    *,
    max_abs_depth: float | None = None,
    min_abs_disparity: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproject disparity into 3D coordinates using OpenCV's Q matrix."""

    disparity_for_cv = np.nan_to_num(disparity.astype(np.float32), nan=0.0)
    points = cv2.reprojectImageTo3D(disparity_for_cv, np.asarray(q, dtype=np.float64).reshape(4, 4))
    valid = np.isfinite(points).all(axis=2)
    if valid_mask is not None:
        valid &= valid_mask.astype(bool)
    if min_abs_disparity > 0:
        valid &= np.abs(disparity.astype(np.float32)) >= float(min_abs_disparity)
    z = points[:, :, 2]
    valid &= np.isfinite(z) & (np.abs(z) > 1.0e-9)
    if max_abs_depth is not None and max_abs_depth > 0:
        valid &= np.abs(z) <= float(max_abs_depth)
    return points.astype(np.float32), valid


def colorize_disparity(disparity: np.ndarray, valid_mask: np.ndarray | None = None) -> np.ndarray:
    """Return a BGR heatmap for a disparity image."""

    if valid_mask is None:
        valid_mask = np.isfinite(disparity)
    values = disparity[np.asarray(valid_mask, dtype=bool) & np.isfinite(disparity)]
    if values.size == 0:
        return np.zeros((*disparity.shape[:2], 3), dtype=np.uint8)
    lo, hi = np.percentile(values, [2.0, 98.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(values)), float(np.nanmax(values) + 1.0)
    normalized = np.clip((np.nan_to_num(disparity, nan=lo) - lo) / max(1.0e-6, hi - lo), 0.0, 1.0)
    heat = cv2.applyColorMap((normalized * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)
    heat[~np.asarray(valid_mask, dtype=bool)] = (24, 24, 24)
    return heat


def colorize_depth(points_3d: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Return a BGR near/far heatmap using absolute Z depth."""

    depth = np.abs(points_3d[:, :, 2].astype(np.float32))
    values = depth[np.asarray(valid_mask, dtype=bool) & np.isfinite(depth)]
    if values.size == 0:
        return np.zeros((*depth.shape[:2], 3), dtype=np.uint8)
    lo, hi = np.percentile(values, [2.0, 98.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(values)), float(np.nanmax(values) + 1.0)
    normalized = 1.0 - np.clip((np.nan_to_num(depth, nan=hi) - lo) / max(1.0e-6, hi - lo), 0.0, 1.0)
    heat = cv2.applyColorMap((normalized * 255.0).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    heat[~np.asarray(valid_mask, dtype=bool)] = (24, 24, 24)
    return heat


def sample_depth_point(
    points_3d: np.ndarray,
    disparity: np.ndarray,
    valid_mask: np.ndarray,
    pixel: tuple[float, float],
    *,
    radius: int = 3,
) -> DepthSample:
    """Sample a robust 3D point around an image pixel."""

    x = int(round(float(pixel[0])))
    y = int(round(float(pixel[1])))
    height, width = valid_mask.shape[:2]
    if x < 0 or y < 0 or x >= width or y >= height:
        raise ValueError("Measurement point is outside the image")

    radius = max(0, int(radius))
    x0, x1 = max(0, x - radius), min(width, x + radius + 1)
    y0, y1 = max(0, y - radius), min(height, y + radius + 1)
    mask = valid_mask[y0:y1, x0:x1].astype(bool)
    local_points = points_3d[y0:y1, x0:x1][mask]
    local_disparity = disparity[y0:y1, x0:x1][mask]
    finite = np.isfinite(local_points).all(axis=1) & np.isfinite(local_disparity)
    if not np.any(finite):
        raise ValueError("No valid depth near that point")

    point = np.median(local_points[finite].astype(np.float64), axis=0).astype(np.float32)
    disp = float(np.median(local_disparity[finite].astype(np.float64)))
    return DepthSample(pixel=(x, y), point=point, disparity=disp, sample_count=int(np.count_nonzero(finite)))


def point_from_rectified_correspondence(
    q: np.ndarray,
    left_pixel: tuple[float, float],
    right_pixel: tuple[float, float],
    *,
    min_abs_disparity: float = 1.0,
    max_vertical_error_px: float | None = 3.0,
) -> CorrespondenceSample:
    """Triangulate one 3D point from manually matched rectified pixels."""

    left_x = float(left_pixel[0])
    left_y = float(left_pixel[1])
    right_x = float(right_pixel[0])
    right_y = float(right_pixel[1])
    disparity = left_x - right_x
    if abs(disparity) < float(min_abs_disparity):
        raise ValueError("Matched points have too little disparity for a stable 3D point")
    vertical_error = left_y - right_y
    if max_vertical_error_px is not None and abs(vertical_error) > float(max_vertical_error_px):
        raise ValueError(
            f"Matched points are {abs(vertical_error):.1f} px apart vertically after rectification"
        )

    y = (left_y + right_y) * 0.5
    homogeneous = np.asarray(q, dtype=np.float64).reshape(4, 4) @ np.array(
        [left_x, y, disparity, 1.0],
        dtype=np.float64,
    )
    if not np.isfinite(homogeneous).all() or abs(float(homogeneous[3])) < 1.0e-12:
        raise ValueError("Matched points could not be triangulated")
    point = (homogeneous[:3] / homogeneous[3]).astype(np.float32)
    if not np.isfinite(point).all():
        raise ValueError("Matched points produced a non-finite 3D point")

    return CorrespondenceSample(
        left_pixel=(int(round(left_x)), int(round(left_y))),
        right_pixel=(int(round(right_x)), int(round(right_y))),
        point=point,
        disparity=float(disparity),
        vertical_error_px=float(vertical_error),
    )


def distance_between_samples(first: DepthSample | CorrespondenceSample, second: DepthSample | CorrespondenceSample) -> float:
    """Return Euclidean distance between two sampled 3D points."""

    return float(np.linalg.norm(np.asarray(first.point, dtype=np.float64) - np.asarray(second.point, dtype=np.float64)))
