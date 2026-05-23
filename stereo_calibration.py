"""Stereo calibration and rectification helpers for saved TritonPilot captures."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np


PointPairs = Sequence[tuple[str | Path, str | Path]]


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
    dictionary: str = "DICT_4X4_50"
    units: str = "cm"


@dataclass(frozen=True)
class StereoObservations:
    """Matched calibration points from multiple stereo image pairs."""

    object_points: list[np.ndarray]
    left_image_points: list[np.ndarray]
    right_image_points: list[np.ndarray]
    image_size: tuple[int, int]
    rejected: list[dict]


def _as_list(array: np.ndarray) -> list:
    return np.asarray(array, dtype=np.float64).tolist()


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

    if image_size is None:
        raise ValueError("No readable stereo image pairs were accepted")
    if len(object_points) < int(min_pairs):
        raise ValueError(f"Only {len(object_points)} valid stereo pairs found; need at least {int(min_pairs)}")
    return StereoObservations(object_points, left_points, right_points, image_size, rejected)


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

    if image_size is None:
        raise ValueError("No readable stereo image pairs were accepted")
    if len(object_points) < int(min_pairs):
        raise ValueError(f"Only {len(object_points)} valid stereo pairs found; need at least {int(min_pairs)}")
    return StereoObservations(object_points, left_points, right_points, image_size, rejected)


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
    }
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


def write_calibration_artifact(artifact: dict, path: str | Path) -> Path:
    """Write a stereo calibration artifact to JSON."""

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def read_calibration_artifact(path: str | Path) -> dict:
    """Read a stereo calibration artifact JSON file."""

    return json.loads(Path(path).read_text(encoding="utf-8"))
