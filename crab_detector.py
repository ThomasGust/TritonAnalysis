"""Reference-board detector for the crab-counting task.

This first rebuild assumes the competition board artwork is fixed. It matches a
curated reference photo of that board into each input image, then projects the
four known European green crab regions back into the camera view.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent
ENV_REFERENCE_IMAGE = "TRITON_CRAB_REFERENCE_IMAGE"
DEFAULT_REFERENCE_RELATIVE = Path("recordings") / "20260530-082812_Arm_Camera_snapshot.png"

# Coordinates are in the default reference image:
# C:\Users\Thoma\Documents\GitHub\TritonPilot\recordings\20260530-082812_Arm_Camera_snapshot.png
DEFAULT_REFERENCE_BOARD_POLYGON = np.array(
    [
        [617.0, 375.0],
        [1159.0, 187.0],
        [1292.0, 759.0],
        [746.0, 904.0],
    ],
    dtype=np.float32,
)

DEFAULT_EUROPEAN_GREEN_BOXES = np.array(
    [
        [[885.0, 294.0], [992.0, 294.0], [992.0, 408.0], [885.0, 408.0]],
        [[907.0, 454.0], [1029.0, 454.0], [1029.0, 586.0], [907.0, 586.0]],
        [[1036.0, 300.0], [1252.0, 300.0], [1252.0, 556.0], [1036.0, 556.0]],
        [[760.0, 626.0], [952.0, 626.0], [952.0, 852.0], [760.0, 852.0]],
    ],
    dtype=np.float32,
)


@dataclass(frozen=True)
class GreenCrabDetection:
    """One projected European green crab region."""

    index: int
    polygon: tuple[tuple[float, float], ...]
    bbox: tuple[int, int, int, int]
    visible_fraction: float


@dataclass(frozen=True)
class CrabDetectionResult:
    """Detected crab-board pose and European green crab regions."""

    count: int
    detections: tuple[GreenCrabDetection, ...]
    reference_image: Path
    board_polygon: tuple[tuple[float, float], ...]
    match_count: int
    inlier_count: int
    detector_name: str
    confidence: float


class CrabDetectionError(RuntimeError):
    """Raised when the detector cannot be initialized or matched."""


def default_reference_image_path() -> Path | None:
    """Return the best available default crab reference image."""
    env_path = os.environ.get(ENV_REFERENCE_IMAGE, "").strip()
    candidates: list[Path] = []
    if env_path:
        candidates.append(Path(env_path).expanduser())

    candidates.extend(
        [
            REPO_ROOT / "data" / "crab_board_reference.png",
            REPO_ROOT / "Workspace" / "sources" / "crab_board_reference.png",
            REPO_ROOT.parent / "TritonPilot" / DEFAULT_REFERENCE_RELATIVE,
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def _create_feature_detector() -> tuple[Any, int, str, float]:
    try:
        return cv2.SIFT_create(nfeatures=2500, contrastThreshold=0.02), cv2.NORM_L2, "SIFT", 0.72
    except Exception:
        return cv2.ORB_create(nfeatures=3500, fastThreshold=5), cv2.NORM_HAMMING, "ORB", 0.76


def _polygon_tuple(points: np.ndarray) -> tuple[tuple[float, float], ...]:
    return tuple((float(point[0]), float(point[1])) for point in np.asarray(points, dtype=np.float32))


def _bbox_from_polygon(points: np.ndarray, image_shape: tuple[int, int, int] | tuple[int, int]) -> tuple[int, int, int, int]:
    height, width = image_shape[:2]
    pts = np.asarray(points, dtype=np.float32)
    x0 = int(np.floor(np.clip(np.min(pts[:, 0]), 0, max(0, width - 1))))
    y0 = int(np.floor(np.clip(np.min(pts[:, 1]), 0, max(0, height - 1))))
    x1 = int(np.ceil(np.clip(np.max(pts[:, 0]), 0, max(0, width - 1))))
    y1 = int(np.ceil(np.clip(np.max(pts[:, 1]), 0, max(0, height - 1))))
    return x0, y0, max(0, x1 - x0), max(0, y1 - y0)


def _visible_fraction(points: np.ndarray, image_shape: tuple[int, int, int] | tuple[int, int]) -> float:
    height, width = image_shape[:2]
    poly_area = abs(float(cv2.contourArea(np.asarray(points, dtype=np.float32))))
    if poly_area <= 1.0:
        return 0.0
    x, y, w, h = _bbox_from_polygon(points, image_shape)
    if w <= 0 or h <= 0:
        return 0.0
    return float(min(1.0, max(0.0, (w * h) / poly_area)))


def _valid_projected_board(points: np.ndarray, image_shape: tuple[int, int, int] | tuple[int, int]) -> bool:
    height, width = image_shape[:2]
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape != (4, 2) or not np.all(np.isfinite(pts)):
        return False
    area = abs(float(cv2.contourArea(pts)))
    image_area = float(max(1, width * height))
    if area < image_area * 0.002 or area > image_area * 0.85:
        return False
    return bool(cv2.isContourConvex(np.round(pts).astype(np.int32)))


class CrabReferenceDetector:
    """Match the fixed crab board and project European green crab boxes."""

    def __init__(
        self,
        reference_image: str | Path | None = None,
        *,
        board_polygon: np.ndarray = DEFAULT_REFERENCE_BOARD_POLYGON,
        green_boxes: np.ndarray = DEFAULT_EUROPEAN_GREEN_BOXES,
    ):
        resolved = Path(reference_image).expanduser().resolve() if reference_image else default_reference_image_path()
        if resolved is None:
            raise CrabDetectionError(
                f"No crab reference image found. Set {ENV_REFERENCE_IMAGE} or place crab_board_reference.png in data/."
            )

        reference = cv2.imread(str(resolved), cv2.IMREAD_COLOR)
        if reference is None:
            raise CrabDetectionError(f"Could not read crab reference image: {resolved}")

        self.reference_image = resolved
        self.board_polygon = np.asarray(board_polygon, dtype=np.float32)
        self.green_boxes = np.asarray(green_boxes, dtype=np.float32)
        self._detector, self._norm, self.detector_name, self._ratio = _create_feature_detector()
        gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.round(self.board_polygon).astype(np.int32), 255)
        self._keypoints, self._descriptors = self._detector.detectAndCompute(gray, mask)
        if self._descriptors is None or len(self._keypoints) < 8:
            raise CrabDetectionError(f"Crab reference image does not have enough board features: {resolved}")
        self._matcher = cv2.BFMatcher(self._norm)

    def detect(self, image_bgr: np.ndarray, *, min_inliers: int = 8) -> CrabDetectionResult | None:
        """Detect European green crab boxes in ``image_bgr``."""
        if image_bgr is None or image_bgr.size == 0:
            return None
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self._detector.detectAndCompute(gray, None)
        if descriptors is None or len(keypoints) < 8:
            return None

        matches = self._matcher.knnMatch(self._descriptors, descriptors, k=2)
        good_matches = [m for m, n in matches if m.distance < self._ratio * n.distance]
        if len(good_matches) < max(10, min_inliers):
            return None

        src = np.float32([self._keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst = np.float32([keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        homography, inlier_mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if homography is None or inlier_mask is None:
            return None
        inlier_count = int(np.count_nonzero(inlier_mask))
        if inlier_count < min_inliers:
            return None

        projected_board = cv2.perspectiveTransform(self.board_polygon.reshape(-1, 1, 2), homography).reshape(-1, 2)
        if not _valid_projected_board(projected_board, image_bgr.shape):
            return None

        detections: list[GreenCrabDetection] = []
        for index, ref_box in enumerate(self.green_boxes, start=1):
            projected = cv2.perspectiveTransform(ref_box.reshape(-1, 1, 2), homography).reshape(-1, 2)
            visible = _visible_fraction(projected, image_bgr.shape)
            bbox = _bbox_from_polygon(projected, image_bgr.shape)
            if visible < 0.15 or bbox[2] <= 2 or bbox[3] <= 2:
                continue
            detections.append(
                GreenCrabDetection(
                    index=index,
                    polygon=_polygon_tuple(projected),
                    bbox=bbox,
                    visible_fraction=visible,
                )
            )

        confidence = min(1.0, inlier_count / 80.0) * min(1.0, len(good_matches) / 120.0)
        return CrabDetectionResult(
            count=len(detections),
            detections=tuple(detections),
            reference_image=self.reference_image,
            board_polygon=_polygon_tuple(projected_board),
            match_count=len(good_matches),
            inlier_count=inlier_count,
            detector_name=self.detector_name,
            confidence=float(confidence),
        )


@lru_cache(maxsize=4)
def _cached_detector(reference_image: str | None) -> CrabReferenceDetector:
    return CrabReferenceDetector(reference_image)


def detect_european_green_crabs(
    image_bgr: np.ndarray,
    *,
    reference_image: str | Path | None = None,
) -> CrabDetectionResult | None:
    """Detect European green crabs using the fixed-board reference detector."""
    key = str(Path(reference_image).expanduser().resolve()) if reference_image else None
    return _cached_detector(key).detect(image_bgr)


def detection_summary_text(result: CrabDetectionResult | None) -> str:
    """Return a short operator-facing summary."""
    if result is None:
        return "Could not match the crab board."
    return (
        f"European green crabs: {result.count} | "
        f"{result.inlier_count}/{result.match_count} reference matches | "
        f"confidence {result.confidence:.2f}"
    )


def draw_european_green_crab_detections(image_bgr: np.ndarray, result: CrabDetectionResult | None) -> np.ndarray:
    """Draw board outline and European green crab boxes on an image copy."""
    annotated = image_bgr.copy()
    if result is None:
        cv2.putText(
            annotated,
            "Crab board not matched",
            (24, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return annotated

    board = np.round(np.asarray(result.board_polygon, dtype=np.float32)).astype(np.int32)
    cv2.polylines(annotated, [board], True, (80, 220, 80), 3, cv2.LINE_AA)
    for detection in result.detections:
        poly = np.round(np.asarray(detection.polygon, dtype=np.float32)).astype(np.int32)
        cv2.polylines(annotated, [poly], True, (0, 180, 255), 3, cv2.LINE_AA)
        label_anchor = poly[np.argmin(poly[:, 1])]
        x = int(label_anchor[0])
        y = int(label_anchor[1])
        cv2.putText(
            annotated,
            f"green {detection.index}",
            (x, max(18, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 180, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        annotated,
        f"European green crabs: {result.count}",
        (24, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 180, 255),
        2,
        cv2.LINE_AA,
    )
    return annotated
