"""Crab vision helpers for the MATE ROV invasive species task."""

from triton_analysis.crab.synthetic import (
    CRAB_CLASS_NAMES,
    SyntheticDatasetConfig,
    SyntheticDatasetResult,
    discover_background_media,
    generate_synthetic_dataset,
)
from triton_analysis.crab.plane_dataset import (
    BoardPlaneAnnotation,
    PlaneProjectedDatasetConfig,
    discover_board_images,
    discover_default_crab_template_paths,
    generate_plane_projected_dataset,
    load_board_plane_annotations,
    save_board_plane_annotations,
)

__all__ = [
    "BoardPlaneAnnotation",
    "CRAB_CLASS_NAMES",
    "PlaneProjectedDatasetConfig",
    "SyntheticDatasetConfig",
    "SyntheticDatasetResult",
    "discover_background_media",
    "discover_board_images",
    "discover_default_crab_template_paths",
    "generate_plane_projected_dataset",
    "generate_synthetic_dataset",
    "load_board_plane_annotations",
    "save_board_plane_annotations",
]
