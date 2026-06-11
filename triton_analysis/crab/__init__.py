"""Crab vision helpers for the MATE ROV invasive species task."""

from triton_analysis.crab.synthetic import (
    CRAB_CLASS_NAMES,
    SyntheticDatasetConfig,
    SyntheticDatasetResult,
    discover_background_media,
    generate_synthetic_dataset,
)

__all__ = [
    "CRAB_CLASS_NAMES",
    "SyntheticDatasetConfig",
    "SyntheticDatasetResult",
    "discover_background_media",
    "generate_synthetic_dataset",
]
