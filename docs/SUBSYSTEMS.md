# TritonAnalysis Subsystem Reference

This guide maps repository areas to responsibilities.

## Entry Points

`triton_analysis/apps/main_*.py` files are thin GUI launchers. They parse
command-line arguments, create a `QApplication`, apply
`triton_analysis/gui/style.py`, construct a window, and start the Qt event loop.

Keep entry points small. New reusable behavior should go in a core module or a
GUI class.

## Core Analysis Modules

- `triton_analysis/crab/detector.py` owns fixed-board reference matching, European green crab
  count/box projection, and annotated rendering.
- `triton_analysis/iceberg/tracking.py` owns coordinate conversions, platform geometry, threat
  assessment, survey validation, and report formatting.
- `triton_analysis/iceberg/measurement.py` owns iceberg variable-segment measurement algorithms
  and validation errors.
- `triton_analysis/stereo/segment_measurement.py` owns stereo endpoint triangulation,
  measurement presets, and repeated-measurement summaries.
- `triton_analysis/stereo/iceberg_measurement.py` provides compatibility wrappers for the
  iceberg keel preset.
- `triton_analysis/measurement/planar.py` owns homography-based planar height and segment
  measurement.
- `triton_analysis/coral/garden_model.py` owns prism construction, model bounds, formatting,
  and OBJ export.
- `triton_analysis/edna/analysis.py` owns species definitions, percent-frequency calculation,
  report text, and CSV text.
- `triton_analysis/apps/color_corr.py` owns underwater video correction and frame export. It also
  contains its GUI because the processing and controls are tightly coupled.

Prefer placing formulas and data transformations in these modules, then calling
them from GUI code.

## `triton_analysis/gui/`

The GUI package owns PyQt windows, dialogs, canvases, responsive helpers, and
shared styling:

- `crab_detection_window.py`
- `file_dialogs.py`
- `image_preview.py`
- `iceberg_tracking_window.py`
- `iceberg_measurement_window.py`
- `planar_height_measurement_window.py`
- `multi_rect_length_measurement_window.py`
- `stereo_segment_measurement_window.py`
- `stereo_iceberg_measurement_window.py`
- `coral_garden_model_window.py`
- `edna_analysis_window.py`
- `responsive.py`
- `style.py`

GUI code should focus on loading files, collecting clicks or typed values,
showing previews, displaying errors, and presenting results.

Shared file selectors use `file_dialogs.py` so timestamped media folders show
image thumbnails while operators browse. Stereo session folders preview images
from their manifest or immediate `left`/`right` subfolders without requiring
operators to enter those subfolders first.

## `tools/`

`tools/crab_image_detect.py` scans saved image files or folders, writes
annotated European green crab images, and records `summary.csv`.
`tools/crab_video_detect.py` remains a placeholder until video sampling is
rebuilt on top of the image detector.

Future tools should remain file-based and analysis-focused. If a tool needs
live vehicle state, reconsider which repository should own it.

## `data/`

`data/` contains small bundled assets when an applet needs them. Large
competition recordings should stay outside the repository and be copied into
local media folders as needed.

## `tests/`

Tests cover both pure analysis modules and important GUI responsiveness
behavior. Computer-vision tests that require larger local recordings are marked
`vision` and skip themselves when the recordings are not present.

When changing applet behavior, add tests near the core module first. Add GUI
tests when layout, responsiveness, or interaction wiring is the risk.

## Dependencies

The applets depend on PyQt6, OpenCV, NumPy, SciPy, and Matplotlib. They should
not require ZeroMQ, pygame, GStreamer, or ROV hardware.
