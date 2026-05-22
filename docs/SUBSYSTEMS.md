# TritonAnalysis Subsystem Reference

This guide maps repository areas to responsibilities.

## Entry Points

Top-level `main_*.py` files are thin GUI launchers. They parse command-line
arguments, create a `QApplication`, apply `gui/style.py`, construct a window,
and start the Qt event loop.

Keep entry points small. New reusable behavior should go in a core module or a
GUI class.

## Core Analysis Modules

- `crab_detector_cv.py` owns crab board detection, image unwrapping, candidate
  masks, classification, video sample scoring, and annotated rendering.
- `iceberg_tracking.py` owns coordinate conversions, platform geometry, threat
  assessment, survey validation, and report formatting.
- `iceberg_measurement.py` owns iceberg variable-segment measurement algorithms
  and validation errors.
- `planar_measurement.py` owns homography-based planar height and segment
  measurement.
- `coral_garden_model.py` owns prism construction, model bounds, formatting,
  and OBJ export.
- `edna_analysis.py` owns species definitions, percent-frequency calculation,
  report text, and CSV text.
- `color_corr.py` owns underwater video correction and frame export. It also
  contains its GUI because the processing and controls are tightly coupled.

Prefer placing formulas and data transformations in these modules, then calling
them from GUI code.

## `gui/`

The GUI package owns PyQt windows, dialogs, canvases, responsive helpers, and
shared styling:

- `crab_detection_window.py`
- `crab_result_dialog.py`
- `iceberg_tracking_window.py`
- `iceberg_measurement_window.py`
- `planar_height_measurement_window.py`
- `multi_rect_length_measurement_window.py`
- `coral_garden_model_window.py`
- `edna_analysis_window.py`
- `responsive.py`
- `style.py`

GUI code should focus on loading files, collecting clicks or typed values,
showing previews, displaying errors, and presenting results.

## `tools/`

`tools/crab_video_detect.py` is a CLI helper for scanning video files with the
crab detector. It writes annotated outputs and a CSV summary to a results
folder.

Future tools should remain file-based and analysis-focused. If a tool needs
live vehicle state, reconsider which repository should own it.

## `data/`

`data/` contains small bundled assets such as crab samples and reference images.
These files support tests, demos, and detector behavior. Large competition
recordings should stay outside the repository and be copied into local media
folders as needed.

## `tests/`

Tests cover both pure analysis modules and important GUI responsiveness
behavior. Computer-vision tests that require larger local recordings are marked
`vision` and skip themselves when the recordings are not present.

When changing applet behavior, add tests near the core module first. Add GUI
tests when layout, responsiveness, or interaction wiring is the risk.

## Dependencies

The applets depend on PyQt6, OpenCV, NumPy, SciPy, and Matplotlib. They should
not require ZeroMQ, pygame, GStreamer, or ROV hardware.
