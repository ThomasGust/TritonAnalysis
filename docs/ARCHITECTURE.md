# TritonAnalysis Architecture

TritonAnalysis uses a simple applet architecture: top-level entry points create
PyQt windows, GUI classes handle operator interaction, and pure analysis modules
perform the calculations. This keeps task logic testable without launching the
full GUI.

## System Boundary

TritonAnalysis owns:

- Mission-specific computer vision
- Mission-specific calculations
- Manual data-entry applets
- Saved media preprocessing
- Judge-facing reports and exports

TritonAnalysis does not own:

- ROV control
- Controller input
- Telemetry subscription
- Live camera streaming
- Onboard hardware access

Those responsibilities belong to TritonPilot and TritonOS.

## Entry Point Pattern

Each `main_*.py` file follows the same pattern:

1. Build an `argparse.ArgumentParser`.
2. Create a `QApplication`.
3. Apply the shared GUI style from `gui/style.py`.
4. Construct the applet window from `gui/`.
5. Show the window and enter the Qt event loop.

This makes applets easy to launch with `python -m ...` while keeping their
logic in importable modules.

## Core Logic Modules

The core analysis modules are ordinary Python modules with functions and data
classes:

- `crab_detector_cv.py` detects boards, unwraps images, classifies crab
  candidates, scores video samples, and renders annotated outputs.
- `iceberg_tracking.py` handles coordinate conversion, closest-approach
  geometry, threat levels, survey sequence validation, and report generation.
- `iceberg_measurement.py` measures iceberg PVC segments using affine,
  line-endpoint, and spatial calibration helpers.
- `planar_measurement.py` performs homography-based planar height and segment
  measurement.
- `coral_garden_model.py` builds a three-prism coral garden model and exports
  OBJ text.
- `edna_analysis.py` calculates percent frequencies and formats judge reports
  or CSV text.
- `color_corr.py` contains both GUI and image-processing classes for video
  correction and frame export.

Tests should target these modules directly when possible.

## GUI Layer

The `gui/` package owns PyQt windows, canvases, dialogs, styling, and responsive
layout helpers. It should gather operator input, preview media, display errors,
and call the core modules for calculations.

Most GUI windows support one or more of these interaction patterns:

- Load image or video media.
- Select a representative frame.
- Click calibration or measurement points.
- Enter reference lengths or task values.
- Review generated results.
- Export or copy results.

The GUI should not duplicate formulas that already live in core modules.

## Data Flow

Image/video applets:

```text
saved media file
        |
        v
PyQt window / OpenCV frame loading
        |
        v
core analysis module
        |
        v
annotated frame, measurement result, report, CSV, or export file
```

Manual applets:

```text
operator-entered task values
        |
        v
core calculation module
        |
        v
judge-facing report, table, CSV, or visual model
```

## Extending The Repository

When adding a new applet:

1. Put reusable calculation logic in a top-level module.
2. Add a `gui/` window for interaction.
3. Add a `main_*.py` entry point.
4. Add tests for core calculations.
5. Add GUI smoke or layout tests when the window has important interaction
   surfaces.
6. Update `README.md`, `docs/APPLET_REFERENCE.md`, and this architecture page.

Avoid adding live ROV dependencies unless the feature has been deliberately
reassigned out of TritonAnalysis.
