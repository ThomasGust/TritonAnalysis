# TritonAnalysis

TritonAnalysis contains standalone mission-analysis tools for Triton's
competition workflow. These applets run on an analysis laptop against saved
images, saved videos, or manually entered task data.

This repository is intentionally separate from TritonPilot and TritonOS.
TritonAnalysis does not talk to the ROV, publish pilot commands, subscribe to
live telemetry, start camera streams, or depend on the topside control UI. Its
job is to help the team interpret mission data without adding complexity to the
live piloting station.

## What Runs On The Analysis Computer

The repository is organized as top-level applets and shared analysis modules:

- Crab detection from images, folders, or video files
- Iceberg tracking and threat assessment
- Iceberg PVC segment measurement
- Planar height measurement
- Multi-rectangle planar length measurement
- Coral garden manual CAD model generation
- eDNA frequency analysis
- Underwater color correction and frame export
- Batch crab-video detection helper

Each applet can be launched directly with `python -m ...` from the repository
root.

## Repository Layout

```text
main_crab_detection.py                 Crab detection GUI entry point
main_iceberg_tracking.py               Iceberg threat-assessment GUI
main_iceberg_measurement.py            Iceberg variable-segment GUI
main_planar_height_measurement.py      Planar height measurement GUI
main_multi_rect_length_measurement.py  Multi-rectangle length GUI
main_coral_garden_model.py             Coral garden CAD-model GUI
main_edna_analysis.py                  eDNA frequency GUI
color_corr.py                          Underwater correction/frame-export GUI
crab_detector_cv.py                    Crab computer-vision pipeline
iceberg_tracking.py                    Coordinate/threat-assessment logic
iceberg_measurement.py                 2D/3D measurement algorithms
planar_measurement.py                  Planar homography measurement algorithms
coral_garden_model.py                  Prism model and OBJ export
edna_analysis.py                       eDNA frequency calculations and reports
gui/                                   PyQt windows and responsive helpers
tools/                                 Batch/CLI helper tools
data/                                  Reference images and bundled samples
tests/                                 Hardware-free tests and optional vision checks
docs/                                  Maintained user and maintainer docs
```

## Start Here

- [Documentation Index](docs/README.md)
- [Setup Guide](docs/SETUP.md)
- [Network And Data Handoff Guide](docs/NETWORKING.md)
- [Operations Guide](docs/OPERATIONS.md)
- [Architecture Overview](docs/ARCHITECTURE.md)
- [Applet Reference](docs/APPLET_REFERENCE.md)
- [Subsystem Reference](docs/SUBSYSTEMS.md)
- [Data And Inputs](docs/DATA_AND_INPUTS.md)
- [Testing And Troubleshooting](docs/TESTING_AND_TROUBLESHOOTING.md)

## Quick Start

On Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install pytest
python -m pytest
```

On macOS or Linux:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install pytest
python -m pytest
```

## Applets

Crab competition analyzer:

```powershell
python -m main_crab_detection [image-folder-or-video ...]
```

Iceberg tracking threat applet:

```powershell
python -m main_iceberg_tracking
```

Coral garden CAD model applet:

```powershell
python -m main_coral_garden_model
```

eDNA frequency analysis applet:

```powershell
python -m main_edna_analysis
python -m main_edna_analysis --sample
```

Measurement applets:

```powershell
python -m main_iceberg_measurement [image-or-video ...]
python -m main_planar_height_measurement [image-or-video ...]
python -m main_multi_rect_length_measurement [image-or-video ...]
```

Underwater color correction and frame export:

```powershell
python -m color_corr
```

Batch crab-video helper:

```powershell
python -m tools.crab_video_detect path\to\video.mp4
```

## Competition Workflow

Use TritonPilot to operate the ROV and capture clean media. Move saved files or
manual task values to the analysis computer. Use TritonAnalysis to run the
appropriate applet, export results, and prepare judge-facing numbers or
artifacts.

Keeping analysis off the pilot station protects the live-control computer from
long-running CV work, accidental UI clutter, and mission-specific changes.
