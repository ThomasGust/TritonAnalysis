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

The repository is organized as a unified competition app, packaged backup
applets, and task-focused analysis modules:

- Unified tabbed competition-day analysis window
- Crab detection for the fixed reference board, including European green crab boxes and synthetic YOLO data
- Iceberg tracking and threat assessment
- Iceberg PVC segment measurement
- Planar height measurement
- Multi-rectangle planar length measurement
- Coral garden manual CAD model generation
- eDNA frequency analysis
- Underwater color correction and frame export
- Stereo rectification, disparity, depth, and length checks
- Stereo segment measurement presets for generic, iceberg, and coral lengths
- RealityScan stereo photogrammetry reconstruction wrapper
- Three.js OBJ viewport and distance measurement for reconstructed models

Each applet can be launched directly with `python -m ...` from the repository
root.

## Repository Layout

```text
main_triton_analysis.py                                  Top-level unified GUI launcher
triton_analysis/apps/main_crab_detection.py                 Crab detection GUI entry point
triton_analysis/apps/main_triton_analysis.py                Unified tabbed competition-day GUI
triton_analysis/apps/main_iceberg_tracking.py               Iceberg threat-assessment GUI
triton_analysis/apps/main_iceberg_measurement.py            Iceberg variable-segment GUI
triton_analysis/apps/main_planar_height_measurement.py      Planar height measurement GUI
triton_analysis/apps/main_multi_rect_length_measurement.py  Multi-rectangle length GUI
triton_analysis/apps/main_coral_garden_model.py             Coral garden CAD-model GUI
triton_analysis/apps/main_edna_analysis.py                  eDNA frequency GUI
triton_analysis/apps/main_stereo_depth.py                   Stereo depth/length-check GUI
triton_analysis/apps/main_stereo_segment_measurement.py     Stereo segment measurement GUI
triton_analysis/apps/main_stereo_iceberg_measurement.py     Iceberg preset shortcut
triton_analysis/apps/main_realityscan_reconstruction.py     Stereo RealityScan reconstruction GUI
triton_analysis/apps/main_realityscan_model_viewer.py       Three.js OBJ measurement viewer
triton_analysis/apps/color_corr.py                          Underwater correction/frame-export GUI
triton_analysis/sync/pilot_transfer.py                      Pull-only TritonPilot media sync helper
triton_analysis/crab/detector.py                       Reference-board crab detector
tools/crab_yolo_train.py               YOLO fine-tuning helper for green crabs
tools/crab_yolo_predict.py             YOLO image prediction helper for green crabs
triton_analysis/iceberg/tracking.py                    Coordinate/threat-assessment logic
triton_analysis/iceberg/measurement.py                 2D/3D measurement algorithms
triton_analysis/measurement/planar.py                  Planar homography measurement algorithms
triton_analysis/stereo/calibration.py                  Stereo calibration artifact generation
triton_analysis/stereo/depth.py                        Stereo rectification/disparity helpers
triton_analysis/stereo/segment_measurement.py          Stereo segment endpoint measurement helpers
triton_analysis/stereo/iceberg_measurement.py          Iceberg measurement compatibility wrappers
triton_analysis/coral/garden_model.py                  Prism model and OBJ export
triton_analysis/edna/analysis.py                       eDNA frequency calculations and reports
triton_analysis/gui/                                   PyQt windows and responsive helpers
tools/                                 Batch/CLI helper tools
data/                                  Bundled app/test assets
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

Unified competition window:

```powershell
python main_triton_analysis.py
python -m triton_analysis.apps.main_triton_analysis
python -m triton_analysis.apps.main_triton_analysis --stereo-manifest path\to\stereo_session --calibration path\to\stereo_calibration.json
python -m triton_analysis.apps.main_triton_analysis --pilot-transfer-url http://10.77.0.1:8765
python -m triton_analysis.apps.main_triton_analysis --workspace D:\TritonAnalysisWorkspace
```

The unified window opens with Coral Reconstruction, then Crab Detection, Stereo
Iceberg Length, Iceberg Tracking, eDNA Analysis, Stereo Calibration, Backup
Coral Measurement, and Backup Iceberg Measurement. The standalone applets below
remain available as backups.
The status bar includes automatic TritonPilot media sync status and the active
destination folder.

By default, synced media and generated outputs live under `.\Workspace` inside
this checkout. The `Workspace` menu can move that root on each computer while
keeping subfolders like `incoming\pilot`, `results`, `reports`, and
`calibrations` consistent.

Crab competition analyzer:

```powershell
python -m triton_analysis.apps.main_crab_detection [image-folder-or-file ...]
python -m triton_analysis.apps.main_crab_detection path\to\archive --detector yolo --yolo-model Workspace\models\crab_yolo\run\weights\best.pt
python -m tools.crab_image_detect path\to\images --output-dir path\to\results
python -m tools.crab_generate_synthetic_dataset path\to\images --train-count 700 --val-count 180
python -m tools.crab_yolo_train Workspace\datasets\crab_green_yolo_YYYYMMDD_HHMMSS\data.yaml
python -m tools.crab_yolo_predict path\to\images --model Workspace\models\crab_yolo\run\weights\best.pt
```

Iceberg tracking threat applet:

```powershell
python -m triton_analysis.apps.main_iceberg_tracking
```

Coral garden CAD model applet:

```powershell
python -m triton_analysis.apps.main_coral_garden_model
```

eDNA frequency analysis applet:

```powershell
python -m triton_analysis.apps.main_edna_analysis
python -m triton_analysis.apps.main_edna_analysis --sample
```

Measurement applets:

```powershell
python -m triton_analysis.apps.main_iceberg_measurement [image-or-video ...]
python -m triton_analysis.apps.main_planar_height_measurement [image-or-video ...]
python -m triton_analysis.apps.main_multi_rect_length_measurement [image-or-video ...]
```

Stereo calibration from TritonPilot capture sessions:

```powershell
python -m triton_analysis.apps.main_stereo_calibration_gui path\to\manifest.json
python -m triton_analysis.apps.main_stereo_calibration path\to\manifest.json [more-manifests-or-folders ...] --charuco
```

Stereo depth and 3D length checks:

```powershell
python -m triton_analysis.apps.main_stereo_depth path\to\manifest.json
python -m triton_analysis.apps.main_stereo_depth path\to\manifest.json --calibration path\to\stereo_calibration.json
python -m triton_analysis.apps.main_stereo_segment_measurement path\to\manifest.json
python -m triton_analysis.apps.main_stereo_segment_measurement path\to\manifest.json --preset coral
python -m triton_analysis.apps.main_stereo_iceberg_measurement path\to\manifest.json
python -m triton_analysis.apps.main_stereo_iceberg_measurement path\to\manifest.json --calibration path\to\stereo_calibration.json
```

For low-texture PVC structures, use the stereo depth applet's rectified
left/right endpoint clicks for direct triangulation; dense disparity is still
best treated as a diagnostic unless the object has visible texture.
For task-focused straight-line measurements, use the stereo segment applet and
choose the Generic Segment, Iceberg Keel, or Coral Rig Length preset. The older
stereo iceberg command still opens the same applet in Iceberg Keel mode.

RealityScan stereo reconstruction and model viewing:

```powershell
python -m triton_analysis.apps.main_realityscan_reconstruction path\to\stereo_session --calibration path\to\stereo_calibration.json
python -m triton_analysis.apps.main_realityscan_model_viewer path\to\underwater_model_metric.obj
```

The reconstruction GUI includes a Model Viewer tab that embeds the Three.js
viewport when `PyQt6-WebEngine` is installed, with a browser fallback.

Underwater color correction and frame export:

```powershell
python -m triton_analysis.apps.color_corr
```

Pilot media transfer helper:

```powershell
python -m tools.pilot_transfer_sync http://10.77.0.1:8765 --output ".\Workspace\incoming\pilot"
```

## Competition Workflow

Use TritonPilot to operate the ROV and capture clean media. Move saved files or
manual task values to the analysis computer. Use TritonAnalysis to run the
appropriate applet, export results, and prepare judge-facing numbers or
artifacts.

Keeping analysis off the pilot station protects the live-control computer from
long-running CV work, accidental UI clutter, and mission-specific changes.
