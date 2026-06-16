# TritonAnalysis Applet Reference

This page describes what each applet does, what it consumes, and where its
implementation lives.

## Unified Competition App

Launch:

```powershell
python -m triton_analysis.apps.main_triton_analysis
python main_triton_analysis.py
python -m triton_analysis.apps.main_triton_analysis --stereo-manifest path\to\stereo_session --calibration path\to\stereo_calibration.json
```

Purpose:

- Keep the competition analysis workflows in one tabbed window.
- Avoid relaunching applets during the 15-minute demonstration window.
- Preserve the standalone applets as focused backups.

Tabs:

- Coral Reconstruction
- Stereo Iceberg Length
- Iceberg Tracking
- eDNA Analysis
- Stereo Calibration
- Backup Coral Measurement
- Backup Iceberg Measurement
- SSH

Primary modules:

- `triton_analysis/apps/main_triton_analysis.py`
- `triton_analysis/gui/triton_analysis_window.py`
- The same window modules used by the standalone applets

## SSH Console

Launch:

```powershell
python -m triton_analysis.apps.main_triton_analysis --tab ssh
python main_triton_analysis.py --tab ssh
```

Purpose:

- Provide a field setup shell without leaving the unified TritonAnalysis app.
- Connect to the Pilot analysis link, a routed ROV address, or localhost using
  saved presets.
- Support password, SSH agent, and private-key authentication.

Primary modules:

- `triton_analysis/gui/ssh_console_window.py`
- `triton_analysis/gui/triton_analysis_window.py`

Inputs:

- Host, port, username, and optional password or private key path
- Trust-host setting for accepting newly seen host keys
- `Triton Pi` preset for `triton@tritonpi.local` with password `triton`

Outputs:

- Interactive terminal output inside the SSH tab
- Connection status and command history for the current app session

Setup notes:

- Requires `paramiko`, installed by `requirements.txt` and verified by
  `setup_windows.ps1`.
- The shared Triton Pi credential is intentionally stored as a default preset
  for field convenience.
- Does not require the Windows OpenSSH client executable to be installed.

## Iceberg Tracking

Launch:

```powershell
python -m triton_analysis.apps.main_iceberg_tracking
```

Purpose:

- Assess platform threat levels from iceberg position, heading, and keel depth.
- Convert and format coordinates.
- Evaluate survey-number sequences.
- Generate judge-facing report text.

Primary modules:

- `triton_analysis/apps/main_iceberg_tracking.py`
- `triton_analysis/gui/iceberg_tracking_window.py`
- `triton_analysis/iceberg/tracking.py`

Inputs:

- Manually entered iceberg latitude/longitude
- Iceberg heading
- Keel depth
- Survey numbers
- Built-in platform table from `iceberg_tracking.DEFAULT_PLATFORMS`

Outputs:

- Surface and subsea threat levels
- Closest-approach geometry
- Map visualization
- Report text

## Iceberg Measurement

Launch:

```powershell
python -m triton_analysis.apps.main_iceberg_measurement [image-or-video ...]
```

Purpose:

- Measure the iceberg hull variable PVC segment from saved media.
- Support affine and more constrained line-endpoint measurement workflows.

Primary modules:

- `triton_analysis/apps/main_iceberg_measurement.py`
- `triton_analysis/gui/iceberg_measurement_window.py`
- `triton_analysis/iceberg/measurement.py`

Inputs:

- Image or video file
- Clicked calibration points
- Known reference lengths/geometric constraints
- Clicked variable-segment endpoints

Outputs:

- Segment length in centimeters
- Diagnostic values such as reprojection or alignment error, depending on mode

## Planar Height Measurement

Launch:

```powershell
python -m triton_analysis.apps.main_planar_height_measurement [image-or-video ...]
```

Purpose:

- Measure a height lying on a known planar surface after perspective
  rectification.

Primary modules:

- `triton_analysis/apps/main_planar_height_measurement.py`
- `triton_analysis/gui/planar_height_measurement_window.py`
- `triton_analysis/measurement/planar.py`

Inputs:

- Image or video file
- Rectangle/corner geometry
- In-plane reference segments and lengths
- Height start/end points

Outputs:

- Height in centimeters
- Reference fit diagnostics
- Rectified-plane measurement state

## Multi-Rectangle Length Measurement

Launch:

```powershell
python -m triton_analysis.apps.main_multi_rect_length_measurement [image-or-video ...]
```

Purpose:

- Measure one or more lengths using a multi-rectangle planar workflow.
- Support quick and refined source-point modes.

Primary modules:

- `triton_analysis/apps/main_multi_rect_length_measurement.py`
- `triton_analysis/gui/multi_rect_length_measurement_window.py`
- `triton_analysis/measurement/planar.py`

Inputs:

- Image or video file
- Source rectangle/vanishing geometry
- Reference segment lengths
- One or more measurement segments

Outputs:

- Segment lengths in centimeters
- Per-segment visual overlays and fit information

## Stereo Calibration

Launch:

```powershell
python -m triton_analysis.apps.main_stereo_calibration_gui [manifest.json]
python -m triton_analysis.apps.main_stereo_calibration path\to\manifest.json [more-manifests-or-folders ...] --charuco
```

Purpose:

- Calibrate a stereo rig from TritonPilot left/right capture sessions.
- Preview saved image pairs from one or more manifests, choose board settings,
  run calibration, review quality diagnostics, and export an OpenCV calibration
  artifact.

Primary modules:

- `triton_analysis/apps/main_stereo_calibration_gui.py`
- `triton_analysis/apps/main_stereo_calibration.py`
- `triton_analysis/gui/stereo_calibration_window.py`
- `triton_analysis/stereo/calibration.py`

Inputs:

- TritonPilot stereo `manifest.json` files or stereo session folders
- Checkerboard or ChArUco board dimensions; defaults match Triton's ChArUco
  board: 12 columns, 9 rows, 6 cm squares, 4.5 cm markers,
  `DICT_5X5_1000`
- Minimum accepted pair/corner thresholds

Outputs:

- Stereo calibration JSON artifact
- RMS, epipolar error, coverage, and baseline summary
- Rejected observation notes

## Stereo Depth

Launch:

```powershell
python -m triton_analysis.apps.main_stereo_depth path\to\manifest.json
python -m triton_analysis.apps.main_stereo_depth path\to\manifest.json --calibration path\to\stereo_calibration.json
```

Purpose:

- Apply a stereo calibration artifact to saved TritonPilot image pairs.
- Preview rectified left/right images, compute disparity and depth maps, and
  make quick 3D point-to-point length checks.
- Measure low-texture PVC or body-part spans by clicking corresponding
  endpoints in both rectified previews for direct triangulation.
- Tune block matching parameters enough to diagnose whether calibration,
  exposure, sync, or texture is limiting depth quality.

Primary modules:

- `triton_analysis/apps/main_stereo_depth.py`
- `triton_analysis/gui/stereo_depth_window.py`
- `triton_analysis/stereo/depth.py`
- `triton_analysis/stereo/calibration.py`

Inputs:

- TritonPilot stereo `manifest.json` file or session folder
- `stereo_calibration.json` artifact, auto-loaded from the session folder when
  present
- Optional disparity/depth filter parameters

Outputs:

- Rectified left/right previews
- Disparity and depth heatmaps
- Valid-depth coverage and median-depth summary
- Dense-depth or manual-correspondence 3D sample coordinates and two-point
  distance in calibration units

## Stereo Segment Measurement

Launch:

```powershell
python -m triton_analysis.apps.main_stereo_segment_measurement path\to\manifest.json
python -m triton_analysis.apps.main_stereo_segment_measurement path\to\manifest.json --preset coral
python -m triton_analysis.apps.main_stereo_iceberg_measurement path\to\manifest.json
python -m triton_analysis.apps.main_stereo_iceberg_measurement path\to\manifest.json --calibration path\to\stereo_calibration.json
```

Purpose:

- Measure straight-line 3D segments from a calibrated stereo pair.
- Provide presets for Generic Segment, Iceberg Keel, and Coral Rig Length
  labels/report wording.
- Use direct endpoint correspondences in rectified left/right
  previews instead of relying on dense disparity over low-texture PVC.
- Record repeated frame measurements and report median/spread diagnostics.
- Auto-correct reversed endpoint order on the right image, with a manual
  `Swap Right` control for ambiguous horizontal segments.

Primary modules:

- `triton_analysis/apps/main_stereo_segment_measurement.py`
- `triton_analysis/apps/main_stereo_iceberg_measurement.py`
- `triton_analysis/gui/stereo_segment_measurement_window.py`
- `triton_analysis/gui/stereo_iceberg_measurement_window.py`
- `triton_analysis/stereo/segment_measurement.py`
- `triton_analysis/stereo/iceberg_measurement.py`
- `triton_analysis/stereo/depth.py`

Inputs:

- TritonPilot stereo `manifest.json` file or session folder
- `stereo_calibration.json` artifact, auto-loaded from the session folder when
  present
- Selected measurement preset
- Clicked segment endpoints in both rectified previews

Outputs:

- Segment length in calibration units, centimeters, and meters when units are
  known
- Per-endpoint vertical rectification error and disparity diagnostics
- Repeated-measurement median and spread for judge-facing confidence

## RealityScan Stereo Reconstruction

Launch:

```powershell
python -m triton_analysis.apps.main_realityscan_reconstruction path\to\stereo_session --calibration path\to\stereo_calibration.json
```

Purpose:

- Wrap the existing Triton stereo RealityScan photogrammetry pipeline in a
  standalone Analysis GUI.
- Select a TritonPilot stereo session and matching calibration artifact.
- Configure reconstruction presets, metric scaling, alignment-tournament
  options, and advanced CLI budgets without retyping a long command.
- Default to the fast `Flat Luma K+` single-variant path, with a
  `Caustic Stable Luma K+` diagnostic path and a `Legacy Enhanced Brown4`
  switch for reproducing the previous default.
- Keep bridge frames across temporal or appearance gaps so higher capture
  frequency becomes a useful candidate pool instead of a pile of duplicates.
- Write color texture-layer sidecars by default so RealityScan can align and
  mesh from luma-stabilized frames while texturing the final model in color.
- Write ranked component summaries and optionally export top fragments as
  `.rsalign` components for alignment diagnostics.
- Stream pipeline/RealityScan progress into the applet and collect output
  paths for review.
- Load the exported OBJ into the adjacent Model Viewer tab for inspection and
  measurement.

Primary modules:

- `triton_analysis/apps/main_realityscan_reconstruction.py`
- `triton_analysis/gui/realityscan_reconstruction_window.py`
- TritonAnalysis pipeline package selected in the GUI's `Pipeline Root` field

Inputs:

- TritonPilot stereo session folder or `manifest.json`
- `stereo_calibration.json`
- RealityScan/RealityCapture executable, auto-detected when possible
- Pipeline root containing `triton_analysis/realityscan/underwater_pipeline.py`

Outputs:

- New timestamped RealityScan workspace under `Workspace/results/realityscan` by default
- Contact sheet, logs, reports, connectivity/component CSVs, project file, and OBJ export
- Metric-scaled OBJ and `metric_scale.json` when stereo scaling succeeds

## RealityScan Model Viewer

Launch:

```powershell
python -m triton_analysis.apps.main_realityscan_model_viewer path\to\underwater_model_metric.obj
```

Purpose:

- Open a reconstructed OBJ model in a Three.js viewport.
- Orbit, pan, zoom, frame the model, and inspect the textured mesh in an
  embedded Qt WebEngine view or browser fallback.
- Create multiple two-point measurement lines and report straight-line
  distances in centimeters by default, with meters or millimeters available.
- Show a small distance label over each completed measurement line in the
  viewport.
- Drag existing endpoints along the mesh surface, delete selected points or
  lines, and clear the measurement set.
- Move the visible 3D orbit cursor to a mesh point and rotate around that
  center.

Primary modules:

- `triton_analysis/apps/main_realityscan_model_viewer.py`
- `triton_analysis/gui/realityscan_model_viewer_window.py`

Inputs:

- RealityScan OBJ model, preferably the metric-scaled
  `underwater_model_metric.obj`
- Adjacent MTL and texture files generated by RealityScan

Outputs:

- Localhost Three.js viewer URL
- Interactive distance readout for selected point pairs

## Coral Garden Model

Launch:

```powershell
python -m triton_analysis.apps.main_coral_garden_model
```

Purpose:

- Build and preview a manual three-prism coral garden model from length,
  height, and width values.
- Export OBJ text for CAD/modeling workflows.

Primary modules:

- `triton_analysis/apps/main_coral_garden_model.py`
- `triton_analysis/gui/coral_garden_model_window.py`
- `triton_analysis/coral/garden_model.py`

Inputs:

- Length in centimeters
- Height in centimeters
- Optional width in centimeters

Outputs:

- 3D preview
- Dimension guides
- PNG preview export
- OBJ export

## eDNA Frequency Analysis

Launch:

```powershell
python -m triton_analysis.apps.main_edna_analysis
python -m triton_analysis.apps.main_edna_analysis --sample
```

Purpose:

- Calculate percent frequency from observed species counts.
- Generate judge-facing report text and CSV data.

Primary modules:

- `triton_analysis/apps/main_edna_analysis.py`
- `triton_analysis/gui/edna_analysis_window.py`
- `triton_analysis/edna/analysis.py`

Inputs:

- Manual count for each species in `DEFAULT_SPECIES`

Outputs:

- Total organisms seen
- Percent frequency per species
- Report text
- CSV text/export

## Underwater Color Correction

Launch:

```powershell
python -m triton_analysis.apps.color_corr
```

Purpose:

- Preprocess underwater video.
- Improve visibility of white PVC-like structures and red targets.
- Export corrected video or selected frames for later analysis/modeling.

Primary modules:

- `triton_analysis/apps/color_corr.py`
- `triton_analysis/gui/responsive.py`

Inputs:

- Saved video file
- Operator-selected processing settings
- Frame-selection settings

Outputs:

- Corrected previews
- Exported video
- Exported frame sets
- CSV/metadata for selected frames, depending on workflow
