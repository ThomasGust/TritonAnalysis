# TritonAnalysis Applet Reference

This page describes what each applet does, what it consumes, and where its
implementation lives.

## Crab Detection

Launch:

```powershell
python -m main_crab_detection [image-folder-or-video ...]
```

Purpose:

- Detect competition crab board imagery.
- Count detected crab candidates.
- Classify supported species/reference-copy candidates.
- Render annotated views for review.

Primary modules:

- `main_crab_detection.py`
- `gui/crab_detection_window.py`
- `gui/crab_result_dialog.py`
- `crab_detector_cv.py`

Inputs:

- Image files
- Folders of image files
- A video file
- Optional manually selected board corners

Useful batch helper:

```powershell
python -m tools.crab_video_detect path\to\video.mp4 --output-dir path\to\results
```

## Iceberg Tracking

Launch:

```powershell
python -m main_iceberg_tracking
```

Purpose:

- Assess platform threat levels from iceberg position, heading, and keel depth.
- Convert and format coordinates.
- Evaluate survey-number sequences.
- Generate judge-facing report text.

Primary modules:

- `main_iceberg_tracking.py`
- `gui/iceberg_tracking_window.py`
- `iceberg_tracking.py`

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
python -m main_iceberg_measurement [image-or-video ...]
```

Purpose:

- Measure the iceberg hull variable PVC segment from saved media.
- Support affine and more constrained line-endpoint measurement workflows.

Primary modules:

- `main_iceberg_measurement.py`
- `gui/iceberg_measurement_window.py`
- `iceberg_measurement.py`

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
python -m main_planar_height_measurement [image-or-video ...]
```

Purpose:

- Measure a height lying on a known planar surface after perspective
  rectification.

Primary modules:

- `main_planar_height_measurement.py`
- `gui/planar_height_measurement_window.py`
- `planar_measurement.py`

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
python -m main_multi_rect_length_measurement [image-or-video ...]
```

Purpose:

- Measure one or more lengths using a multi-rectangle planar workflow.
- Support quick and refined source-point modes.

Primary modules:

- `main_multi_rect_length_measurement.py`
- `gui/multi_rect_length_measurement_window.py`
- `planar_measurement.py`

Inputs:

- Image or video file
- Source rectangle/vanishing geometry
- Reference segment lengths
- One or more measurement segments

Outputs:

- Segment lengths in centimeters
- Per-segment visual overlays and fit information

## Coral Garden Model

Launch:

```powershell
python -m main_coral_garden_model
```

Purpose:

- Build and preview a manual three-prism coral garden model from length,
  height, and width values.
- Export OBJ text for CAD/modeling workflows.

Primary modules:

- `main_coral_garden_model.py`
- `gui/coral_garden_model_window.py`
- `coral_garden_model.py`

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
python -m main_edna_analysis
python -m main_edna_analysis --sample
```

Purpose:

- Calculate percent frequency from observed species counts.
- Generate judge-facing report text and CSV data.

Primary modules:

- `main_edna_analysis.py`
- `gui/edna_analysis_window.py`
- `edna_analysis.py`

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
python -m color_corr
```

Purpose:

- Preprocess underwater video.
- Improve visibility of white PVC-like structures and red targets.
- Export corrected video or selected frames for later analysis/modeling.

Primary modules:

- `color_corr.py`
- `gui/responsive.py`

Inputs:

- Saved video file
- Operator-selected processing settings
- Frame-selection settings

Outputs:

- Corrected previews
- Exported video
- Exported frame sets
- CSV/metadata for selected frames, depending on workflow
