# TritonAnalysis Operations Guide

This guide describes how to use TritonAnalysis during development, practice,
and competition. Competition runs should normally use the unified tabbed app so
the analysis operator can move between tasks without relaunching tools. The
standalone applets remain available as backups.

## Competition-Day Pattern

1. TritonPilot records source media or the team writes down task values.
2. The analysis operator copies or syncs the source media or values to the
   analysis computer.
3. The operator launches `main_triton_analysis` or a focused backup applet.
4. The applet produces counts, measurements, corrected frames, CSV text, OBJ
   files, reports, or annotated images.
5. The team preserves both source evidence and exported results.

Do not run TritonAnalysis as part of the live pilot UI. It is a separate station
workflow.

## Before The Event

On the analysis computer:

- Create and test the virtual environment.
- Run `python -m pytest`.
- Open each GUI applet once.
- Confirm the TritonAnalysis workspace root, normally `.\Workspace` inside the
  checkout.
- Practice the USB drive or dedicated-Ethernet transfer workflow from
  [Network And Data Handoff Guide](NETWORKING.md).
- Practice the exact applet workflow with representative images and videos.

## Launching The Unified App

From the TritonAnalysis repository root:

```powershell
.\.venv\Scripts\activate
```

Competition-day tabbed app:

```powershell
python -m main_triton_analysis
python -m main_triton_analysis --stereo-manifest path\to\stereo_session --calibration path\to\stereo_calibration.json
```

The unified app includes Coral Reconstruction, Crab Detection, Stereo Iceberg
Length, Iceberg Tracking, eDNA Analysis, Stereo Calibration, Backup Coral
Measurement, and Backup Iceberg Measurement tabs in one window. RealityScan
reconstruction runs in a `QProcess` and stereo calibration runs in a worker
thread, so those jobs do not block the other tabs.

The top `Pilot Sync` panel shows the Pilot URL, exact receiving folder, last
sync time, and live file-transfer state. When auto sync is enabled, the app
periodically pulls new saved TritonPilot files into `Workspace\incoming\pilot`
without blocking the analysis tabs. Use the `Workspace` menu to change the root
folder and the `Pilot Sync` menu to change the URL, destination folder, or
trigger a manual sync.

## Backup Applets

Crab detection:

```powershell
python -m main_crab_detection path\to\images
python -m tools.crab_image_detect path\to\images --output-dir path\to\results
```

Iceberg tracking:

```powershell
python -m main_iceberg_tracking
```

Iceberg measurement:

```powershell
python -m main_iceberg_measurement path\to\image_or_video
```

Stereo segment measurement:

```powershell
python -m main_stereo_segment_measurement path\to\stereo_session
python -m main_stereo_segment_measurement path\to\stereo_session --preset coral
python -m main_stereo_iceberg_measurement path\to\stereo_session
```

RealityScan stereo reconstruction:

```powershell
python -m main_realityscan_reconstruction path\to\stereo_session --calibration path\to\stereo_calibration.json
python -m main_realityscan_model_viewer path\to\underwater_model_metric.obj
```

Planar height measurement:

```powershell
python -m main_planar_height_measurement path\to\image_or_video
```

Multi-rectangle length measurement:

```powershell
python -m main_multi_rect_length_measurement path\to\image_or_video
```

Coral garden model:

```powershell
python -m main_coral_garden_model
```

eDNA frequency:

```powershell
python -m main_edna_analysis
```

Color correction:

```powershell
python -m color_corr
```

## Crab Detection Workflow

The rebuilt crab detector assumes the printed competition board artwork is
fixed. It matches a known reference photo of that board into the current image,
projects the four European green crab regions into camera coordinates, and draws
bounding boxes around them.

Use the GUI for review:

```powershell
python -m main_crab_detection path\to\images
```

The GUI's `Auto` detector mode uses the newest trained YOLO crab weights under
`Workspace\models\crab_yolo` when available; otherwise it falls back to board
projection. To force a specific YOLO model for archive review:

```powershell
python -m main_crab_detection path\to\archive --detector yolo --yolo-model Workspace\models\crab_yolo\run\weights\best.pt
```

Use the batch image helper when processing many still photos:

```powershell
python -m tools.crab_image_detect path\to\images --output-dir path\to\results
```

If the default local reference image is not available, pass `--reference-image`
or set `TRITON_CRAB_REFERENCE_IMAGE` to a clear image of the same board.

To build a trainable detector, generate synthetic YOLO data from verified still
images, then fine-tune a pretrained YOLO model:

```powershell
python -m tools.crab_generate_synthetic_dataset path\to\images --train-count 700 --val-count 180
python -m tools.crab_yolo_train Workspace\datasets\crab_green_yolo_YYYYMMDD_HHMMSS\data.yaml --epochs 50
```

Run the trained model on still photos with:

```powershell
python -m tools.crab_yolo_predict path\to\images --model Workspace\models\crab_yolo\run\weights\best.pt
```

If `--model` is omitted, the prediction helper uses the newest crab YOLO
weights under `Workspace\models\crab_yolo`.

## Measurement Workflows

The measurement applets load images or videos, let the operator choose a frame,
and collect clicked geometry. The pure measurement modules validate the clicked
points and raise clear errors when the geometry cannot support the calculation.
For the stereo segment applet, load the TritonPilot stereo session and matching
`stereo_calibration.json`, choose a clear frame, select the Generic Segment,
Iceberg Keel, or Coral Rig Length preset, click both endpoints in both
rectified views, and add several results so the reported median is less
sensitive to one shaky click. For horizontal segments, make sure the same
physical endpoint is first in both views; the app auto-corrects obvious right
endpoint reversals and also provides `Swap Right` when the pair order is
ambiguous.

For repeatability:

- Use the sharpest available frame.
- Keep the original media unchanged.
- Save screenshots or exported reports when the result will be used in a
  judge-facing answer.
- Record the reference lengths used for each measurement.

## RealityScan Reconstruction Workflow

Use the RealityScan reconstruction applet for long-running stereo
photogrammetry jobs after the stereo session has been copied off the pilot
station.

1. Select the `Pipeline Root` that contains the current
   `tools/realityscan_underwater_pipeline.py`; the applet auto-detects a
   sibling `TritonPilot` checkout when present.
2. Select the stereo session folder or its `manifest.json`.
3. Select the matching `stereo_calibration.json`; `Find Recent` searches recent
   TritonPilot stereo sessions.
4. Choose `max-detail` for final models, or a lighter preset for trial runs.
5. Keep metric stereo scaling enabled when the output will be measured.
6. Run the job and watch the live log/progress indicators.
7. Open the metric OBJ, contact sheet, report, or `metric_scale.json` from the
   output panel.

The default output workspace is a new timestamped subfolder under
`Workspace\results\realityscan` for each run. That keeps photogrammetry
artifacts with the analysis evidence trail instead of inside the live pilot UI,
and prevents a new reconstruction from overwriting the previous one.

Use the `Model Viewer` tab in the reconstruction applet, the `View Metric
Model` output button, or launch `main_realityscan_model_viewer` directly to
inspect the OBJ. The viewer starts a localhost server for the model directory
so Three.js can load the OBJ, MTL, and texture files together. With
`PyQt6-WebEngine` installed, the Three.js viewport is embedded in the Qt app;
otherwise the same viewer URL opens in the system browser. The viewport opens
with damped full-tumble orbit controls so tilted or inverted reconstructions can
be rotated fully around the target. `Set Floor` lets you pick three mesh points
and level that plane against the 3D reference grid. After the third point, the
camera snaps to a top-down `Floor View` and squares the screen direction to the
model footprint. `Level` removes camera roll from the current view, and `Reset
Floor` undoes the alignment. Use `Roll -` / `Roll +` and the roll-step selector
for manual camera roll when a model needs a custom viewing angle. The viewer
also includes `Fit`, `Reset`, preset camera views, a toggleable 3D grid, and
optional measurement labels.
Select `Measure`, click pairs of points on the mesh, and choose meters,
centimeters, or millimeters for the readout. Each completed line gets a small
in-viewport distance tag near its midpoint. Existing endpoints can be selected
and dragged while Measure is active; `Delete` removes the selected endpoint, or
the active measurement when no endpoint is selected. `Set Center` moves the
visible 3D cursor and orbit pivot to the next clicked mesh point. `Pick Assist`
is enabled by default in picking modes; it draws a bright marker on the exact
mesh hit and shows a magnified cursor loupe so light objects like PVC pipe ends
are easier to distinguish from the pool floor. Use `Edges` when extra cyan
surface boundary lines help separate objects from the background.

For post-submission ground truth, select the completed measurement that was used
as the guess, enter the true length and units in the ground-truth controls, and
click `Apply Scale`. The viewer applies a uniform scale factor to the model,
measurements, markers, grid, camera target, and saved view so follow-up
measurements use the corrected scale. Keep `underwater_model_metric.obj` as the
preferred input when you need real-world distances.

## Manual Data Workflows

The eDNA and iceberg tracking applets primarily use manually entered values.
Use them as calculation/report tools:

- Enter the task values carefully.
- Review the generated report text.
- Copy or export the result for team records.
- Keep any handwritten or source notes with the exported result.

## Color Correction And Frame Export

`color_corr.py` is for preprocessing underwater videos and exporting corrected
versions or selected frames. It is intentionally repeatable and conservative;
it is not meant to invent visual detail.

Use it when another applet or modeling workflow needs clearer saved media. Keep
both original and corrected files.

## Preserving Results

A useful results folder usually contains:

- Source media or a pointer to its original location
- Annotated images or selected frames
- CSV summaries
- Text reports copied from the applet
- Notes about reference lengths, clicked points, and operator assumptions
- Final judge-facing answer values

That folder becomes the evidence trail for technical documentation and
post-run review.
