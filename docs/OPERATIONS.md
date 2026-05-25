# TritonAnalysis Operations Guide

This guide describes how to use TritonAnalysis during development, practice,
and competition. The applets are independent tools, so operators should launch
only the one needed for the current task.

## Competition-Day Pattern

1. TritonPilot records source media or the team writes down task values.
2. The analysis operator copies the source media or values to the analysis
   computer.
3. The operator launches the relevant TritonAnalysis applet.
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
- Confirm sample/reference data under `data/` is present.
- Make a folder for incoming competition media.
- Practice the exact applet workflow with representative images and videos.

## Launching Applets

From the TritonAnalysis repository root:

```powershell
.\.venv\Scripts\activate
```

Crab detection:

```powershell
python -m main_crab_detection path\to\images_or_video
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

Use `main_crab_detection.py` for interactive review. It accepts image files,
folders, and one video file. The GUI can detect board corners automatically or
use manually picked corners when automatic board detection is unreliable.

For batch video processing:

```powershell
python -m tools.crab_video_detect path\to\video.mp4 --output-dir path\to\results
```

The batch helper saves a best frame, annotated frame, mask, and CSV summary.

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
