# TritonAnalysis Data And Inputs

TritonAnalysis works from saved files and manually entered values. This page
summarizes expected inputs and output conventions.

## Supported Media

Most image applets accept:

- `.jpg`
- `.jpeg`
- `.png`
- `.bmp`
- `.tif`
- `.tiff`
- `.webp`

Most video applets accept:

- `.mp4`
- `.mov`
- `.m4v`
- `.avi`
- `.mkv`
- `.wmv`

Exact support depends on OpenCV's local codec support. If a file will not open,
try converting it to `.mp4` with a common H.264 encoding before competition.

## Bundled Data

The `data/` folder contains small assets used by applets and tests:

```text
data/crab_samples/
data/crab_reference/
```

These assets are appropriate to keep in the repository. Large raw videos,
competition runs, and generated result folders should remain outside git.

## Source Media Rules

For competition and technical documentation:

- Preserve original media unchanged.
- Keep corrected or annotated media in a separate results folder.
- Record which camera and run produced each file.
- Keep manual notes with the media they describe.
- Prefer descriptive names over screenshots with default names.

Suggested layout:

```text
TritonCompetitionMedia/
  run_01_source/
  run_01_results/
  run_02_source/
  run_02_results/
```

## Applet Inputs

Crab detection:

- Source image, folder, or video
- Optional manual board-corner clicks

Iceberg tracking:

- Iceberg coordinates
- Heading
- Keel depth
- Survey numbers

Iceberg and planar measurements:

- Source image or video
- Representative frame selection
- Clicked calibration/measurement points
- Known reference lengths

Coral garden:

- Length, height, and optional width

eDNA:

- Species counts

Color correction:

- Source video
- Processing settings
- Frame-selection settings

## Outputs

Depending on the applet, outputs may include:

- Annotated images
- Best-frame images
- Masks
- CSV summaries
- Text reports
- Corrected videos
- Exported frame sets
- OBJ model text
- PNG model previews

For judge-facing results, keep exported artifacts and the source media together
so a teammate can reproduce or explain the answer later.

## Results From `tools.crab_video_detect`

The batch crab-video helper writes to `results/crab_video_detection` by default
unless `--output-dir` is provided. Its typical outputs are:

- `best_frame.jpg`
- `best_annotated.jpg`
- `best_mask.png`
- `summary.csv`

Use `--output-dir` during competition so generated files land in the run's
results folder.
