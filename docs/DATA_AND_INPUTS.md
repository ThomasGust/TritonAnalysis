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

## Workspace Layout

TritonAnalysis uses one machine-specific workspace root with stable folders
under it. The absolute root can differ on every laptop, but the structure below
it should stay the same.

Default root:

```text
%USERPROFILE%\Documents\TritonAnalysisWorkspace
```

Override it before launch when needed:

```powershell
$env:TRITON_ANALYSIS_WORKSPACE="D:\TritonAnalysisWorkspace"
```

Recommended structure:

```text
TritonAnalysisWorkspace/
  incoming/
    pilot/                auto-synced TritonPilot recordings
    usb/                  manual USB-drive drops, if used
  sources/                curated source media copied from incoming
    run_01/
    run_02/
  calibrations/           stereo_calibration.json and related artifacts
  results/
    crab_detection/
    coral_garden/
    realityscan/
    color_correction/
  reports/
    edna/
    iceberg_tracking/
  exports/                judge-ready bundles or final copied files
  scratch/                temporary experiments safe to delete
```

The unified app shows workspace-relative paths such as
`Workspace\incoming\pilot` in the status bar. The tooltip still contains the
full absolute path for Windows Explorer.

## Source Media Rules

For competition and technical documentation:

- Preserve original media unchanged.
- Keep corrected or annotated media in a separate results folder.
- Record which camera and run produced each file.
- Keep manual notes with the media they describe.
- Prefer descriptive names over screenshots with default names.

Suggested run layout under `sources/`:

```text
sources/
  run_01/
    primary_camera/
    arm_camera/
    stereo_sessions/
    notes.txt
  run_02/
```

When using the TritonPilot transfer link, `incoming/pilot/` is the first
landing folder. After each pull, copy or rename the relevant files into the
run-specific source folder you want to preserve.

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

Stereo measurements:

- TritonPilot stereo session folder or `manifest.json`
- Matching `stereo_calibration.json`
- Generic, Iceberg Keel, or Coral Rig Length preset
- Rectified left/right endpoint clicks

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

The batch crab-video helper writes to
`Workspace/results/crab_detection` by default unless `--output-dir` is
provided. Its typical outputs are:

- `best_frame.jpg`
- `best_annotated.jpg`
- `best_mask.png`
- `summary.csv`

Use `--output-dir` during competition so generated files land in the run's
results folder.
