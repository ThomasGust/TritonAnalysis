# Stereo Calibration

TritonAnalysis calibrates stereo rigs from saved TritonPilot stereo sessions.
It does not talk to the ROV or open live streams.

## Inputs

Copy the full TritonPilot stereo session folder to the analysis computer:

```text
stereo_sessions/
  20260522-153012/
    left/
    right/
    manifest.json
```

The manifest supplies the left/right image paths and timing metadata. The GUI
can also load multiple `manifest.json` files or scan a folder of stereo
sessions, as long as they all belong to the same stereo pair and mount. Board
dimensions still come from the operator because they must match the physical
calibration board used in the pool.

## GUI Workflow

Launch the stereo calibration applet:

```powershell
python -m triton_analysis.apps.main_stereo_calibration_gui path\to\manifest.json
```

The window lets you inspect one manifest or a combined set of manifests,
preview left/right image pairs, choose checkerboard or ChArUco board settings,
set minimum accepted pairs, run calibration, review rejected observations, and
write the calibration artifact.
Set `Use First Frames` below 100% when you want a faster calibration pass from
the first portion of a large capture session.
During calibration, the progress bar reports manifest loading, per-pair board
detection counts, solve stages, and artifact writing. OpenCV does not expose
inner iteration progress for the solve calls, so the bar switches to an
animated working state while those blocking steps run.
The default board settings match Triton's pool ChArUco board:
9 rows by 12 columns, 6 cm square width, 4.5 cm marker width, and the
calib.io default `DICT_5X5_1000` dictionary.
Checkerboard-only fields are hidden when ChArUco mode is selected.
Selecting a pair overlays detected board markers/corners on both previews and
shows left, right, and matched detection counts for quick pool-side review.

## Checkerboard Calibration

The command-line path is still useful for repeatable batch runs. For a
checkerboard, `--columns` and `--rows` are inner-corner counts:

```powershell
python -m triton_analysis.apps.main_stereo_calibration path\to\manifest.json `
  --checkerboard `
  --columns 9 `
  --rows 6 `
  --square-size 2.5 `
  --units cm `
  --rig-id explorehd_forward_v1 `
  --pair-name "Forward Stereo"
```

## ChArUco Calibration

For a ChArUco board, `--squares-x` and `--squares-y` are full square counts.
The defaults match Triton's board, so the normal command is:

```powershell
python -m triton_analysis.apps.main_stereo_calibration path\to\manifest.json `
  --charuco
```

The CLI and GUI default to at least 24 matched ChArUco corners per stereo pair.
That keeps tiny edge-only detections out of the final solve while still
accepting useful partial-board views.

You can also pass several manifests or a parent folder:

```powershell
python -m triton_analysis.apps.main_stereo_calibration path\to\stereo_sessions --charuco
python -m triton_analysis.apps.main_stereo_calibration path\to\session1\manifest.json path\to\session2\manifest.json --charuco
python -m triton_analysis.apps.main_stereo_calibration path\to\stereo_sessions --charuco --frame-percent 25
```

Equivalent explicit values:

```powershell
python -m triton_analysis.apps.main_stereo_calibration path\to\manifest.json `
  --charuco `
  --squares-x 12 `
  --squares-y 9 `
  --square-size 6 `
  --marker-size 4.5 `
  --dictionary DICT_5X5_1000 `
  --units cm
```

ChArUco is preferred when the board is partially visible or when the fisheye
field of view makes full checkerboard detection unreliable.

## Output Artifact

The default GUI output is under `Workspace/calibrations`, named from the stereo
session when possible. It stores:

- Left and right intrinsic matrices and distortion coefficients.
- Stereo rotation and translation.
- Essential and fundamental matrices.
- Rectification matrices and the `Q` reprojection matrix.
- RMS error, accepted observation count, rejected pair notes, rig id, pair
  name, image size, board metadata, and units.
- Quality diagnostics: per-camera reprojection RMS, rectified vertical
  epipolar RMS, raw distorted-pixel fundamental-matrix residuals, left/right
  image coverage, accepted pair metadata, and warning messages.

The artifact is the handoff point for the stereo depth applet:

```powershell
python -m triton_analysis.apps.main_stereo_depth path\to\manifest.json
python -m triton_analysis.apps.main_stereo_segment_measurement path\to\manifest.json
python -m triton_analysis.apps.main_stereo_iceberg_measurement path\to\manifest.json
```

If `stereo_calibration.json` lives next to the manifest, the depth applet loads
it automatically. Otherwise pass the workspace artifact explicitly with
`--calibration`.
For pool-scale work, keep the maximum depth cap and left/right consistency
check enabled in the depth applet; they prevent low-texture, near-zero-disparity
speckles from turning into hundreds-of-meters depth samples. For PVC endpoints,
arm spans, and other low-texture objects, prefer clicking the same endpoint in
both rectified previews so TritonAnalysis triangulates the correspondence
directly instead of sampling the dense disparity map. For task-focused
straight-line measurements, use `main_stereo_segment_measurement`; it offers
Generic Segment, Iceberg Keel, and Coral Rig Length presets plus
repeated-measurement median/spread output. `main_stereo_iceberg_measurement`
remains as a shortcut that opens the same tool in Iceberg Keel mode.

## Quality Targets

- Use final underwater calibration captures, not air captures, for competition
  measurements.
- Keep the board flat, known, and rigid.
- Accept many poses across the entire image, not only centered board shots.
- Re-run calibration whenever the stereo mount moves.
- Review rejected observations and RMS error before trusting any measurement.

## Pool-Test Acceptance Checks

The GUI, CLI, and stereo depth applet report enough calibration quality
information to make a pool-side decision:

- Accepted pairs: 20 is the minimum practical target; 30 to 60 varied accepted
  pairs is better.
- Stereo RMS: aim below 1 px. Values above 1 px are a reason to inspect frame
  sync, motion blur, lighting, board flatness, and board coverage.
- Rectified epipolar RMS: aim below 1 px. This checks that matched left/right
  points land on the same scanline after undistortion and rectification.
- Reprojection RMS: each camera should stay low and similar. One bad side often
  points to focus, exposure, blur, or board detection problems on that camera.
- Coverage: get ChArUco corners into the center, all four sides, and all four
  corners of both images. Center-only captures can produce a deceptively clean
  RMS while still giving poor measurements away from the center.
- Baseline: compare the artifact baseline against the measured physical camera
  spacing in the same units as the board. A large mismatch usually means board
  dimensions, units, or camera ordering are wrong.
- Rejections: a few rejected frames are normal. Many rejections usually mean
  glare, blur, too much partial occlusion, or a pose where one camera sees too
  little of the board.

For Sunday's pool test, save several capture sessions rather than overwriting
one. Calibrate each candidate, keep the artifact with the lowest errors and
best coverage, and preserve the raw session folder that produced it.
