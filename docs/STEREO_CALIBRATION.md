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
python -m main_stereo_calibration_gui path\to\manifest.json
```

The window lets you inspect one manifest or a combined set of manifests,
preview left/right image pairs, choose checkerboard or ChArUco board settings,
set minimum accepted pairs, run calibration, review rejected observations, and
write the calibration artifact.
The default board settings match Triton's shipped ChArUco board:
17 rows by 24 columns, 30 mm square width, and 22 mm marker width.
Checkerboard-only fields are hidden when ChArUco mode is selected.

## Checkerboard Calibration

The command-line path is still useful for repeatable batch runs. For a
checkerboard, `--columns` and `--rows` are inner-corner counts:

```powershell
python -m main_stereo_calibration path\to\manifest.json `
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
python -m main_stereo_calibration path\to\manifest.json `
  --charuco
```

You can also pass several manifests or a parent folder:

```powershell
python -m main_stereo_calibration path\to\stereo_sessions --charuco
python -m main_stereo_calibration path\to\session1\manifest.json path\to\session2\manifest.json --charuco
```

Equivalent explicit values:

```powershell
python -m main_stereo_calibration path\to\manifest.json `
  --charuco `
  --squares-x 24 `
  --squares-y 17 `
  --square-size 30 `
  --marker-size 22 `
  --dictionary DICT_4X4_50 `
  --units mm
```

ChArUco is preferred when the board is partially visible or when the fisheye
field of view makes full checkerboard detection unreliable.

## Output Artifact

The default output is `stereo_calibration.json` next to the manifest, or next
to the first selected source when several manifests are loaded. It stores:

- Left and right intrinsic matrices and distortion coefficients.
- Stereo rotation and translation.
- Essential and fundamental matrices.
- Rectification matrices and the `Q` reprojection matrix.
- RMS error, accepted observation count, rejected pair notes, rig id, pair
  name, image size, board metadata, and units.
- Quality diagnostics: per-camera reprojection RMS, symmetric epipolar RMS,
  left/right image coverage, accepted pair metadata, and warning messages.

The artifact is the handoff point for future rectification, disparity, 3D
measurement, and coral-garden modeling tools.

## Quality Targets

- Use final underwater calibration captures, not air captures, for competition
  measurements.
- Keep the board flat, known, and rigid.
- Accept many poses across the entire image, not only centered board shots.
- Re-run calibration whenever the stereo mount moves.
- Review rejected observations and RMS error before trusting any measurement.

## Pool-Test Acceptance Checks

The GUI and CLI now report enough calibration quality information to make a
pool-side decision before disparity or depth tools exist:

- Accepted pairs: 20 is the minimum practical target; 30 to 60 varied accepted
  pairs is better.
- Stereo RMS: aim below 1 px. Values above 1 px are a reason to inspect frame
  sync, motion blur, lighting, board flatness, and board coverage.
- Epipolar RMS: aim below 1 px. This is the most direct check that matching
  left/right points agree with the stereo geometry.
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
