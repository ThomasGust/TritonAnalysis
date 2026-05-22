# TritonAnalysis Testing And Troubleshooting

The test suite is hardware-free. Some computer-vision cases use local sample
recordings when present and skip themselves when those recordings are missing.

## Run The Quick Trust Check

From the TritonAnalysis repository root:

```powershell
.\.venv\Scripts\activate
python -m pytest
```

If `pytest` is not installed:

```powershell
python -m pip install -r requirements-dev.txt
```

`pytest.ini` sets `tests/` as the test root, quiet output, and strict marker
validation. The default suite is the fast trust check: it includes lightweight
logic, GUI-offscreen, synthetic, and bundled-sample tests, while skipping tests
marked `network`, `hardware`, `slow`, or `groundtruth`.

The helper script exposes the common tiers:

```powershell
python .\tools\trust_check.py quick
python .\tools\trust_check.py groundtruth
python .\tools\trust_check.py extended
python .\tools\trust_check.py full
```

Equivalent direct pytest commands:

| Goal | Command |
| --- | --- |
| Fast software-only check | `python -m pytest` |
| Saved CV recording/data regression tests | `python -m pytest --run-groundtruth -m groundtruth` |
| All non-hardware optional tiers | `python -m pytest --run-extended` |
| Physical hardware tests, if added | `python -m pytest --run-hardware -m hardware` |
| Everything | `python -m pytest --run-all-trust` |
| Coverage report, if `pytest-cov` is installed | `python .\tools\trust_check.py coverage` |

Environment variables work for CI or shell profiles:

- `TRITON_RUN_NETWORK=1`
- `TRITON_RUN_GROUNDTRUTH=1`
- `TRITON_RUN_SLOW=1`
- `TRITON_RUN_HARDWARE=1`

## Focused Tests

Run tests by applet area:

```powershell
python -m pytest tests\test_crab_detector.py
python -m pytest tests\test_iceberg_tracking.py
python -m pytest tests\test_iceberg_measurement.py
python -m pytest tests\test_planar_measurement.py
python -m pytest tests\test_coral_garden_model.py
python -m pytest tests\test_edna_analysis.py
python -m pytest tests\test_color_corr.py
python -m pytest tests\test_analysis_app_windows_responsive.py
```

`pytest.ini` defines the `vision` marker for computer-vision tests. Lightweight
vision tests with bundled or synthetic fixtures stay in the quick suite. CV tests
that need saved recordings or other larger local datasets should also be marked
`groundtruth` and usually `slow`.

## Test Marker Policy

Use the default suite for deterministic tests that can run on any developer
machine. Mark new tests when they leave that boundary:

- `vision`: exercises image/video analysis behavior.
- `groundtruth`: depends on optional saved media or datasets outside the normal
  repository fixtures.
- `slow`: intentionally takes long enough that it should not block quick checks.
- `network`: opens sockets or depends on active networking.
- `hardware`: touches physical hardware or live services.
- `integration`: crosses applet/module boundaries but remains deterministic and
  hardware-free.

## What The Tests Cover

- Crab detector sample counts and video selection behavior
- eDNA frequency math, validation, GUI count updates, and reports
- Iceberg coordinate conversion, threat thresholds, survey validation, and
  report formatting
- Iceberg measurement geometry and error handling
- Planar measurement homography and reference constraints
- Coral garden model dimensions and OBJ export
- Color-correction masks, frame export timing, and target-processing behavior
- GUI windows fitting available screens and exposing scrollable controls

## GUI Does Not Open

- Confirm the virtual environment is active.
- Confirm PyQt6 is installed in that environment.
- Confirm the machine has a desktop session available.
- On Linux, confirm the Qt platform plugin dependencies are installed.

## Image Or Video Will Not Load

- Confirm the file path exists.
- Confirm the suffix is supported by the applet.
- Confirm OpenCV can read the codec on this machine.
- Try converting the source to a standard `.mp4` or `.jpg`.
- Keep the original file even if a converted copy is used for analysis.

## Crab Detection Looks Wrong

- Try a sharper frame.
- Use manual board-corner selection when automatic board detection fails.
- Check whether the board is occluded, cropped, or heavily distorted.
- Compare the annotated output to the original before trusting the count.

## Measurement Applet Reports Geometry Errors

- Re-check point order and reference lengths.
- Avoid nearly overlapping clicked points.
- Use reference segments that constrain both axes when the applet requires it.
- Choose a frame with less motion blur and clearer corners.

## Color Correction Is Too Aggressive

- Reduce white balance, red restore, sharpening, PVC boost, or target boost.
- Turn off diagnostic mask drawing for final exports.
- Keep corrected media separate from originals.

## Debugging Order

When an applet fails:

1. Run the relevant focused test file.
2. Reproduce with a small sample image or video.
3. Confirm the GUI can load the file.
4. Confirm the core module handles the same inputs.
5. Add or update a test for the failing case before changing formulas.
