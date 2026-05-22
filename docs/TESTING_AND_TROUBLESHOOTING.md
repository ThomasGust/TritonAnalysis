# TritonAnalysis Testing And Troubleshooting

The test suite is hardware-free. Some computer-vision cases use local sample
recordings when present and skip themselves when those recordings are missing.

## Run The Full Test Suite

From the TritonAnalysis repository root:

```powershell
.\.venv\Scripts\activate
python -m pytest
```

If `pytest` is not installed:

```powershell
python -m pip install pytest
```

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

`pytest.ini` defines the `vision` marker for CV tests that may depend on larger
local recordings.

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
