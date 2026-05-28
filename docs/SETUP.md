# TritonAnalysis Setup Guide

TritonAnalysis is a standalone Python application set. It does not require the
ROV, a controller, ZeroMQ, or GStreamer. Most applets use PyQt6 for their GUI,
OpenCV/NumPy for image work, and small pure-Python modules for task logic.

## Required Software

- Python 3.10 or newer
- `pip`
- A virtual environment
- A desktop session capable of opening PyQt6 windows

The base dependency list is in `requirements.txt`:

- `PyQt6`
- `matplotlib`
- `numpy`
- `opencv-python`
- `scipy`

`requirements-windows.txt` and `requirements-macos.txt` currently include the
same base file.

## Windows Setup

From the TritonAnalysis repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run tests:

```powershell
python -m pip install pytest
python -m pytest
```

Launch an applet:

```powershell
python -m main_crab_detection
```

## macOS And Linux Setup

From the TritonAnalysis repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run tests:

```sh
python -m pip install pytest
python -m pytest
```

Launch an applet:

```sh
python -m main_iceberg_tracking
```

## Applet Smoke Tests

These commands should at least open the GUI on a properly configured desktop:

```powershell
python -m main_crab_detection
python -m main_iceberg_tracking
python -m main_coral_garden_model
python -m main_edna_analysis --sample
python -m main_iceberg_measurement
python -m main_planar_height_measurement
python -m main_multi_rect_length_measurement
python -m color_corr
```

Close each window before launching the next one if the machine is resource
constrained.

## Optional Local Data

The repository includes small reference/sample assets under `data/`, including
crab sample and reference images. Larger videos used by some computer-vision
tests are intentionally not required; those tests skip themselves when the
recordings are absent.

For competition use, create a local workspace folder outside the repo for
incoming media, reports, and generated results:

```powershell
mkdir "$env:USERPROFILE\Documents\TritonAnalysisWorkspace"
```

Keep original captures unchanged and write applet outputs to a separate results
folder inside that workspace.

## No Live ROV Dependencies

Do not install TritonPilot-only or TritonOS-only runtime dependencies here
unless a specific analysis feature needs them. TritonAnalysis should remain
easy to run on any laptop that can open PyQt6 and process saved files.
