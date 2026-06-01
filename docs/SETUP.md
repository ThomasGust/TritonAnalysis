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
- `ultralytics` for optional YOLO crab fine tuning and inference

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

The repository may include small reference/sample assets under `data/` when an
applet needs them. Larger videos used by computer-vision tests are intentionally
not required; those tests skip themselves when the recordings are absent.
For crab detection, either keep the known TritonPilot reference recording
available as a sibling checkout or set:

```powershell
$env:TRITON_CRAB_REFERENCE_IMAGE="D:\path\to\crab_board_reference.png"
```

For competition use, TritonAnalysis creates a local ignored `Workspace` folder
inside the repo for incoming media, reports, and generated results:

```powershell
python -c "from analysis_workspace import workspace_paths; workspace_paths(create=True)"
```

Keep original captures unchanged and write applet outputs to a separate results
folder inside that workspace.

YOLO crab training uses Ultralytics on top of PyTorch. If PyTorch can use the
local GPU, `tools.crab_yolo_train` will select it; otherwise it falls back to
CPU. On newer GPUs, install a PyTorch build that explicitly supports that GPU
architecture before starting long training runs.

## No Live ROV Dependencies

Do not install TritonPilot-only or TritonOS-only runtime dependencies here
unless a specific analysis feature needs them. TritonAnalysis should remain
easy to run on any laptop that can open PyQt6 and process saved files.
