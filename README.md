# TritonAnalysis

Standalone mission-analysis tools for Triton's competition workflow.

This repository is intentionally separate from `TritonPilot`. It does not talk
to the ROV, publish pilot commands, subscribe to live telemetry, or depend on
the topside control UI. During competition, run these scripts on the analysis
laptop against saved images, video files, or manually entered task data.

## Setup

```sh
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On macOS or Linux:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Applets

Crab competition analyzer:

```sh
python -m main_crab_detection [image-folder-or-video ...]
```

Iceberg tracking threat applet:

```sh
python -m main_iceberg_tracking
```

Coral garden CAD model applet:

```sh
python -m main_coral_garden_model
```

eDNA frequency analysis applet:

```sh
python -m main_edna_analysis
```

Iceberg measurement applet:

```sh
python -m main_iceberg_measurement
```

Planar height measurement applet:

```sh
python -m main_planar_height_measurement
```

Multi-rectangle length measurement applet:

```sh
python -m main_multi_rect_length_measurement
```

Underwater color correction / frame export applet:

```sh
python -m color_corr
```

Crab video batch helper:

```sh
python -m tools.crab_video_detect path\to\video.mp4
```

## Tests

```sh
python -m pip install pytest
python -m pytest
```

Computer-vision tests that need larger local recordings are marked `vision`.
They skip themselves when those recordings are not present.
