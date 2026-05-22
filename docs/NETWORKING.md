# TritonAnalysis Network And Data Handoff Guide

TritonAnalysis does not require a live network connection to the ROV. It should
not publish commands, subscribe to telemetry, open camera streams, or call
TritonOS RPC services during competition.

## Runtime Network Requirements

There are no required network ports for normal TritonAnalysis operation.

The applets consume:

- Saved image files
- Saved video files
- Manually entered task values
- Bundled reference data under `data/`

This is deliberate. The pilot station should remain focused on vehicle control,
and the analysis station should be able to keep working even if the tether,
ROV, or pilot UI is unavailable.

## Data Handoff From TritonPilot

The normal handoff is:

```text
TritonPilot records media/data
        |
        v
Files are copied or moved to the analysis computer
        |
        v
TritonAnalysis applets load saved files or manual values
        |
        v
Results are exported for the team/judges
```

Use clear folder names such as:

```text
competition_media/
  run_01/
    primary_camera/
    arm_camera/
    raw_sensor_logs/
    notes.txt
  run_01_results/
```

Preserve original captures. If an applet exports corrected images, annotated
frames, OBJ files, CSV files, or judge reports, place them in a results folder
rather than overwriting source media.

## Transfer Methods

Any ordinary file-transfer method is acceptable:

- USB drive
- Shared folder
- Local network share
- Cloud sync after the run

Choose the method that is least likely to interrupt the pilot computer. Avoid
large live copies while TritonPilot is actively recording if the transfer could
stress the disk or network.

## Why This Boundary Matters

Keeping TritonAnalysis network-free:

- Makes the applets easier to test on normal development machines.
- Lets analysis continue if the ROV is offline.
- Prevents long-running computer-vision work from affecting piloting.
- Keeps competition documentation honest about the three-part system split:
  TritonOS onboard, TritonPilot topside, and TritonAnalysis standalone.

If a future feature truly needs live data, document it as an exception and
consider whether it belongs in TritonPilot or TritonOS instead.
