# TritonAnalysis Network And Data Handoff Guide

TritonAnalysis does not require a live network connection to the ROV. It should
not publish commands, subscribe to telemetry, open camera streams, or call
TritonOS RPC services during competition. When an Ethernet handoff is useful,
TritonAnalysis should pull saved files from TritonPilot instead of becoming a
live-control dependency.

## Runtime Network Requirements

There are no required network ports for normal TritonAnalysis operation.

The applets consume:

- Saved image files
- Saved video files
- Manually entered task values
- Bundled reference data under `data/`

This is deliberate. The pilot station should remain focused on vehicle control,
and the analysis station should be able to keep working even if the tether,
ROV, or pilot UI is unavailable. The optional transfer helper only copies saved
files from TritonPilot; it does not talk to TritonOS or control the ROV.

## Optional USB-Ethernet Handoff

Use a dedicated USB-to-Ethernet adapter pair or a small unmanaged switch between
the pilot and analysis laptops:

```text
Pilot analysis adapter    10.77.0.1/24
Analysis adapter          10.77.0.2/24
Gateway                   leave blank
DNS                       leave blank
```

Keep the normal ROV tether on its existing `192.168.1.x` network.

On the pilot computer, start the read-only transfer server from the
TritonPilot app. The pilot status bar shows `Analysis Share` with the served
folder, URL, file count, active file sends, and the last Analysis pull time.
The backup CLI command is:

```powershell
python -m tools.analysis_transfer_server --root recordings --host 0.0.0.0 --port 8765
```

If Windows Firewall prompts on the pilot computer, allow Python on private
networks for the dedicated analysis link.

On the analysis computer, the unified TritonAnalysis app pulls saved files into
`Workspace\incoming\pilot` automatically. Its top `Pilot Sync` panel shows the
URL, connection state, exact destination folder, and whether it is checking,
listening for new files, receiving, or done receiving files. Current TritonPilot servers expose a
long-poll event endpoint, and TritonAnalysis uses it by default so new stable
Pilot files sync immediately instead of waiting for the next periodic poll.
Older Pilot servers still work; Analysis falls back to periodic index checks.
The backup CLI command is:

```powershell
python -m tools.pilot_transfer_sync http://10.77.0.1:8765 --output ".\Workspace\incoming\pilot"
```

Preview what would copy without writing files:

```powershell
python -m tools.pilot_transfer_sync http://10.77.0.1:8765 --output ".\Workspace\incoming\pilot" --dry-run
```

The sync preserves the folder layout advertised by TritonPilot. Files that
already match by size and modification time are skipped.

If you delete files from `incoming\pilot`, the next manual or automatic sync
should receive those missing files again. If they do not appear where expected,
check the `Receiving to:` line in the `Pilot Sync` panel first; the app may be
using a different workspace root or custom sync folder than the one you cleared.

Useful sync timing overrides:

```powershell
$env:TRITON_ANALYSIS_SYNC_INTERVAL_S="1.0"
$env:TRITON_ANALYSIS_SYNC_WATCH="1"
$env:TRITON_ANALYSIS_SYNC_EVENT_TIMEOUT_S="20.0"
```

Inside the unified TritonAnalysis app, use the `Workspace` menu to choose the
workspace root, and the `Pilot Sync` menu to toggle auto sync, sync now, set a
custom destination folder, or change the Pilot URL.

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

Use clear workspace folders such as:

```text
Workspace/
  incoming/pilot/
  sources/run_01/
    primary_camera/
    arm_camera/
    raw_sensor_logs/
    notes.txt
  results/
  reports/
```

Preserve original captures. If an applet exports corrected images, annotated
frames, OBJ files, CSV files, or judge reports, place them in a results folder
rather than overwriting source media.

## Transfer Methods

Any ordinary file-transfer method is acceptable:

- USB drive
- Dedicated TritonPilot transfer link
- Shared folder or local network share
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
