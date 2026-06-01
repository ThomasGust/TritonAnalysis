# TritonPilot Transfer Link Quickstart

Use this when setting up a dedicated Ethernet link from TritonPilot to
TritonAnalysis on a Windows analysis computer.

## Where Synced Files Go

This quickstart syncs into a repo-local inbox:

```text
<TritonAnalysis repo>\Workspace\incoming\pilot
```

On this machine, the current synced files are in:

```text
C:\Users\Thoma\Documents\GitHub\TritonAnalysis\Workspace\incoming\pilot
```

Open it from PowerShell:

```powershell
explorer ".\Workspace\incoming\pilot"
```

## Expected Network

Use a dedicated USB Ethernet adapter, Ethernet cable, or small unmanaged switch
between the pilot computer and the analysis computer.

```text
Pilot analysis adapter    10.77.0.1/24
Analysis adapter          10.77.0.2/24
Gateway                   leave blank
DNS                       leave blank
TritonPilot transfer URL  http://10.77.0.1:8765
```

Keep the normal ROV tether on its existing `192.168.1.x` network.

## Setup Script

Run PowerShell as Administrator from the TritonAnalysis repository root.

First list the adapters:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\setup_pilot_transfer_link.ps1 -ListAdapters
```

Then configure the adapter Windows is using for the dedicated pilot link. On
this machine it was `Ethernet 3`; another computer may use a different name.

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\setup_pilot_transfer_link.ps1 -AdapterAlias "Ethernet 3"
```

The script sets the analysis adapter to `10.77.0.2/24`, clears gateway and DNS
for that adapter, marks the link Private, and tests `10.77.0.1:8765`.
By default it syncs into `.\Workspace\incoming\pilot` inside this checkout.
It also creates the repo-local `Workspace` folder structure, including
`results`, `reports`, and `calibrations`. That folder is ignored by git.

Preview the transfer:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\setup_pilot_transfer_link.ps1 -AdapterAlias "Ethernet 3" -DryRun
```

Configure and sync:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\setup_pilot_transfer_link.ps1 -AdapterAlias "Ethernet 3" -Sync
```

If the pilot computer uses a different address or port:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\setup_pilot_transfer_link.ps1 -AdapterAlias "Ethernet 3" -PilotAddress 10.77.0.1 -PilotPort 8765 -Sync
```

To use a different sync folder:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\setup_pilot_transfer_link.ps1 -AdapterAlias "Ethernet 3" -Output "D:\TritonAnalysisWorkspace\incoming\pilot" -Sync
```

Launch the unified GUI against the same repo-local workspace:

```powershell
python -m triton_analysis.apps.main_triton_analysis --workspace ".\Workspace"
```

## Pilot Side

Start the read-only Analysis Share from TritonPilot. The backup command from a
TritonPilot checkout is:

```powershell
python -m tools.analysis_transfer_server --root recordings --host 0.0.0.0 --port 8765
```

If Windows Firewall prompts on the pilot computer, allow Python on private
networks.

## Healthy Checks

A healthy analysis-side check looks like this:

```text
InterfaceAlias      Ethernet 3
IPv4Address         10.77.0.2
IPv4DefaultGateway  blank
DNS                 blank
TcpTestSucceeded    True
SourceAddress       10.77.0.2
```

After a current sync, a dry run should report:

```text
Would copy: 0 file(s), 0 byte(s)
Skipped current: <all advertised files>
```
