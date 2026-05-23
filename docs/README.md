# TritonAnalysis Documentation

This folder is the maintained documentation set for TritonAnalysis. It is meant
to be the primary reference for operators and maintainers working with the
competition analysis applets.

## Guides

- [Setup Guide](SETUP.md) - Create the Python environment and run the applets
  on Windows, macOS, or Linux.
- [Network And Data Handoff Guide](NETWORKING.md) - Clarifies that
  TritonAnalysis has no live ROV network dependency and describes how data
  should move from pilot station to analysis station.
- [Operations Guide](OPERATIONS.md) - Competition-day workflow for launching
  applets, loading media, entering task data, exporting results, and preserving
  evidence.
- [Architecture Overview](ARCHITECTURE.md) - How entry points, GUI wrappers,
  pure analysis modules, and tests fit together.
- [Applet Reference](APPLET_REFERENCE.md) - Purpose, inputs, outputs, and
  important implementation modules for each applet.
- [Subsystem Reference](SUBSYSTEMS.md) - Repository package/module ownership.
- [Data And Inputs](DATA_AND_INPUTS.md) - Supported file types, bundled
  reference data, output locations, and handoff conventions.
- [Stereo Calibration](STEREO_CALIBRATION.md) - Calibrating stereo rigs from
  TritonPilot stereo capture sessions and writing calibration artifacts.
- [Testing And Troubleshooting](TESTING_AND_TROUBLESHOOTING.md) - Test
  commands, optional vision-test behavior, and common setup/runtime problems.

## Related Repositories

- `TritonOS` runs onboard the ROV and owns hardware control.
- `TritonPilot` runs on the pilot computer and owns live control, telemetry,
  video display, and recording.

TritonAnalysis should consume artifacts from those systems. It should not
become a live-control or live-network dependency.

## Documentation Style

When updating these docs:

- Write commands relative to the TritonAnalysis repository root.
- State whether an applet expects media files, manual values, or both.
- Keep judge-facing formulas and assumptions near the applet that uses them.
- Link to repository paths with relative paths.
- Update this index when adding, renaming, or removing a maintained guide.
