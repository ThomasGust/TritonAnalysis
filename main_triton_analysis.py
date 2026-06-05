"""Top-level launcher for the unified TritonAnalysis GUI."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _qt_python_candidates() -> list[Path]:
    repo_root = Path(__file__).resolve().parent
    candidates = [
        repo_root / ".venv" / "Scripts" / "python.exe",
        repo_root / ".venv" / "bin" / "python",
        repo_root.parent / "TritonPilot" / ".venv" / "Scripts" / "python.exe",
        repo_root.parent / "TritonPilot" / ".venv" / "bin" / "python",
    ]
    return [candidate for candidate in candidates if candidate.exists()]


def _reexec_with_qt_python() -> None:
    if os.environ.get("TRITON_ANALYSIS_NO_ENV_REEXEC", "").strip():
        return
    try:
        current = Path(sys.executable).resolve()
    except OSError:
        current = Path(sys.executable)
    for candidate in _qt_python_candidates():
        try:
            resolved = candidate.resolve()
        except OSError:
            resolved = candidate
        if resolved == current:
            continue
        os.environ["TRITON_ANALYSIS_REEXECED"] = "1"
        os.execv(str(resolved), [str(resolved), str(Path(__file__).resolve()), *sys.argv[1:]])


try:
    from triton_analysis.apps.main_triton_analysis import main
except ModuleNotFoundError as exc:
    if exc.name != "PyQt6":
        raise
    _reexec_with_qt_python()
    raise RuntimeError(
        "PyQt6 is not installed in this Python. Use .venv\\Scripts\\python.exe "
        "from the TritonAnalysis checkout, or install requirements-windows.txt."
    ) from exc


if __name__ == "__main__":
    sys.exit(main())
