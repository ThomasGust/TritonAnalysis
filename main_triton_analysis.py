"""Top-level launcher for the unified TritonAnalysis GUI."""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _hydrate_user_environment(names: tuple[str, ...]) -> None:
    """Load selected Windows user env vars when the parent process is stale."""

    if os.name != "nt":
        return
    try:
        import winreg
    except ImportError:
        return
    try:
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment")
    except OSError:
        return
    with key:
        for name in names:
            if os.environ.get(name):
                continue
            try:
                value, _value_type = winreg.QueryValueEx(key, name)
            except OSError:
                continue
            if isinstance(value, str) and value.strip():
                os.environ[name] = value


def _project_python_candidates() -> list[Path]:
    repo_root = Path(__file__).resolve().parent
    candidates = [
        repo_root / ".venv" / "Scripts" / "python.exe",
        repo_root / ".venv" / "bin" / "python",
        repo_root.parent / "TritonPilot" / ".venv" / "Scripts" / "python.exe",
        repo_root.parent / "TritonPilot" / ".venv" / "bin" / "python",
    ]
    return [candidate for candidate in candidates if candidate.exists()]


def _reexec_with_project_python() -> None:
    if os.environ.get("TRITON_ANALYSIS_NO_ENV_REEXEC", "").strip():
        return
    try:
        current = Path(sys.executable).resolve()
    except OSError:
        current = Path(sys.executable)
    resolved_candidates: list[Path] = []
    for candidate in _project_python_candidates():
        try:
            resolved_candidates.append(candidate.resolve())
        except OSError:
            resolved_candidates.append(candidate)
    if any(candidate == current for candidate in resolved_candidates):
        return
    if resolved_candidates:
        python = resolved_candidates[0]
        os.environ["TRITON_ANALYSIS_REEXECED"] = "1"
        os.execv(str(python), [str(python), str(Path(__file__).resolve()), *sys.argv[1:]])


_hydrate_user_environment(("OPENAI_API_KEY",))
_reexec_with_project_python()

try:
    from triton_analysis.apps.main_triton_analysis import main
except ModuleNotFoundError as exc:
    if exc.name != "PyQt6":
        raise
    _reexec_with_project_python()
    raise RuntimeError(
        "PyQt6 is not installed in this Python. Use .venv\\Scripts\\python.exe "
        "from the TritonAnalysis checkout, or install requirements-windows.txt."
    ) from exc


if __name__ == "__main__":
    sys.exit(main())
