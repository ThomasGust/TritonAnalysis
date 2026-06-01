"""Top-level launcher for the unified TritonAnalysis GUI."""

from __future__ import annotations

import sys

from triton_analysis.apps.main_triton_analysis import main


if __name__ == "__main__":
    sys.exit(main())
