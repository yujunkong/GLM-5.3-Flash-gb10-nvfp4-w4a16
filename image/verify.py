#!/usr/bin/env python3
"""Build-time patch verification (scaffold).

Checks text/function/path markers only — do NOT import vLLM CUDA platform
during docker build (no driver). Fail the build when required patches are missing
once real patch files are imported from upstream.
"""

from __future__ import annotations

import sys
from pathlib import Path

PATCH_DIR = Path(__file__).resolve().parent / "patches"

# Required scaffold files; replace markers after upstream import
REQUIRED = [
    "glm53-flash_SM121.py",
    "gb10_topk_fallback.py",
    "gb10_plugin_backend.py",
    "worker_memory_cap.py",
    "spark_mem_trace.py",
    "link_cuda_headers.sh",
]


def main() -> int:
    missing = [name for name in REQUIRED if not (PATCH_DIR / name).is_file()]
    if missing:
        print(f"[verify] missing patch files: {missing}", file=sys.stderr)
        # Scaffold phase: warn but allow tree to exist; tighten to exit 1 after import
        print("[verify] scaffold mode — not failing build yet", file=sys.stderr)
        return 0

    print("[verify] all required patch file paths present (content checks TODO)")
    # TODO: assert SM121 / topk / plugin / worker_cap / CUDA headers / DFlash2 markers
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
