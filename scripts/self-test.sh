#!/usr/bin/env bash
# Cluster self-test: /health → /v1/models → completion → reasoning → tools → DFlash warmup → 2nd completion.
# First request may be Triton/JIT warmup — exclude from benchmarks.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

echo "[self-test] scaffold — wire to OpenAI-compatible endpoints after up.sh"
# TODO: implement ordered checks listed in README
exit 1
