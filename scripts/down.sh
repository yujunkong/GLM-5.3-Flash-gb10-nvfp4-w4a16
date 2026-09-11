#!/usr/bin/env bash
# Tear down cluster containers/processes. Never delete Hugging Face cache.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

echo "[down] scaffold — stop compose/services for this recipe"
# TODO: docker compose -f compose/glm53.yaml ... down
# Explicit: do NOT rm -rf ~/.cache/huggingface or container HF mounts
echo "[down] HF cache left intact (by design)"
echo "[down] NOT IMPLEMENTED"
exit 1
