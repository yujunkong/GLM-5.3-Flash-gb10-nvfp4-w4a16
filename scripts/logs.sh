#!/usr/bin/env bash
# Follow or dump cluster logs (scaffold).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

echo "[logs] scaffold — wire to docker compose logs for head/worker"
# TODO: docker compose -f compose/glm53.yaml logs -f "$@"
exit 1
