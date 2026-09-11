#!/usr/bin/env bash
# Follow glm53 logs on head (default) or WORKER=1 for worker.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if [[ -f .env ]]; then
  # shellcheck disable=SC1091
  set -a && source .env && set +a
fi

SSH_USER="${SSH_USER:-${USER}}"
TARGET="${HEAD_HOST:?Set HEAD_HOST}"
[[ "${WORKER:-0}" == "1" ]] && TARGET="${WORKER_HOST:?Set WORKER_HOST}"

exec ssh -t "${SSH_USER}@${TARGET}" "docker logs -f --tail=200 glm53 ${*:+$*}"
