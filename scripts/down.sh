#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
[[ -f .env ]] && set -a && source .env && set +a
SSH_USER="${SSH_USER:-$USER}"
HEAD_HOST="${HEAD_HOST:?}"
WORKER_HOST="${WORKER_HOST:?}"
REMOTE_ROOT="${REMOTE_ROOT:-$ROOT}"
echo "[down] stopping glm53 (HF cache untouched)"
docker rm -f glm53 2>/dev/null || true
ssh -o BatchMode=yes "${SSH_USER}@${WORKER_HOST}" "docker rm -f glm53 2>/dev/null || true"
echo "[down] done"
