#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
[[ -f .env ]] && set -a && source .env && set +a
HEAD_HOST="${HEAD_HOST:-127.0.0.1}"
API_PORT="${API_PORT:-8000}"
echo "IMAGE=${IMAGE:-glm53-spark:2x-sm121} DIST=${DIST_BACKEND:-mp} MODEL=${MODEL:-}"
docker ps --filter name=glm53 --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}'
curl -sf -m 5 "http://${HEAD_HOST}:${API_PORT}/health" && echo ' health:OK' || echo ' health:NOK'
curl -sf -m 5 "http://${HEAD_HOST}:${API_PORT}/v1/models" 2>/dev/null | head -c 300 || true
echo
