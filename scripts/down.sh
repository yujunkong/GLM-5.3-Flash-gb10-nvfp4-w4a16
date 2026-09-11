#!/usr/bin/env bash
# Stop glm53 (+ optional mentat) on both nodes. Never delete Hugging Face cache.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if [[ -f .env ]]; then
  # shellcheck disable=SC1091
  set -a && source .env && set +a
fi

SSH_USER="${SSH_USER:-${USER}}"
HEAD_HOST="${HEAD_HOST:?Set HEAD_HOST in .env}"
WORKER_HOST="${WORKER_HOST:?Set WORKER_HOST in .env}"
REMOTE_ROOT="${REMOTE_ROOT:-${ROOT}}"
COMPOSE_FILES="${COMPOSE_FILES:--f compose/glm53.yaml -f compose/overrides/production.yaml -f compose/overrides/dflash2.yaml}"
STOP_MENTAT="${STOP_MENTAT:-0}"

ssh_n() { ssh -o BatchMode=yes "${SSH_USER}@$1" "${@:2}"; }

echo "[down] stopping glm53 on ${HEAD_HOST} and ${WORKER_HOST}"
for h in "${HEAD_HOST}" "${WORKER_HOST}"; do
  ssh_n "$h" "cd '${REMOTE_ROOT}' && docker compose ${COMPOSE_FILES} --env-file .env stop glm53 2>/dev/null || docker stop glm53 2>/dev/null || true" &
done
wait

if [[ "${STOP_MENTAT}" == "1" ]]; then
  echo "[down] stopping mentatd / mentatd-serve"
  ssh_n "${HEAD_HOST}" "cd '${REMOTE_ROOT}' && docker compose -f compose/mentatd-serve.yaml --env-file .env stop 2>/dev/null || true" || true
  for h in "${HEAD_HOST}" "${WORKER_HOST}"; do
    ssh_n "$h" "cd '${REMOTE_ROOT}' && docker compose -f compose/mentatd.yaml --env-file .env stop 2>/dev/null || true" &
  done
  wait
fi

echo "[down] HF cache left intact (by design)"
