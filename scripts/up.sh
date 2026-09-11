#!/usr/bin/env bash
# Bring up 2× DGX Spark GLM-5.3-Flash (mentatd → glm53 on head+worker).
# Usage: ./scripts/up.sh
# Requires SSH to HEAD_HOST and WORKER_HOST; repo checked out at REMOTE_ROOT on both.
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
IMAGE="${IMAGE:-glm53-spark:2x-sm121}"
COMPOSE_FILES="${COMPOSE_FILES:--f compose/glm53.yaml -f compose/overrides/production.yaml -f compose/overrides/dflash2.yaml}"
API_PORT="${API_PORT:-8002}"

ssh_n() { ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new "${SSH_USER}@$1" "${@:2}"; }

echo "[up] 1/12 configuration"
[[ "${TP_SIZE:-${TP:-2}}" == "2" ]] || { echo "FATAL: TP_SIZE must be 2 for this recipe" >&2; exit 1; }
[[ -n "${MODEL:-}" ]] || { echo "FATAL: MODEL required" >&2; exit 1; }
echo "  HEAD=${HEAD_HOST} WORKER=${WORKER_HOST} IMAGE=${IMAGE} MODEL=${MODEL}"

echo "[up] 2/12 host connectivity"
for h in "${HEAD_HOST}" "${WORKER_HOST}"; do
  ssh_n "$h" "echo ok:\$(hostname)" >/dev/null
done

echo "[up] 3/12 docker check"
for h in "${HEAD_HOST}" "${WORKER_HOST}"; do
  ssh_n "$h" "docker info >/dev/null"
done

echo "[up] 4/12 network/interface check"
IFACE="${NCCL_SOCKET_IFNAME:-}"
if [[ -n "$IFACE" ]]; then
  for h in "${HEAD_HOST}" "${WORKER_HOST}"; do
    ssh_n "$h" "ip link show dev '${IFACE}' >/dev/null" \
      || { echo "FATAL: ${IFACE} missing on ${h}" >&2; exit 1; }
  done
else
  echo "  NCCL_SOCKET_IFNAME unset — skipping iface existence check"
fi

echo "[up] 5/12 image check"
for h in "${HEAD_HOST}" "${WORKER_HOST}"; do
  ssh_n "$h" "docker image inspect '${IMAGE}' >/dev/null" \
    || { echo "FATAL: image ${IMAGE} missing on ${h}; build/load first" >&2; exit 1; }
done

echo "[up] 6/12 mentatd on both nodes"
ssh_n "${HEAD_HOST}" "cd '${REMOTE_ROOT}' && docker compose -f compose/mentatd.yaml --env-file .env up -d" &
ssh_n "${WORKER_HOST}" "cd '${REMOTE_ROOT}' && docker compose -f compose/mentatd.yaml --env-file .env up -d" &
wait

echo "[up] 7/12 mentatd-serve on head (optional front door :6381)"
ssh_n "${HEAD_HOST}" "cd '${REMOTE_ROOT}' && docker compose -f compose/mentatd-serve.yaml --env-file .env up -d" || true

echo "[up] 8–9/12 distributed vLLM (worker + head)"
# Worker first is fine under mentat (retries); still start worker then head for status clarity.
ssh_n "${WORKER_HOST}" "cd '${REMOTE_ROOT}' && ROLE=worker NODE_RANK=1 MENTAT_NODE_IP=${WORKER_HOST} MENTAT_PEERS=${HEAD_HOST}:6379 docker compose ${COMPOSE_FILES} --env-file .env up -d" &
ssh_n "${HEAD_HOST}" "cd '${REMOTE_ROOT}' && ROLE=head NODE_RANK=0 MENTAT_NODE_IP=${HEAD_HOST} MENTAT_PEERS=${WORKER_HOST}:6379 docker compose ${COMPOSE_FILES} --env-file .env up -d" &
wait

echo "[up] 10–11/12 wait for /health and /v1/models on head:${API_PORT}"
ok=0
for _ in $(seq 1 120); do
  code="$(curl -s -o /dev/null -w '%{http_code}' -m 5 "http://${HEAD_HOST}:${API_PORT}/v1/models" || true)"
  if [[ "$code" == "200" ]]; then ok=1; break; fi
  sleep 15
done
[[ "$ok" == "1" ]] || { echo "FATAL: timed out waiting for :${API_PORT}/v1/models" >&2; exit 1; }
echo "  /v1/models OK"

echo "[up] 12/12 inference self-test"
if [[ -x "${ROOT}/scripts/self-test.sh" ]]; then
  HEAD_HOST="${HEAD_HOST}" API_PORT="${API_PORT}" "${ROOT}/scripts/self-test.sh" || true
fi
echo "[up] done"
