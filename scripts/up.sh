#!/usr/bin/env bash
# Bring up NEW-repo stack: upstream-based glm53-spark:2x-sm121, mp TP=2.
# Applies KEEP overlays from the validated 2× Spark recipe (not that launcher).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
[[ -f .env ]] || cp .env.example .env
set -a; source .env; set +a

SSH_USER="${SSH_USER:-$USER}"
HEAD_HOST="${HEAD_HOST:?}"
WORKER_HOST="${WORKER_HOST:?}"
REMOTE_ROOT="${REMOTE_ROOT:-$ROOT}"
IMAGE="${IMAGE:-glm53-spark:2x-sm121}"
API_PORT="${API_PORT:-8000}"

ssh_n() { ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new "${SSH_USER}@$1" "${@:2}"; }

echo "[up] generate hybrid APC from $IMAGE"
bash "$ROOT/scripts/gen-apc.sh"
echo "[up] generate DFlash GLM5 KV-group overlay from $IMAGE"
bash "$ROOT/scripts/gen-kv-groups.sh"
echo "[up] generate SM90 fp8 plan-dtype overlay from $IMAGE"
bash "$ROOT/scripts/gen-sm90-fp8.sh"

if [[ "${LOCK_CLOCKS:-1}" == "1" ]]; then
  echo "[up] lock GB10 clocks to ${CLOCK_MHZ:-2400} MHz"
  bash "$ROOT/clocks.sh" "${CLOCK_MHZ:-2400}" 2>&1 | tail -n 8 \
    || echo "[up] WARN: clocks.sh failed — continuing at stock clocks" >&2
else
  echo "[up] LOCK_CLOCKS=0 — stock auto-boost"
fi

echo "[up] stop old/new containers on both nodes"
docker rm -f glm53 vllm_glm53_w4a16 mentatd mentatd-serve 2>/dev/null || true
ssh_n "$WORKER_HOST" 'docker rm -f glm53 vllm_glm53_w4a16 mentatd 2>/dev/null || true'

echo "[up] ensure image on worker"
if ! ssh_n "$WORKER_HOST" "docker image inspect '$IMAGE' >/dev/null 2>&1"; then
  echo "[up] transferring $IMAGE (large)..."
  docker save "$IMAGE" | ssh_n "$WORKER_HOST" 'docker load'
fi

echo "[up] sync repo (compose/patches/runtime/env) to worker"
rsync -az --delete --exclude '.git' --exclude 'reference-notes' --exclude '__pycache__' \
  "$ROOT/" "${SSH_USER}@${WORKER_HOST}:${REMOTE_ROOT}/"

echo "[up] worker (rank 1)"
ssh_n "$WORKER_HOST" "cd '$REMOTE_ROOT' && docker compose -f compose/glm53.yaml --profile worker --env-file .env up -d --force-recreate"
sleep 20
echo "[up] head (rank 0)"
docker compose -f compose/glm53.yaml --profile head --env-file .env up -d --force-recreate

echo "[up] wait for http://${HEAD_HOST}:${API_PORT}/health"
for i in $(seq 1 180); do
  if curl -sf -m 5 "http://${HEAD_HOST}:${API_PORT}/health" >/dev/null; then
    echo "[up] READY"
    curl -sf "http://${HEAD_HOST}:${API_PORT}/v1/models" | head -c 400; echo
    exit 0
  fi
  docker ps --format '{{.Names}}' | grep -qx glm53 || { echo 'head died'; docker logs --tail 120 glm53; exit 1; }
  echo "  ... ${i}0s"
  sleep 10
done
echo "[up] TIMEOUT"; docker logs --tail 100 glm53; exit 1
