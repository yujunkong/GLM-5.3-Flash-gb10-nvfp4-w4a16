#!/usr/bin/env bash
# Cluster status snapshot (env + remote docker / health probes).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if [[ -f .env ]]; then
  # shellcheck disable=SC1091
  set -a && source .env && set +a
fi

SSH_USER="${SSH_USER:-${USER}}"
HEAD_HOST="${HEAD_HOST:-unset}"
WORKER_HOST="${WORKER_HOST:-unset}"
API_PORT="${API_PORT:-8002}"
STATUS_PORT="${STATUS_PORT:-8082}"

ssh_n() { ssh -o BatchMode=yes -o ConnectTimeout=5 "${SSH_USER}@$1" "${@:2}" 2>/dev/null || echo "unreachable"; }

echo "=== GLM-5.3-Flash 2× Spark status ==="
cat <<EOF
HEAD_HOST=${HEAD_HOST}
WORKER_HOST=${WORKER_HOST}
TP=${TP_SIZE:-${TP:-2}}
NNODES=${NNODES:-2}
MODEL=${MODEL:-canada-quant/glm-5.3-w4a16-mtp}
DRAFT_MODEL=${DRAFT_MODEL:-incoai/GLM-5.3-Flash-DFlash2}
DFLASH_TOKENS=${DFLASH_TOKENS:-7}
DFLASH_SELECTOR_TOP_K=${DFLASH_SELECTOR_TOP_K:-32}
DFLASH_WALK_MODE=${DFLASH_WALK_MODE:-edge}
MOE_BACKEND=${MOE_BACKEND:-marlin}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-1048576}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-6}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-8192}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.88}
NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-unset}
NCCL_IB_HCA=${NCCL_IB_HCA:-unset}
NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-unset}
IMAGE=${IMAGE:-glm53-spark:2x-sm121}
EOF

if [[ "${HEAD_HOST}" != "unset" ]]; then
  echo "--- head docker ---"
  ssh_n "${HEAD_HOST}" "docker ps --filter name=glm53 --filter name=mentatd --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}'"
  echo "--- health ---"
  echo -n "/v1/models: "; curl -s -o /dev/null -w '%{http_code}\n' -m 5 "http://${HEAD_HOST}:${API_PORT}/v1/models" || echo fail
  echo -n "status :${STATUS_PORT}: "; curl -s -o /dev/null -w '%{http_code}\n' -m 5 "http://${HEAD_HOST}:${STATUS_PORT}/" || echo fail
fi
if [[ "${WORKER_HOST}" != "unset" ]]; then
  echo "--- worker docker ---"
  ssh_n "${WORKER_HOST}" "docker ps --filter name=glm53 --filter name=mentatd --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}'"
fi
