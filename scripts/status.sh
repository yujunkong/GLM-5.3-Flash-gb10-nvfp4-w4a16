#!/usr/bin/env bash
# Print HEAD/WORKER, ranks, model, DFlash, NCCL, health snapshot (scaffold).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if [[ -f .env ]]; then
  # shellcheck disable=SC1091
  set -a && source .env && set +a
fi

cat <<EOF
[status] scaffold snapshot
HEAD_HOST=${HEAD_HOST:-unset}
WORKER_HOST=${WORKER_HOST:-unset}
TP_SIZE=${TP_SIZE:-2}
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
NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-unset}
NCCL_IB_HCA=${NCCL_IB_HCA:-unset}
NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-unset}
# TODO: live GPU/rank, vLLM/CUDA/FlashInfer versions, health, OpenAI API
EOF
