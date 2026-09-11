#!/usr/bin/env bash
# Bring up 2× DGX Spark GLM-5.3-Flash cluster (scaffold).
# Order: validate → connectivity → docker → iface → image → worker → head → vLLM → health → self-test
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

# Load .env if present (never commit secrets)
if [[ -f .env ]]; then
  # shellcheck disable=SC1091
  set -a && source .env && set +a
fi

: "${HEAD_HOST:?Set HEAD_HOST in .env (example: 192.168.100.10)}"
: "${WORKER_HOST:?Set WORKER_HOST in .env}"
: "${TP_SIZE:=2}"
: "${NNODES:=2}"
: "${NCCL_SOCKET_IFNAME:?Set NCCL_SOCKET_IFNAME in .env}"

echo "[up] scaffold — configuration snapshot"
echo "  HEAD_HOST=${HEAD_HOST} WORKER_HOST=${WORKER_HOST} TP=${TP_SIZE} NNODES=${NNODES}"
echo "  NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME} NCCL_IB_HCA=${NCCL_IB_HCA:-} GID=${NCCL_IB_GID_INDEX:-}"

# TODO: 1) config validation 2) host connectivity 3) docker 4) iface exists
# TODO: 5) image 6) worker prep 7) head prep 8) distributed start 9) rank ready
# TODO: 10) /health 11) /v1/models 12) inference self-test
echo "[up] NOT IMPLEMENTED — complete after upstream compose/launcher port"
exit 1
