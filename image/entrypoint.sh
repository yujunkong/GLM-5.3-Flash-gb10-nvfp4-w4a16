#!/usr/bin/env bash
# Container entrypoint scaffold — distributed vLLM TP=2 startup goes here.
# Never pin Hugging Face revision. Use standard HF cache only.
set -euo pipefail

# ROLE=head|worker and HEAD_HOST must come from the environment / launcher
: "${ROLE:=head}"
: "${HEAD_HOST:?HEAD_HOST is required (same head IP on every node)}"
: "${MODEL:=canada-quant/glm-5.3-w4a16-mtp}"
: "${TP_SIZE:=2}"
: "${NNODES:=2}"

echo "[entrypoint] ROLE=${ROLE} HEAD_HOST=${HEAD_HOST} TP_SIZE=${TP_SIZE} NNODES=${NNODES}"
echo "[entrypoint] MODEL=${MODEL} (revision not pinned)"
echo "[entrypoint] scaffold only — replace with vLLM multi-node serve"

# Example shape only (not executed as final recipe yet):
# exec vllm serve "${MODEL}" --tensor-parallel-size "${TP_SIZE}" ...

exec "$@"
