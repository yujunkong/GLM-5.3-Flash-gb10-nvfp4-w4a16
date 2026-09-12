#!/usr/bin/env bash
# clocks.sh — lock GB10 GPU clocks for serving (measured +5–7% decode).
# No sudo: privileged docker on each node. Idempotent; re-apply after GPU reset/reboot.
# Usage: ./clocks.sh [max_mhz|reset]   (default 2400; reset = stock auto-boost)
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
# shellcheck disable=SC1091
[[ -f "$SCRIPT_DIR/.env" ]] && { set -a; source "$SCRIPT_DIR/.env"; set +a; }

IMAGE="${IMAGE:-glm53-spark:2x-sm121}"
SSH_USER="${SSH_USER:-$USER}"
WORKER_HOST="${WORKER_HOST:-${WORKER_IP:-192.168.100.20}}"
WORKER_SSH="${WORKER_SSH:-${SSH_USER}@${WORKER_HOST}}"
MAXMHZ="${1:-${CLOCK_MHZ:-2400}}"

ssh_n() { ssh -o BatchMode=yes -o ConnectTimeout=30 -o StrictHostKeyChecking=accept-new "$WORKER_SSH" "$@"; }

if [ "$MAXMHZ" = "reset" ]; then
  echo "[clocks] resetting to stock auto-boost (head + worker)"
  docker run --rm --privileged --gpus all --entrypoint '' "$IMAGE" nvidia-smi -rgc 2>&1 | tail -n 1 | sed 's/^/[head] /'
  ssh_n "docker run --rm --privileged --gpus all --entrypoint '' $IMAGE nvidia-smi -rgc 2>&1 | tail -n 1" 2>&1 | sed 's/^/[worker] /'
  sleep 4
  echo "[clocks] head:  $(nvidia-smi -q -d CLOCK 2>/dev/null | awk '/Graphics/{print $3; exit}')"
  exit 0
fi

sleep 2
echo "[clocks] locking to ${MAXMHZ}MHz (head + worker)"
docker run --rm --privileged --gpus all --entrypoint '' "$IMAGE" nvidia-smi -lgc "$MAXMHZ,$MAXMHZ" 2>&1 | tail -n 2 | sed 's/^/[head] /'
ssh_n "docker run --rm --privileged --gpus all --entrypoint '' $IMAGE nvidia-smi -lgc $MAXMHZ,$MAXMHZ 2>&1 | tail -n 2" 2>&1 | sed 's/^/[worker] /'
sleep 4
echo "[clocks] head:  $(nvidia-smi -q -d CLOCK 2>/dev/null | awk '/Graphics/{print $3; exit}')"
echo "[clocks] worker: $(ssh_n 'nvidia-smi -q -d CLOCK 2>/dev/null | grep -oE \"[0-9]+ MHz\" | head -n 1' 2>/dev/null)"
