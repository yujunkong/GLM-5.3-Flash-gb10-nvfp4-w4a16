#!/usr/bin/env bash
# Host-side GB10 clock control (checklist 20260912 Phase 1).
# Wraps repo-root clocks.sh. Run on HEAD host, not inside the container.
# Usage:
#   scripts/clocks.sh set [mhz]   # default 2400 (or CLOCK_MHZ)
#   scripts/clocks.sh reset       # stock auto-boost
#   scripts/clocks.sh status
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
[[ -f "$ROOT/.env" ]] && { set -a; source "$ROOT/.env"; set +a; }

CMD="${1:-status}"
MHZ="${2:-${CLOCK_MHZ:-2400}}"

status() {
  echo "[clocks/status] head:"
  nvidia-smi --query-gpu=clocks.current.graphics,clocks.max.graphics,temperature.gpu,power.draw \
    --format=csv 2>/dev/null || nvidia-smi -q -d CLOCK,TEMPERATURE 2>/dev/null | head -40
  echo "[clocks/status] worker (${WORKER_HOST:-192.168.100.20}):"
  ssh -o BatchMode=yes -o ConnectTimeout=20 \
    "${SSH_USER:-$USER}@${WORKER_HOST:-192.168.100.20}" \
    "nvidia-smi --query-gpu=clocks.current.graphics,clocks.max.graphics,temperature.gpu,power.draw --format=csv 2>/dev/null" \
    || echo "(worker query failed)"
}

case "$CMD" in
  set)
    bash "$ROOT/clocks.sh" "$MHZ"
    status
    ;;
  reset)
    bash "$ROOT/clocks.sh" reset
    status
    ;;
  status)
    status
    ;;
  *)
    echo "usage: $0 {set [mhz]|reset|status}" >&2
    exit 2
    ;;
esac
