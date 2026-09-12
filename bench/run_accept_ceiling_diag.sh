#!/usr/bin/env bash
# Accept-ceiling diagnostic harness. Does NOT change Golden knobs.
# Requires a live server with DFLASH2_ACC_PROBE=1 (temporary boot).
# After measurement, set DFLASH2_ACC_PROBE=0 and re-up for production.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/bench"
# shellcheck disable=SC1091
[[ -f "$ROOT/.env" ]] && set -a && source "$ROOT/.env" && set +a

API="${API:-http://127.0.0.1:${API_PORT:-8000}}"
LOG_DIR="${LOG_DIR:-$HOME/logs/glm53}"
TS="$(date +%Y%m%d-%H%M%S)"
OUTDIR="${OUTDIR:-$ROOT/benchmarks/accept-ceiling-diag-$TS}"
mkdir -p "$OUTDIR"

echo "=== accept-ceiling diag ==="
echo "API=$API  LOG_DIR=$LOG_DIR  OUTDIR=$OUTDIR"
echo "NOTE: A/B/C / CSV / dflash2_acc_* gauges need DFLASH2_ACC_PROBE=1."
echo "      Do NOT leave PROBE=1 for production (soak ~-10%)."
echo "      Golden knobs stay TOP_K=32 WALK=edge TOKENS=7."
echo

if ! curl -sf "$API/health" >/dev/null; then
  echo "FAIL: $API/health not ready" >&2
  exit 1
fi

PROBE_STATE="$LOG_DIR/dflash2-acc-state.json"
PROBE_CSV="$LOG_DIR/dflash2-acc.csv"

echo "--- acceptance_ratio ---"
python3 acceptance_ratio.py "$API" | tee "$OUTDIR/acceptance.txt"

echo "--- reject_split x10 ---"
export LOG_DIR
python3 bench_reject_split.py --base "$API" --runs 10 --max-tokens 256 \
  --out "$OUTDIR/reject_split.json" | tee "$OUTDIR/reject_split.txt"

echo "--- probe artifacts ---"
if [[ -f "$PROBE_STATE" ]]; then
  cp -a "$PROBE_STATE" "$OUTDIR/dflash2-acc-state.json"
  echo "copied $PROBE_STATE"
  python3 -c "import json; d=json.load(open('$OUTDIR/dflash2-acc-state.json')); c=d.get('cum',{}); print({k:c.get(k) for k in ('draft_rounds','accepted_tokens','drafted_tokens','lm_head_topk_miss','lm_head_topk_hit_walk_miss','walk_hit_verify_reject','b_unary_would_hit','b_unary_also_miss')})" \
    | tee "$OUTDIR/probe_cum.txt"
else
  echo "WARN: missing $PROBE_STATE (is DFLASH2_ACC_PROBE=1?)" | tee "$OUTDIR/probe_cum.txt"
fi
if [[ -f "$PROBE_CSV" ]]; then
  cp -a "$PROBE_CSV" "$OUTDIR/dflash2-acc.csv"
  echo "copied $PROBE_CSV ($(wc -l < "$PROBE_CSV") lines)"
else
  echo "WARN: missing $PROBE_CSV" | tee -a "$OUTDIR/probe_cum.txt"
fi
PROBE_PROM="$LOG_DIR/dflash2-acc.prom"
if [[ -f "$PROBE_PROM" ]]; then
  cp -a "$PROBE_PROM" "$OUTDIR/dflash2-acc.prom"
  echo "copied $PROBE_PROM"
else
  echo "WARN: missing $PROBE_PROM (worker textfile metrics)" | tee -a "$OUTDIR/probe_cum.txt"
fi

curl -sf "$API/metrics" | rg '^(dflash2_acc_|vllm:spec_decode_num_(accepted|draft))' \
  > "$OUTDIR/metrics_snip.txt" || true

cat > "$OUTDIR/README.txt" <<EOF
accept-ceiling diagnostic — $TS
API=$API
Expect DFLASH2_ACC_PROBE=1 for non-zero A/B/C and CSV.
After diag: set DFLASH2_ACC_PROBE=0 in .env and re-run scripts/up.sh.
Do not change TOP_K / WALK / TOKENS / ENFORCE_EAGER.
EOF

echo
echo "=== done: $OUTDIR ==="
echo "RESTORE Golden: DFLASH2_ACC_PROBE=0 then scripts/up.sh"
