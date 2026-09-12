# Runtime patches for W4A16 v1.0.x (2× DGX Spark / GB10)

Upstream GB10 notes recommend four ops/runtime items. **This stack measured
several of them** — Golden defaults follow evidence, not blind copy.

| ID | Patch | In repo? | Golden default | This-stack evidence |
|----|-------|----------|----------------|---------------------|
| 01 | Spin-wait `busy_loop_s=0.002` | ✅ `runtime/spinwait/*.py` | **pristine (1.0)** | TPS **−1.6%**, drift **worse** → not enabled |
| 02 | `expandable_segments:True` | ✅ compose / `.env` | **ON** | OFF was TPS **−4%** / accept↓ → keep ON |
| 03 | Logits budget 64 MiB | ✅ env passthrough | **unset (=512)** | 64 MiB short soak TPS **−2.77%** → not default |
| 04 | TPS drift logger | ✅ `scripts/tps_drift_logger.py` | optional side process | ops only; no quality impact |
| 05 | Top-K GB10 fallback | ✅ image + kpool overlay | already on | **PATCH=NONE** (no rewrite; TOP_K=32) |

## What actually helped long-run drift here

- **`LOCK_CLOCKS=1` / 2400 MHz** — drift ~6.6% → ~0.6% (keep).
- Sparse indexer **workspace buffer reuse** (code overlay) — +0.7% KEEP.
- Spin-wait 0.002 and logits=64 did **not** improve soak drift/TPS on W4A16 Golden.

## Enable / retest (not Golden)

```bash
# Spin-wait retest only
SPINWAIT_PATCH_HOST=../runtime/spinwait/shm_broadcast.patched.py bash scripts/up.sh

# Logits 64 experiment only
# compose/overrides/logits64.yaml 또는:
# VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=64 bash scripts/up.sh  # after compose passthrough

# Telemetry (host)
python3 scripts/tps_drift_logger.py --out "$LOG_DIR/tps-drift.csv" --interval 30
```

See `evidence/spin-wait/`, `evidence/logits-budget/`, `evidence/expandable-segments/`,
`evidence/longrun-20260912/report.md`.
