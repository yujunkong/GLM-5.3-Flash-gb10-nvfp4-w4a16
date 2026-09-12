# Golden Configuration — W4A16 2× DGX Spark (v1.0.0-golden)

**Status:** production KEEP · last verified 2026-09-12  
**Tag / version:** `v1.0.0-golden`  
**Principle:** stability over chasing 40–60 tok/s with unverified runtime patches.

This document is the **single source of truth** for what is enabled in production.
Numbers below are **measured** on this stack unless marked otherwise.

---

## Stack identity

| Item | Value |
|------|--------|
| Hardware | 2× NVIDIA DGX Spark / GB10 / SM121 |
| Topology | TP=2, `DIST_BACKEND=mp`, HEAD `192.168.100.10` / WORKER `192.168.100.20` |
| Image | `glm53-spark:2x-sm121` |
| Base | `vllm/vllm-openai:glm53-flash-arm64-cu130` |
| vLLM | `0.28.1rc1.dev580+g385dce36b` |
| Torch | `2.13.0+cu130` |
| CUDA | `13.0` |
| FlashInfer | `flashinfer-python 0.6.18.dev20260819` |
| Target | `canada-quant/glm-5.3-w4a16-mtp` (Hub id, **no revision pin**) |
| Draft | `incoai/GLM-5.3-Flash-DFlash2` |
| Spec | DFlash2, `DFLASH_TOKENS=7` |
| KV | `fp8_e4m3`, pinned `KV_CACHE_MEMORY=9663676416` |
| HF cache | host `${HOST_CACHE}` → `/root/.cache` (standard HF; **no `/var/tmp`**) |

Bring-up: `bash scripts/up.sh` with `.env` copied from `.env.example`.

---

## ENABLED / KEEP (production)

| Knob / component | Value | Why KEEP |
|------------------|-------|----------|
| Clock lock | `LOCK_CLOCKS=1`, `CLOCK_MHZ=2400` | Drift ~6.6%→~0.6% vs stock boost; repro |
| Async scheduling | `ASYNC_SCHEDULING=1` | Already Golden; not re-tuned |
| MoE | `MOE_BACKEND=marlin` | Production path |
| Eager | `ENFORCE_EAGER=1`, `CUDA_GRAPHS=0` | Graphs not on W4A16 Golden path |
| FlashInfer autotune | `DISABLE_FLASHINFER_AUTOTUNE=1` | Stability |
| DFlash2 K | `DFLASH_TOKENS=7` | Spec depth |
| Selector | `DFLASH_SELECTOR_TOP_K=32` | 16→32 helped; 48 no meaningful gain → stay 32 |
| Walk | `DFLASH_WALK_MODE=edge` | Golden |
| ACC probe | `DFLASH2_ACC_PROBE=0` | `=1` ≈ −10% soak |
| KV dtype | `KV_CACHE_DTYPE=fp8_e4m3` | Golden |
| Allocator | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | OFF A/B: TPS **−3.99%**, accept↓ |
| Logits budget | `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=512` | 64 A/B: TPS **−2.77%** |
| Spin-wait | **pristine** `busy_loop_s=1.0` via `SPINWAIT_PATCH_HOST=.../shm_broadcast.pristine.py` | patched 0.002: TPS **−1.63%**, drift worse |
| MLA path | `GLM53_SM121_MLA=0` | SM90 + overlays; SM121 MLA not production default |
| Gate linear | `APPLY_GATE_LINEAR=0` | Off |
| APC | `APPLY_APC_PATCH=1` | Required hybrid APC |
| Sparse kpool overlay | `patches/sparse_attn_indexer_kpool.py` (+ workspace reuse) | SM121 top-k route + **+0.70%** KEEP |
| TPS drift logger | `scripts/tps_drift_logger.py` (optional host side-process) | Ops only; no decode-path change |

### Measured Golden band (PROBE=0, clocks 2400)

| Metric | Band / suite | Source |
|--------|--------------|--------|
| Production soak (short/medium) | **~34.7–35.5 tok/s** (suite peaks to **35.9**) | `upstream-baseline-clocks2400` med 35.87 / mean 34.54; checklist-final 35.09 |
| Acceptance | **~0.43–0.44** (observed **0.417–0.448**) | `/metrics` suites |
| Longrun 60m continuous | med **33.50**, accept **0.400** | `evidence/longrun-20260912/` — floor under sustained load |
| C6 concurrent | **~81–87 tok/s** agg | stretch; not primary gate |

**Interpretation:** Golden prioritizes **stability + reproducibility**. Goal gates (40–50 tok/s, 50–70% accept) remain **unmet** and are **not** closed by further micro runtime patches on this W4A16+DFlash2 pair.

---

## DISABLED / DO NOT ENABLE (Rejected / Tested)

Format: **change → measured → decision**.

| Change | Measured | Decision |
|--------|----------|----------|
| Spin-wait `busy_loop_s=0.002` | TPS **−1.63%**, drift **3.88%** (worse vs ~0.55%) | **OFF** — keep pristine |
| `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=64` | TPS **−2.77%** | **OFF** — keep 512 |
| `expandable_segments` OFF | TPS **−3.99%**, accept **−0.014** | **DO NOT disable** — keep True |
| `DFLASH2_ACC_PROBE=1` in prod | ≈ **−10%** soak | **OFF** — analysis only |
| `#54282 IS_DRAFTING` overlay walk | soak ~17–23, accept ~0.36 | **DISCARD** — restore pre-54282 local walk |
| FlashInfer SM121 native rebuild | reported ~**32 tok/s** class regression on this class of attempt | **NOT applied** |
| Marlin `_C` / SM121 kernel rebuild | out of scope; quality/risk | **NOT applied** |
| `CUBLAS_WORKSPACE_CONFIG` tuning | no meaningful gain / some path worse (prior tests) | **NOT applied** |
| `GLM53_SM121_MLA=1` as default | not Golden path | **OFF** (`=0`) |
| `APPLY_GATE_LINEAR=1` | not Golden | **OFF** |
| CUDA Graphs ON | W4A16 uses eager | **OFF** |
| Expert Parallel | not used | **OFF** |
| BF16 KV | not Golden | **OFF** (fp8_e4m3) |
| TOP_K=48 / TOKENS retune | no meaningful accept gain past 32 | **frozen at 32 / 7** |
| Blind chase **60 tok/s** via runtime patches | not evidenced on this stack | **forbidden** |

Experimental files kept for retest only (never default):

- `runtime/spinwait/shm_broadcast.patched.py`
- `compose/overrides/logits64.yaml`
- `compose/overrides/alloc-no-expandable.yaml`

---

## Related docs

- [`docs/performance.md`](performance.md) — experiments & rejected opts detail  
- [`docs/runtime-patches-v1.0.md`](runtime-patches-v1.0.md) — runtime patch inventory  
- [`docs/applied-stack.md`](applied-stack.md) — stack + next steps (NVFP4 / draft)  
- [`docs/benchmark.md`](benchmark.md) / [`benchmarks/RESULTS.md`](../benchmarks/RESULTS.md)  
- [`evidence/final/optimization-report-20260912.md`](../evidence/final/optimization-report-20260912.md)  
