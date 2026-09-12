# Applied stack & status (2× DGX Spark)

Last update: 2026-09-12 · **Version:** `v1.0.0-golden`  
Canonical: [`docs/golden-configuration.md`](golden-configuration.md) · Perf: [`docs/performance.md`](performance.md)

## Goals

| | Target | Golden (measured) | Gap |
|--|--------|-------------------|-----|
| Decode | **40–50 tok/s** | **~34.7–35.5** (longrun60m **33.5**) | unmet |
| Accept | **50–70%** | **~0.43–0.44** | unmet |

Concurrent C6 (~80+ tok/s) OK. Primary gates unmet; **stability Golden finalized**.

## What is applied (KEEP)

### Image / bring-up

- Image: `glm53-spark:2x-sm121`
- vLLM `0.28.1rc1.dev580+g385dce36b` · Torch `2.13.0+cu130` · CUDA `13.0` · FlashInfer `0.6.18.dev20260819`
- Dist: **mp TP=2**, HEAD `192.168.100.10` / WORKER `192.168.100.20`
- HF cache: `HOST_CACHE` → `/root/.cache` (no `/var/tmp`)
- Clocks: **2400 MHz** (`LOCK_CLOCKS=1`)

### Models

- Target: `canada-quant/glm-5.3-w4a16-mtp`
- Draft: `incoai/GLM-5.3-Flash-DFlash2`
- Spec: dflash, K=7

### Golden serve knobs

| Knob | Value |
|------|--------|
| `DFLASH_SELECTOR_TOP_K` | 32 |
| `DFLASH_WALK_MODE` | edge |
| `DFLASH_TOKENS` | 7 |
| `DFLASH2_ACC_PROBE` | **0** |
| `ENFORCE_EAGER` | 1 |
| `CUDA_GRAPHS` | 0 |
| `ASYNC_SCHEDULING` | 1 |
| `DISABLE_FLASHINFER_AUTOTUNE` | 1 |
| `MOE_BACKEND` | marlin |
| `KV_CACHE_DTYPE` | fp8_e4m3 |
| `PYTORCH_CUDA_ALLOC_CONF` | expandable_segments:True |
| `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB` | 512 |
| Spin-wait | **pristine** |
| `GLM53_SM121_MLA` | 0 |
| `APPLY_GATE_LINEAR` | 0 |

### Runtime overlays

| Overlay | Role |
|---------|------|
| `patches/glm5next_model.py` | W4A16 dense MLP BF16 + SupportsEagle3 |
| `patches/sparse_attn_indexer_kpool.py` | kpool + SM121 top-k route + workspace reuse |
| `patches/qwen3_dflash2.py` | DFlash2 model |
| `patches/dflash2_speculator.py` | edge walk + optional ACC probe |
| `patches/*warmup*.py` | warmups |
| `runtime/apc|kv|sm90/*` | APC / KV groups / SM90 fp8 plan |
| `runtime/spinwait/shm_broadcast.pristine.py` | Golden spin-wait |
| `scripts/tps_drift_logger.py` | optional ops telemetry |

## Rejected (do not enable)

See [`docs/performance.md`](performance.md) §4 — spin-wait 0.002, logits 64, expandable OFF, PROBE=1 serve, #54282 walk, FlashInfer SM121 rebuild chase, CUBLAS workspace tuning, etc.

## Next steps (outside runtime micro-patch)

| Priority | Action |
|----------|--------|
| P0 | W4A16-aligned draft or NVFP4 lane |
| P1 | Re-measure soak/accept after draft/target change |
| Avoid | Rejected table; chasing 60 tok/s via unproven patches |

## Related

- `README.md`, `CHANGELOG.md`, `VERSION`
- `benchmarks/RESULTS.md`, `evidence/final/optimization-report-20260912.md`
