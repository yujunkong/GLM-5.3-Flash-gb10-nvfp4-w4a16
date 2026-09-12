# Applied stack & status (2× DGX Spark)

Last update: 2026-09-11.

## Goals

| | Target | Current (Golden) | Gap |
|--|--------|------------------|-----|
| Decode | **40–50 tok/s** (soak / C1-class) | ~**33–36** tok/s | **−4 ~ −15 tok/s** |
| Accept | **50–70%** (`/metrics`) | ~**43–45%** | **−5 ~ −25 pp** |

Concurrent C6 (~80+ tok/s)는 이미 도달. **싱글스트림 디코드 + 승인율**이 미달.

## What is applied (KEEP)

### Image / bring-up

- Image: `glm53-spark:2x-sm121` (upstream SM121 patches + GB10)
- Dist: **mp TP=2** (`DIST_BACKEND=mp`), HEAD `192.168.100.10` / WORKER `192.168.100.20`
- RoCE: `enp1s0f0np0` / `rocep1s0f0`, GID 3
- HF cache: host `HOST_CACHE` → `/root/.cache` (Hub id, no revision pin)
- Clocks: `clocks.sh` **2400 MHz** via `up.sh` (`LOCK_CLOCKS=1`)

### Models

- Target: `canada-quant/glm-5.3-w4a16-mtp`
- Draft: `incoai/GLM-5.3-Flash-DFlash2`
- Spec: `method=dflash`, `num_speculative_tokens=7`

### Golden serve knobs (do not retune casually)

| Knob | Value |
|------|--------|
| `DFLASH_SELECTOR_TOP_K` | 32 |
| `DFLASH_WALK_MODE` | edge |
| `DFLASH_TOKENS` | 7 |
| `DFLASH2_ACC_PROBE` | **0** (prod) |
| `ENFORCE_EAGER` | 1 |
| `CUDA_GRAPHS` | 0 |
| `DISABLE_FLASHINFER_AUTOTUNE` | 1 |
| `MOE_BACKEND` | marlin |
| `KV_CACHE_DTYPE` | fp8_e4m3 |
| `KV_CACHE_MEMORY` | 9 GiB pin |
| `MAX_MODEL_LEN` | 1048576 |
| `GLM53_SM121_MLA` | 0 (SM90 path + overlays) |

### Runtime overlays / patches

| Overlay | Role |
|---------|------|
| `patches/glm5next_model.py` | W4A16 dense MLP BF16 + SupportsEagle3 |
| `patches/sparse_attn_indexer_kpool.py` | kpool indexer + workspace reuse for pool_topk buffers |
| `patches/qwen3_dflash2.py` | DFlash2 model |
| `patches/dflash2_speculator.py` | edge walk + **optional** ACC probe |
| `patches/*warmup*.py` | DFlash / mHC warmups (API aligned) |
| `runtime/apc/coordinator.patched.py` | hybrid APC (`gen-apc.sh`) |
| `runtime/kv/kv_cache_utils.patched.py` | DFLASH2-DRAFTER-GROUP (`gen-kv-groups.sh`) |
| `runtime/sm90/flashinfer_mla_sparse_sm90.patched.py` | fp8 plan dtype (`gen-sm90-fp8.sh`) |
| `chat_template_mm.jinja` | multimodal chat template |

### Analysis / debug (off by default)

- `DFLASH2_ACC_PROBE=1` → `/logs/dflash2-acc-state.json`, `.csv`, `.prom` + A/B/C
- Harness: `bench/run_accept_ceiling_diag.sh`
- Docs: `docs/accept-ceiling-debug.md`, `docs/impl-accept-ceiling-debug.md`
- PROBE=1 ≈ **−10% soak** (GPU→CPU sync) — never leave on in prod

### Bench harness

- `bench/bench_config.sh`, `bench_decode.py`, `bench_c.py`, `bench_reject_split.py`, `acceptance_ratio.py`
- Results under `benchmarks/` + summary `benchmarks/RESULTS.md`

## Problems (explicit)

1. **Accept ceiling ~0.43–0.45**  
   - First-reject **B ≈ 87%**, of which **B2 ≈ 90%** (target in top-k but not unary#1; edge walk misses).  
   - `TOP_K` / `WALK` / unary / graphs 실험은 구 레포에서 discard.  
   - Upstream NVFP4 문서의 0.6–0.8은 **다른 타깃 정합** — 이 W4A16 스택에 전가 불가.

2. **Decode soak below 40 tok/s**  
   - Golden soak mean/median ≈ **33–36** tok/s.  
   - Accept이 ~0.44라 speculative 이득이 목표(0.50–0.70)보다 작음 → 디코드 상한도 같이 눌림.  
   - C6 concurrent는 ~80+로 양호하나 **목표 게이트는 싱글스트림 디코드**.

3. **Draft–target mismatch**  
   - Drafter `incoai/GLM-5.3-Flash-DFlash2` vs quantised W4A16 target.  
   - 0.50+ accept는 **draft 재학습 / 타깃 교체(예: NVFP4 lane 분리)** 없이는 비현실적.

4. **Debug vs serve**  
   - 분석용 PROBE는 성능 희생. 운영은 반드시 `PROBE=0`.

## Next steps (toward goals)

| Priority | Action | Helps |
|----------|--------|--------|
| P0 | W4A16-aligned draft 또는 NVFP4 target 실험 lane | Accept → 50–70% |
| P1 | Accept 개선 후 soak 재측정 (40–50 tok/s) | Decode |
| P2 | PROBE=1로 regress A/B/C after any draft change | Validation |
| Avoid | TOP_K=48, unary prod, ACC_PROBE=1 serving, graphs ON | Already discarded |

## Related docs

- `docs/benchmark.md` — numbers & goals  
- `docs/accept-ceiling-debug.md` — how to run diag  
- `docs/upstream-sync-audit.md` — DFlash2/GLM sync  
- `docs/w4a16-vs-nvfp4.md` — code comparison  
- `docs/final-improvement-report.md` — this cycle’s completion report  
- `benchmarks/RESULTS.md` — latest suite table  
- `benchmarks/pr54282-RESULTS.md` — #54282 A/B (**DISCARD** on overlay)  
- `README.md` §13–15 — user-facing summary  
