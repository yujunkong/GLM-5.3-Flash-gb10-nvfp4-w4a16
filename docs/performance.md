# Performance notes — W4A16 2× DGX Spark

**Version:** v1.0.0-golden · **Last update:** 2026-09-12  
**Rule:** only measured results and explicitly retained prior profiling conclusions. No projected 40/60 tok/s claims.

Canonical knobs: [`docs/golden-configuration.md`](golden-configuration.md).

---

## 1. Goals vs measured

| Gate | Target | Measured Golden | Status |
|------|--------|-----------------|--------|
| Soak decode | 40–50 tok/s | **~34.7–35.5** (short/medium); longrun60m med **33.5** | FAIL |
| Accept | 50–70% | **~0.43–0.44** (K=7) | FAIL |
| C6 aggregate | stretch | **~81–87** tok/s | OK concurrent |

Headroom to goal gates requires **draft/target alignment or NVFP4 lane**, not more W4A16 micro-patches.

---

## 2. Bottleneck conclusions (retained)

Prior profiling / engineering review on this stack (not re-run as full nsys in the 2026-09-12 checklist2 cycle):

| Topic | Conclusion | Action |
|-------|------------|--------|
| FlashInfer MLA | **Not** the primary single-stream bottleneck on Golden (eager + SM90 overlays) | Do not chase FlashInfer SM121 rebuild for TPS |
| Marlin `_moe` | **Not** the primary bottleneck | No Marlin `_C` / SM121 rebuild |
| Decode cost | Dominated by **BF16 / cublasLt-class GEMM** and **LPDDR bandwidth** on GB10 | Kernel swaps out of scope |
| KDA `in_proj_qkvbfg_a` | On the order of **~10%** CUDA time class; safe win room small under current structure | No hot-path rewrite |
| Shared expert / `lm_head` | Not treated as large safe optimization targets | No integration rewrite |
| DFlash2 accept @ K=7 | **~0.43–0.44** ceiling; reject **B2-dominated** | Knobs frozen |
| `TOP_K` 16→32 | Improved; **48** no meaningful further gain | **KEEP 32** |
| 60 tok/s via runtime patches | **Not evidenced** | Do not add risky patches to chase it |

---

## 3. Validated suite table

| Date | Label | Soak med | Accept | Notes |
|------|-------|----------|--------|-------|
| 2026-09-11 | `upstream-baseline-clocks2400` | **35.87** | **0.430** | first Golden baseline |
| 2026-09-11 | `golden-post-acc-debug` | 32.55* | 0.448 | *3-run noise |
| 2026-09-12 | clocks B 2400 10m | 34.33 | 0.417 | KEEP lock (stability) |
| 2026-09-12 | checklist-final (then spinwait reverted) | **35.09** | 0.421 | C6 **87.2** |
| 2026-09-12 | longrun 60m | **33.50** | 0.400 | continuous load floor |
| 2026-09-12 | sparse reuse KEEP confirm | 33.63 | 0.422 | +0.70% vs same-day control |

Full table: [`benchmarks/RESULTS.md`](../benchmarks/RESULTS.md), [`docs/benchmark.md`](benchmark.md).

---

## 4. Rejected / Tested (do not delete)

| Experiment | Change | Result | Decision |
|------------|--------|--------|----------|
| Spin-wait | `busy_loop_s` 1.0→**0.002** | TPS **−1.63%**, drift worse | **OFF** |
| Logits budget | **64** MiB vs 512 | TPS **−2.77%** | **OFF** |
| Allocator | expandable **OFF** | TPS **−3.99%**, accept↓ | keep **ON** |
| ACC probe | `PROBE=1` serve | ≈ **−10%** soak | analysis only |
| PR #54282 walk | `IS_DRAFTING` in overlay | soak ~17–23 / accept ~0.36 | **DISCARD** |
| FlashInfer SM121 native build | rebuild path | ~**32 tok/s** class worsening (retained finding) | **NOT applied** |
| CUBLAS_WORKSPACE_CONFIG | env tuning | no meaningful gain / some worse | **NOT applied** |
| SM121 native kernel chase | various | not production | **NOT applied** |
| Marlin / BF16 “integration” rewrite | hot-path copy removal ideas | not A/B’d as KEEP; skipped after longrun | **NOT applied** |
| TOP_K=48 | selector | no meaningful accept gain | stay **32** |

Evidence roots: `evidence/spin-wait/`, `evidence/logits-budget/`, `evidence/expandable-segments/`, `evidence/clocks/`, `evidence/longrun-20260912/`, `benchmarks/pr54282-RESULTS.md`.

---

## 5. KEEP micro-opt (small positive)

| Change | Δ | Notes |
|--------|---|-------|
| Sparse indexer workspace reuse for `pool_topk` | **+0.70%** TPS | Landed on `main`; accept flat |
| Clock lock 2400 | ~+0.4% vs boost + **much less drift** | KEEP for stability |

---

## 6. Policy

1. Do not enable any row in §4 on Golden.
2. Do not invent unmeasured speedups in docs.
3. Next real headroom: **NVFP4 lane** or **W4A16-aligned draft** (out of pure runtime-patch scope).
