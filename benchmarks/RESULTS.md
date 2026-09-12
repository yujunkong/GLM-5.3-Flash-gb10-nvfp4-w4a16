# Benchmark results

**Version:** v1.0.0-golden  
Goals: **decode 40–50 tok/s**, **accept 50–70%** (aspirational).  
Golden: `PROBE=0`, `TOP_K=32`, `WALK=edge`, `TOKENS=7`, clocks 2400, expandable ON, logits **512**, pristine spin-wait.

Canonical: [`docs/golden-configuration.md`](../docs/golden-configuration.md) · [`docs/performance.md`](../docs/performance.md)

## Suites

| Suite | Path | Notes |
|-------|------|--------|
| Baseline | `upstream-baseline-clocks2400/` | first Golden |
| Post debug feature | `golden-post-acc-debug/` | PROBE=0 |
| Overhead smoke | `probe0-overhead-post-debug/` | soak-only |
| Accept diag | `accept-ceiling-diag-20260911-224401/` | PROBE=1 |
| 2026-09-12 opt cycle | `../evidence/*-20260912/` | checklist2 A/B + longrun |

## vs goals

| Metric | Goal | Baseline | Post-debug | 2026-09-12 | Status |
|--------|------|----------|------------|------------|--------|
| Soak decode median tok/s | 40–50 | **35.87** | 32.55* | band **34.7–35.5**; longrun **33.5** | **FAIL** |
| Soak decode mean tok/s | 40–50 | 34.54 | 33.18 | longrun **32.9** | **FAIL** |
| `/metrics` accept | 0.50–0.70 | **0.430** | **0.448** | **~0.43–0.44** | **FAIL** |
| C6 agg tok/s | (stretch) | 81.0 | **82.1** | up to **87.2** | OK concurrent |
| P1 | correct | OK | OK | OK | OK |

\*3-run soak median noisy. Golden operating band cites short/medium soaks (~34.7–35.5).

## 2026-09-12 checklist2 A/B

| Candidate | Δ TPS | Accept | Verdict |
|-----------|-------|--------|---------|
| sparse indexer buffer reuse | **+0.70%** | flat | **KEEP** on `main` |
| `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=64` | −2.77% | +0.005 | **REVERT** (keep 512) |
| `expandable_segments` OFF | −3.99% | −0.014 | **REVERT** (keep ON) |
| spin-wait 0.002 | −1.63% | flat | **REVERT** (pristine) |
| top-k fallback rewrite | — | — | PATCH=NONE |

Conclusion: **W4A16 code/allocator patch headroom exhausted**. Next: NVFP4 lane or draft align.

## Accept-ceiling (PROBE=1)

| Split | % |
|-------|---|
| A | 12.9 |
| B | **87.1** |
| C | 0 |
| B2 / B | **~90** |

→ 노브로 accept 0.50+ 불가.

## Re-run

```bash
bash scripts/up.sh   # DFLASH2_ACC_PROBE=0
bash bench/bench_config.sh <label>
bash bench/run_accept_ceiling_diag.sh   # then restore PROBE=0
```
