# Benchmark results

Goals: **decode 40–50 tok/s**, **accept 50–70%**.  
Golden: `PROBE=0`, `TOP_K=32`, `WALK=edge`, `TOKENS=7`, clocks 2400, expandable ON.

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
| Soak decode median tok/s | 40–50 | **35.87** | 32.55* | **33.5** (60m) / **33.6** (KEEP confirm) | **FAIL** |
| Soak decode mean tok/s | 40–50 | 34.54 | 33.18 | **32.9** (60m) | **FAIL** |
| `/metrics` accept | 0.50–0.70 | **0.430** | **0.448** | **0.40–0.42** | **FAIL** |
| C6 agg tok/s | (stretch) | 81.0 | **82.1** | (unchanged Golden) | OK concurrent |
| P1 | correct | OK | OK | OK | OK |

\*3-run soak median noisy; earlier overhead soak mean was **34.84**. Gate on mean + C6 + accept.

## 2026-09-12 checklist2 A/B

| Candidate | Δ TPS | Accept | Verdict |
|-----------|-------|--------|---------|
| sparse indexer buffer reuse | **+0.70%** | flat | **KEEP** on `main` |
| `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=64` | −2.77% | +0.005 | REVERT |
| `expandable_segments` OFF | −3.99% | −0.014 | REVERT (keep ON) |
| top-k fallback rewrite | — | — | PATCH=NONE |

Longrun 60m: median **33.50**, accept **0.400**, clock locked ~2392, no host/docker mem growth.  
Conclusion: **W4A16 code/allocator patch headroom exhausted** for this stack. Next: NVFP4 lane or draft align.

Details: `evidence/final/optimization-report-20260912.md`.

## Accept-ceiling (PROBE=1)

| Split | % |
|-------|---|
| A | 12.9 |
| B | **87.1** |
| C | 0 |
| B2 / B | **~90** |

→ 노브로 accept 0.50+ 불가. draft–target 정렬 필요.

## Re-run

```bash
bash scripts/up.sh   # DFLASH2_ACC_PROBE=0
bash bench/bench_config.sh <label>

# diag only (then restore PROBE=0)
bash bench/run_accept_ceiling_diag.sh
```

See `docs/benchmark.md`, `docs/applied-stack.md`.
