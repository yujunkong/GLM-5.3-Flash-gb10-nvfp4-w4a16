# Benchmark results

Goals: **decode 40–50 tok/s**, **accept 50–70%**.  
Golden: `PROBE=0`, `TOP_K=32`, `WALK=edge`, `TOKENS=7`, clocks 2400.

## Suites

| Suite | Path | Notes |
|-------|------|--------|
| Baseline | `upstream-baseline-clocks2400/` | first Golden |
| Post debug feature | `golden-post-acc-debug/` | PROBE=0 |
| Overhead smoke | `probe0-overhead-post-debug/` | soak-only |
| Accept diag | `accept-ceiling-diag-20260911-224401/` | PROBE=1 |

## vs goals

| Metric | Goal | Baseline | Post-debug | Status |
|--------|------|----------|------------|--------|
| Soak decode median tok/s | 40–50 | **35.87** | 32.55* | **FAIL** |
| Soak decode mean tok/s | 40–50 | 34.54 | 33.18 | **FAIL** |
| `/metrics` accept | 0.50–0.70 | **0.430** | **0.448** | **FAIL** |
| C6 agg tok/s | (stretch) | 81.0 | **82.1** | OK concurrent |
| P1 | correct | OK | OK | OK |

\*3-run soak median noisy; earlier overhead soak mean was **34.84** (+0.9% vs baseline). Gate on mean + C6 + accept.

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
