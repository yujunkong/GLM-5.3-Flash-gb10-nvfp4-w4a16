# Benchmark

Canonical knobs: [`golden-configuration.md`](golden-configuration.md) · Narrative: [`performance.md`](performance.md).

## Goals (target)

| Metric | Target | Gate |
|--------|--------|------|
| Decode (soak / single-stream C1-class) | **40–50 tok/s** | warm, temp=0, code prompts |
| Spec accept (`/metrics` accepted÷drafted) | **50–70%** | soak cumulative, PROBE=0 |
| Concurrent C6 aggregate | stretch — hold ≥~80 tok/s if possible | not the primary gate |

## Current Golden band (measured, 2× Spark)

Stack: `glm53-spark:2x-sm121`, W4A16 + DFlash2, TOP_K=32 / WALK=edge / K=7 / PROBE=0 / clocks **2400**.

| Metric | Production Golden band | Example suite |
|--------|------------------------|---------------|
| Soak decode | **~34.7–35.5 tok/s** | baseline med **35.87** / mean **34.54**; checklist-final **35.09** |
| Accept | **~0.43–0.44** | **0.430–0.448** across suites |
| Longrun 60m | med **33.50**, accept **0.400** | continuous load floor |
| C6 | **~81–87** | concurrent OK |

## Current (suite table excerpt)

| Metric | Baseline `upstream-baseline-clocks2400` | Post-debug `golden-post-acc-debug` | vs goal |
|--------|----------------------------------------|--------------------------------------|---------|
| Soak decode median tok/s | **35.87** | 32.55 (3-run noise) | **미달** (목표 40–50) |
| Soak decode mean tok/s | 34.54 | 33.18 | **미달** |
| `/metrics` accept | **0.430** | **0.448** | **미달** (목표 0.50–0.70) |
| C1 agg tok/s | 33.5 | 44.7 | C1 한 점이라 변동 큼 |
| C2 agg tok/s | 43.9 | 46.0 | 참고 |
| C6 agg tok/s | 81.0 | **82.1** | concurrent OK |
| P1 (391 + Tokyo) | OK | OK | OK |

상세 원본: `benchmarks/RESULTS.md`, `benchmarks/golden-post-acc-debug/`, `benchmarks/upstream-baseline-clocks2400/`.

## Accept-ceiling diag (PROBE=1, temporary)

`benchmarks/accept-ceiling-diag-20260911-224401/`

| First-reject | Share |
|--------------|-------|
| A (not in lm_head top-k) | **12.9%** |
| B (in pool, walk miss) | **87.1%** |
| C (walk hit, verify reject) | **0%** |
| of B → **B2** (unary also miss) | **~90%** |

→ Soak accept 천장 ≈ **0.43–0.45**. 서빙 노브만으로 **0.50+ 불가**.

## Policy

- Warmup 1회 제외
- 미측정 수치 기재 금지
- 운영 벤치는 `DFLASH2_ACC_PROBE=0`
- 재실행: `bash bench/bench_config.sh <label>`

## Results table (dated)

| Date | Label | Soak med | Accept | C1 | C2 | C6 | Notes |
|------|-------|----------|--------|----|----|-----|-------|
| 2026-09-11 | upstream-baseline-clocks2400 | 35.87 | 0.430 | 33.5 | 43.9 | 81.0 | first Golden baseline |
| 2026-09-11 | golden-post-acc-debug | 32.55 | 0.448 | 44.7 | 46.0 | 82.1 | after debug feature; PROBE=0 |
| 2026-09-11 | accept-ceiling-diag | — | reject mean 0.455 | — | — | — | PROBE=1 A/B/C |
| 2026-09-12 | pr54282-is-drafting | **17.16** | **0.364** | 27.7 | 36.5 | 57.7 | #54282 stock import — **DISCARD** |
| 2026-09-12 | pr54282-local-gumbel | **23.44** | **0.374** | 22.2 | 39.2 | 56.0 | #54282 local salt — **DISCARD** |
| 2026-09-12 | control-pre54282 | **36.10** | **0.419** | — | — | — | pre-#54282 restore OK |
| 2026-09-12 | clocks A stock 10m | 34.21 | 0.418 | — | — | — | evidence/clocks |
| 2026-09-12 | clocks B 2400 10m | 34.33 | 0.417 | — | — | — | KEEP lock (stability) |
| 2026-09-12 | spinwait 0.002 20m | 33.77 | 0.415 | — | — | — | **REVERT** |
| 2026-09-12 | checklist-final-spinwait-2400 | **35.09** | **0.421** | 38.1 | 47.2 | **87.2** | then spinwait reverted |
| 2026-09-12 | longrun-60m | **33.50** | 0.400 | — | — | — | evidence/longrun |
| 2026-09-12 | logits-64 | −2.77% vs 512 | +0.005 | — | — | — | **REVERT** |
| 2026-09-12 | expandable-OFF | −3.99% | −0.014 | — | — | — | **REVERT** (keep ON) |
| 2026-09-12 | sparse-reuse | +0.70% | flat | — | — | — | **KEEP** |
