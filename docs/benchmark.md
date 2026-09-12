# Benchmark

## Goals (target)

| Metric | Target | Gate |
|--------|--------|------|
| Decode (soak / single-stream C1-class) | **40–50 tok/s** | warm, temp=0, code prompts |
| Spec accept (`/metrics` accepted÷drafted) | **50–70%** | soak cumulative, PROBE=0 |
| Concurrent C6 aggregate | stretch — hold ≥~80 tok/s if possible | not the primary gate |

## Current (measured 2026-09-11, 2× Spark, Golden)

Stack: `glm53-spark:2x-sm121`, `canada-quant/glm-5.3-w4a16-mtp` + `incoai/GLM-5.3-Flash-DFlash2`,  
`TOP_K=32` / `WALK=edge` / `TOKENS=7` / `DFLASH2_ACC_PROBE=0` / clocks **2400 MHz**.

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
