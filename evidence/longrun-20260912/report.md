# Phase A — longrun TPS degradation (2026-09-12)

## Workload

- 60 min continuous soak (`soak_timed.py`, temperature=0, max_tokens=512)
- Collectors: nvidia-smi @10s, `free`, `docker stats`, plus later `vmstat`/`mpstat`/`pidstat`
- Code path: **pre-Phase-B** (process started before workspace-reuse patch; import cached)

## Results

| Metric | Value |
|--------|-------|
| runs | 225 |
| decode median | **33.50 tok/s** |
| decode mean | 32.90 tok/s |
| decode min/max | 25.8 / 44.8 |
| decode_drift_pct (harness) | 6.95% |
| Q1→Q4 median TPS | 32.95 → 32.47 (**−1.46%**) |
| accept (cumulative /metrics) | **0.400** (86081/215005) |
| SM clock | median **2392 MHz** (2385–2398) — locked band |
| GPU temp | 47→63°C (Δ+16, mean 61.7) |
| GPU util mean | ~93.7% |
| docker mem | 9559 → 7757 MiB (**decreased**) |
| docker CPU mean | ~304% |

## Diagnosis

| Signal | Observation | Implication |
|--------|-------------|-------------|
| GPU clock ↓ | **No** (stable ~2392) | No further clock patch; Golden 2400 lock OK |
| memory ↑ | **No** (docker mem ↓) | Not a clear UM growth smoking gun in this 60m soak |
| CPU ↑ | High but expected under TP=2 serve | Not enough alone to force sync surgery |
| GPU util high + mild TPS drift | Yes | Runtime/kernel class possible but **weak** |
| acceptance ↓ | Soft vs Phase0 ~0.416 (longrun 0.400) | Report only — **no TOP_K retune** |

## Decision for later phases

- **Do not** start Phase F (DFlash sync) or Phase G (Marlin/BF16 integration) unless B–D fail and a stronger CPU/sync signal appears.
- **Proceed** Phase B (workspace reuse — checklist candidate), C (logits 64 MiB env), D (expandable OFF vs ON).
- Aggressive patches **not** justified by this longrun alone.

## Artifacts

- `tps/soak-60m.json`, `tps/soak-60m.log`
- `smi/smi.csv`
- `sys/free.txt`, `vmstat.txt`, `mpstat.txt`, `pidstat.txt`
- `docker/stats.txt`
