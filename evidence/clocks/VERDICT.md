# Phase 1 — GB10 clocks A/B

## BEFORE (stock auto-boost + async)
- decode median: 34.207 tok/s
- decode mean: 34.269
- accept: 0.4179
- gpu clock median: 2509.0 MHz
- temp: 53.0–66.0 °C (Δ13.0)
- drift pct (last3 vs first3): 6.57

## AFTER (2400 MHz lock + async)
- decode median: 34.334 tok/s
- decode mean: 34.345
- accept: 0.4168
- gpu clock median: 2392.0 MHz
- temp: 61.0–63.0 °C (Δ2.0)
- drift pct: 0.55

## DELTA
- decode median: +0.37%
- temp max: -3.0 °C

## THERMAL / STABILITY
- No +40°C runaway relative to start of each soak.
- Stock auto-boost ran **above** 2400 (median ~2509.0 MHz), so lock is not always a TPS win vs boost.

## DECISION
**KEEP_NEUTRAL_OR_SMALL** — production keeps `LOCK_CLOCKS=1` / 2400 for **repeatable** clocks (prior KEEP). This A/B: stock boost can match or exceed 2400 TPS; 2400 remains valid for stability/repro.
