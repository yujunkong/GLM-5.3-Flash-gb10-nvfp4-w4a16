# Benchmark — PR #54282 (`IS_DRAFTING`) A/B

Date: 2026-09-12  
Golden knobs unchanged. Clocks ~2400 MHz. `DFLASH2_ACC_PROBE=0`.

## Variants

| Label | Walk / Gumbel | Path |
|-------|---------------|------|
| baseline | pre-#54282 (earlier Golden) | `benchmarks/upstream-baseline-clocks2400/` |
| pr54282-is-drafting | import stock `gumbel_noised_argmax` + `IS_DRAFTING=True` | `benchmarks/pr54282-is-drafting/` |
| pr54282-local-gumbel | local gumbel copy + `IS_DRAFTING=True` + salt | `benchmarks/pr54282-local-gumbel/` |
| control-pre54282 | local gumbel **without** `IS_DRAFTING` | `benchmarks/control-pre54282/` |

## Results

| Label | Soak med | Soak mean | `/metrics` accept | reject_split mean | C1 | C2 | C6 |
|-------|----------|-----------|-------------------|-------------------|----|----|-----|
| baseline | 35.87 | 34.54 | **0.430** | 0.470 | 33.5 | 43.9 | 81.0 |
| pr54282 (stock import) | **17.16** | 18.68 | **0.364** | **0.307** | 27.7 | 36.5 | 57.7 |
| pr54282 (local+salt) | **23.44** | 23.86 | **0.374** | 0.410 | 22.2 | 39.2 | 56.0 |
| control-pre54282 | **36.10** | 34.75 | **0.419** | **0.452** | — | — | — |

P1 (391 + Tokyo): OK on all runs.

## Verdict

**`MEASURED_DISCARD` for overlaying #54282 `IS_DRAFTING` into `patches/dflash2_speculator.py` on this W4A16 Golden path.**

- Applying `IS_DRAFTING=True` (stock import or local salt copy) **cuts soak ~35→17–23 tok/s** and **accept ~0.43→0.36–0.37**.
- Reverting walk/gumbel to **pre-#54282** restores soak **~36** / accept **~0.42** (baseline band).
- Image stock `gumbel.py` already has #54282; the **target** sampler uses it. The regression is specific to wiring `IS_DRAFTING` into **our DFlash2 walk overlay** (temp=0 benches still regress — Triton/overlay interaction, not only sampling noise).

## Production decision

- Keep overlay walk on **pre-#54282** gumbel (current tree after control restore).
- Do **not** leave `IS_DRAFTING=True` in the mounted `dflash2_speculator.py` until a fixed overlay is re-validated.
- Probe / B0–B2 / window metrics code may remain; `PROBE=0` stays default.
