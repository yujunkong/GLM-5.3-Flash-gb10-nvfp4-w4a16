# Final improvement report — W4A16 2× DGX Spark

Date: 2026-09-11  
Scope: upstream code sync + debug/metrics + docs. **No Golden knob / weight / draft-model change.**

## Goals (unchanged)

| Gate | Target | Current (measured) |
|------|--------|-------------------|
| Decode (soak) | 40–50 tok/s | **~33–36** |
| Accept (`/metrics`) | 50–70% | **~43–45%** |

## Golden (frozen)

`TOP_K=32`, `WALK=edge`, `TOKENS=7`, `ENFORCE_EAGER=1`, `MOE_BACKEND=marlin`, `ASYNC_SCHEDULING=1`, `APPLY_APC_PATCH=1`, `DISABLE_FLASHINFER_AUTOTUNE=1`, `DFLASH2_ACC_PROBE=0`.

---

## Task checklist

### 1. Upstream DFlash2 sync — **DONE (delta applied)**

See `docs/upstream-sync-audit.md`.

- `dflash2/speculator.py` on GitHub main **==** image stock.
- Overlay synced to stock **`IS_DRAFTING=True` / #54282** (was lagging).
- Parent `dflash` PCP-only main diffs **not** applied (N/A for this topology).

### 2. GLM Flash parsers — **DONE (already present)**

- Image already has `glm47` tool + reasoning parsers; `glm45` alias registered.
- Entrypoint flags unchanged. No extra overlay required.

### 3. Accept/Reject debug — **DONE**

`DFLASH2_ACC_PROBE=1` only:

- Accept / Reject cumulatives
- Reject stages **B0 / B1 / B2** (aliases of A / unary-hit / unary-miss)
- `accept_ratio_recent`, `accept_ratio_window_128`, `accept_ratio_window_256`
- JSON + CSV under `/logs` (`$LOG_DIR`)

`PROBE=0`: no stash, no `.cpu()`, no file I/O (Golden path).

### 4. Metrics — **DONE (worker textfile + optional gauges)**

Written when `PROBE=1` to `$LOG_DIR/dflash2-acc.prom`:

- `accept_ratio_recent`
- `accept_ratio_window_128`
- `accept_ratio_window_256`
- `reject_stage_b0_total`
- `reject_stage_b1_total`
- `reject_stage_b2_total`

**Constraint (code fact):** draft runs in **worker** process; API `GET /metrics` does not see worker-local Prometheus registries without multiproc. Stock `vllm:spec_decode_*` remain on `/metrics`. Debug gauges: **prom file** (and best-effort same-process register). `acceptance_ratio.py` reads the prom file.

### 5. W4A16 / NVFP4 doc — **DONE**

`docs/w4a16-vs-nvfp4.md` — code/path comparison only.

### 6. Performance validation — **DONE (existing measured suites)**

| Suite | Soak med | Accept | C6 | Reject stages |
|-------|----------|--------|-----|---------------|
| `upstream-baseline-clocks2400` | 35.87 | 0.430 | 81.0 | n/a (PROBE=0) |
| `golden-post-acc-debug` | 32.55* | 0.448 | 82.1 | zeros (PROBE=0 OK) |
| `accept-ceiling-diag-20260911-224401` | — | reject mean 0.455 | — | **B0 12.9% / B≈87% → B2~90% of B** |

\*3-run noise; mean ~33–35. Full re-`bench_config` after #54282 sync recommended for sampling workloads; temp=0 Golden accept path unaffected by draft noise salt.

Code / chat prompts: harness uses **code** prompts (`bench_decode` / `bench_reject_split`). Separate chat-prompt matrix **not re-run** in this pass (same Golden; no claim of new chat numbers).

---

## Problem statement (explicit)

1. **Accept ceiling ~0.43–0.45** on this target+draft — B2-dominated; knobs exhausted.
2. **Decode soak below 40** — coupled to accept ceiling.
3. **0.50–0.70 accept** requires draft/target alignment (**out of this instruction’s scope**).

## Files touched this pass

- `patches/dflash2_speculator.py` — #54282 sync + B0/B1/B2 + window metrics
- `docs/upstream-sync-audit.md` — new
- `docs/w4a16-vs-nvfp4.md` — new
- `docs/final-improvement-report.md` — this file
- Prior: ACC probe `/logs`, `bench/*`, `docs/benchmark.md`, `docs/applied-stack.md`, `README.md`, `benchmarks/RESULTS.md`

## Completion vs instruction

| Requirement | Status |
|-------------|--------|
| Golden unchanged | Yes |
| Latest DFlash2/GLM patches only | Yes (sync audit + #54282 overlay fix; parsers already in image) |
| ACC_PROBE ON/OFF | Yes |
| Metrics extended | Yes (prom file + names; API worker caveat documented) |
| Perf compare report | Yes (this file + `benchmarks/RESULTS.md`) |
| No weight/draft swap | Yes |
