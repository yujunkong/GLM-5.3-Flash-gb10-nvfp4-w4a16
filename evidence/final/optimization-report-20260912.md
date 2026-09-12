# Optimization report — 2026-09-12 (checklist2 cycle)

## 1. Baseline (실측)

| Source | decode median | accept |
|--------|---------------|--------|
| Phase 0 short soak | 33.58 | ~0.416 (band) |
| Phase A 60m longrun | 33.50 | 0.400 cumulative |
| Final Golden confirm (5m) | **34.07** | **0.424** | |

Tag: `glm53-w4a16-baseline-20260912`  
Image: `glm53-spark:2x-sm121`  
Golden knobs unchanged (TOP_K=32, WALK=edge, TOKENS=7, PROBE=0, expandable ON, logits default 512, clocks locked).

## 2. Phases performed / skipped

| Phase | Action |
|-------|--------|
| 0 | Repo/image/baseline snapshot — done |
| A | 60m instrumentation — done (`evidence/longrun-20260912/report.md`) |
| B | sparse workspace reuse A/B — **KEEP** (+0.70%, user: stack sub-1%) |
| C | logits 64 MiB A/B — **REVERT** |
| D | expandable OFF A/B — **REVERT** (keep ON) |
| E | Top-K fallback compare — **PATCH=NONE** |
| F | DFlash sync — **SKIP** (A: no clock↓/mem↑/CPU smoking gun) |
| G | Marlin/BF16 integration — **SKIP** (same) |
| H | No KEEP merges; production = baseline Golden |

Skipped checklist1: clocks / async / spinwait / memcap / TOP_K retune.

## 3. Candidate KEEP/REVERT

| Candidate | Δ TPS | Δ accept | Verdict |
|-----------|-------|----------|---------|
| sparse buffer reuse | +0.70% | +0.0002 | **KEEP** (user: land sub-1%) |
| logits 64 MiB | −2.77% | +0.0053 | **REVERT** |
| expandable OFF | −3.99% | −0.0136 | **REVERT** (keep ON) |
| top-k fallback | — | — | **NONE** |

## 4. Final vs baseline

- Production: Golden knobs + **sparse indexer workspace reuse** (`main` cherry-pick).
- C/D remain REVERT; F/G skipped.
- Post-KEEP confirm: restart + short soak after land.

## 5. Long-run stability

- Clock locked ~2392 MHz; temp 47–63°C; docker mem did not grow.
- Mild Q1→Q4 TPS −1.5%; harness drift ~7%.
- No allocator/thermal patch indicated beyond existing Golden.

## 6. Remaining limits

- Accept ceiling ~0.40–0.45 on this W4A16 + DFlash2 draft; **code patches alone will not deliver 0.50+**.
- Draft–target mismatch / B2-dominated rejects remain P0 outside this cycle.
- 40 tok/s is aspirational; do not break quality/Golden to chase it.

## 7. Next recommendations

1. **Stay on W4A16 Golden** for production this cycle.
2. Consider **NVFP4 lane** or **W4A16-aligned draft** for accept/TPS headroom (out of code-patch scope).
3. Do not re-A/B clocks/async/spinwait/TOP_K without new evidence.

## Artifacts

- `evidence/baseline-20260912/`
- `evidence/longrun-20260912/`
- `evidence/sparse-indexer/`
- `evidence/logits-budget/`
- `evidence/expandable-segments/`
- `evidence/topk-fallback/`
- `evidence/final/confirm-soak-20260912.*`
