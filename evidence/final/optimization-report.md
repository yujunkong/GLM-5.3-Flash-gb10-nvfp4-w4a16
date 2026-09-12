# Optimization report — checklist 20260912

Date: 2026-09-12  
Stack: W4A16 + DFlash2 TP=2, Golden knobs unchanged.  
Source checklist: `docs/20260912_checklist.md`

## Targets

| Gate | Target | Result |
|------|--------|--------|
| Decode | ≥40 tok/s | **FAIL** (~34–36 soak) |
| Accept | ≥0.70 | **FAIL** (~0.42) |
| TPS drift | no sustained drop | **improved with 2400 lock** |

→ Checklist **Phase 7**: stop further W4A16 knob chasing for 40/0.70; NVFP4 is a separate track.

---

## Phase 1 — GB10 clocks (stock vs 2400)

Evidence: `evidence/clocks/`

| | Stock auto-boost | 2400 lock |
|--|------------------|-----------|
| Soak median tok/s | 34.21 | **34.33** (+0.37%) |
| Accept | 0.418 | 0.417 |
| GPU clock median | **2509** MHz | 2392 MHz |
| Temp range | 53–66 °C (Δ13) | 61–63 °C (**Δ2**) |
| Drift (last3−first3)/first3 | **+6.57%** | **+0.55%** |

### BEFORE / AFTER / DELTA / THERMAL / STABILITY / DECISION

- **BEFORE:** stock + async  
- **AFTER:** 2400 + async  
- **DELTA:** +0.37% median (**< 3% adopt-for-speed gate**)  
- **THERMAL:** cooler max (−3 °C), much tighter band  
- **STABILITY:** drift much better under lock  
- **DECISION: KEEP** `LOCK_CLOCKS=1` / 2400 for **reproducibility + stability** (not raw TPS vs boost)

Tools: `scripts/clocks.sh {set|reset|status}` (+ root `clocks.sh`).

---

## Phase 2 — spin-wait (`busy_loop_s` 1.0 → 0.002)

Evidence: `evidence/spin-wait/soak-20m.json` (20 min, clocks 2400)

| | 2400 only (10m) | + spinwait 0.002 (20m) |
|--|-----------------|-------------------------|
| Soak median | 34.33 | **33.77** (−1.63%) |
| Drift | 0.55% | **3.88%** (worse) |
| Temp max | 63 | 60 |
| Accept | 0.417 | 0.415 |

- **DECISION: REVERT** — patched overlay does not reduce drift / TPS on this stack.  
- Production `.env`: `SPINWAIT_PATCH_HOST=../runtime/spinwait/shm_broadcast.pristine.py`  
- Patched file kept at `runtime/spinwait/shm_broadcast.patched.py` for retest only.

---

## Phase 3 — worker memory cap / trace

- Image already bakes `worker_memory_cap.py` + `spark_mem_trace.py`.  
- `TORCH_MEM_FRACTION` **unset** → cap inactive (upstream-like).  
- Snapshot: `evidence/memory/snapshot-before.txt` (no runaway correlation forced this run).  
- **DECISION: no new cap value** (checklist: measure first; no blind copy of 4× GX10 numbers).

---

## Phase 4 — gb10_topk_fallback

- Already applied in image Dockerfile (`gb10_topk_fallback.py`).  
- **DECISION: no change.** `DFLASH_SELECTOR_TOP_K=32` unchanged.

---

## Phase 5 — KEEP knobs (unchanged)

`TOP_K=32`, `WALK=edge`, `TOKENS=7`, `PROBE=0`, `marlin`, `ENFORCE_EAGER=1`, `ASYNC=1`, `no autotune`, `APC=1`, `SM121_MLA=0`, `fp8_e4m3`, `block_size=2304`.

Forbidden items (graphs / EP / BF16 KV / block-size / TOP_K retune / rebuilds): **not run**.

---

## Phase 6 — final bench (spinwait still mounted during that run)

Label: `benchmarks/checklist-final-spinwait-2400/`

| Metric | Value |
|--------|-------|
| Soak median / mean | **35.09 / 35.98** |
| Accept | **0.421** |
| C1 / C2 / C6 | 38.1 / 47.2 / **87.2** |
| P1 | 391 + Tokyo OK |

After reporting, spinwait host path reverted to **pristine** in `.env` (restart `up.sh` to unload patched file from next boot).

---

## Adopted for production

| Change | Status |
|--------|--------|
| 2400 MHz lock | **KEEP** |
| async scheduling | already on |
| spin-wait 0.002 | **REVERT** |
| memory cap fraction | unset (no change) |
| gb10 topk | already in image |

## Not met

decode ≥40, accept ≥0.70 — require model/draft track (NVFP4 / retrain), not more Golden knobs.

## Next (checklist Phase 7)

Separate branch: NVFP4 + DFlash2 vs this W4A16 Golden under **identical** 512-token harness.
