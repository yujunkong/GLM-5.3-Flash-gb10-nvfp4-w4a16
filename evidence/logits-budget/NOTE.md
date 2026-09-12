# Phase C — logits budget 64 MiB (prep)

## Path exists?

**Yes.** `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB` (default **512**).

## This stack

- Env currently unset → default 512
- kpool overlay uses it mainly as **prefill_cap** in profiling sentinel:
  `max(decode_worst_case, MAX_LOGITS_MB)`
- With `MAX_MODEL_LEN=1048576`, decode sentinel usually **dominates** 512 MiB
- MLA indexer / QSA also chunk by this budget (if those paths hit)

## A/B plan (after longrun)

- Baseline: default 512 (Golden — do not permanently change)
- Experiment: compose/env override `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=64` only
- Measure soak TPS/accept + memory; long-context if feasible
- KEEP only if no TPS/accept regression and clear stability benefit; else REVERT
