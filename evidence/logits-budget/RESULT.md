# Phase C — logits budget 64 MiB A/B

## Config

| | `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB` |
|--|--|
| baseline | 512 (default / unset) |
| experiment | 64 (compose override only) |

Path is real (`envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB`); kpool overlay uses it as prefill_cap in profiling sentinel (decode worst-case often dominates at 1M ctx).

## Results (5 min soak)

| | decode median | accept |
|--|---------------|--------|
| 512 | 33.312 | 0.4117 |
| 64 | 32.390 | 0.4170 |
| DELTA | **−2.77%** | +0.0053 |

## Decision

**REVERT** — TPS drop exceeds −2% gate. Do not change Golden default (512 / unset).
No long-context stability win measured on this short soak; leave env unset.
