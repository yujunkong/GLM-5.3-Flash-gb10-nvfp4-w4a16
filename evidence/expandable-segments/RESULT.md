# Phase D — expandable_segments A/B

## Config

| | `PYTORCH_CUDA_ALLOC_CONF` |
|--|--|
| ON (Golden) | `expandable_segments:True` |
| OFF | `garbage_collection_threshold:0.8` (no expandable) |

## Results (5 min soak)

| | decode median | accept |
|--|---------------|--------|
| ON | 33.312 | 0.4117 |
| OFF | 31.984 | 0.3980 |
| DELTA | **−3.99%** | **−0.0136** |

## Decision

**REVERT / KEEP Golden ON** — OFF fails both TPS (−2%+) and accept gates.
