# Phase B — sparse-indexer buffer reuse A/B

## Patch

Branch/`main` commit: workspace reuse for `pool_topk` int32 + int64 ids.

## Results (5 min soak, temp=0)

| | decode median | accept |
|--|---------------|--------|
| BEFORE (Golden kpool) | 33.312 | 0.4117 |
| AFTER (workspace reuse) | 33.544 | 0.4118 |
| DELTA | **+0.70%** | +0.0002 |

## Decision

**KEEP** (user override 2026-09-12: sub-1% gains stack; land even if &lt; +2%).

Accept unchanged; no memory/quality regression observed.
