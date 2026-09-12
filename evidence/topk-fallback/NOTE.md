# Phase E — Top-K fallback code comparison (2026-09-12)

## Finding

**`PATCH = NONE`**

## Why

- Production already mounts `patches/sparse_attn_indexer_kpool.py` with SM121 routing:
  - `multi_processor_count < 78` → `top_k_per_row_decode` (not `persistent_topk`)
  - Comment documents GB10 48 SM / 99KB smem FilteredTopK failure mode
- Image + overlay already implement the needed GB10 top-k fallback path
- No safe additional top-k fallback patch without changing `DFLASH_SELECTOR_TOP_K` or duplicating existing routing

## Action

No code change for Phase E. Do not re-implement top-k fallback.
