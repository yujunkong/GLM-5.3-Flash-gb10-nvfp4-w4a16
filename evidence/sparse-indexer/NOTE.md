# Phase B — sparse-indexer buffer reuse (prep)

## Investigation

- Hot-path per-step alloc when `index_kpool > 1`:
  - `torch.full((num_rows, select_k), -1, int32)` for `pool_topk` (prefill + decode)
  - `pool_topk.to(torch.int64)` for expand
- Top-k workspace already uses `current_workspace_manager().get_simultaneous`
- Conditions met: same-shape steady decode, engine-lifetime workspace, fill/-1 + copy semantics preserved

## Patch (branch `opt/sparse-indexer-buffer-reuse`)

- Helper `_workspace_kpool_topk_bufs` allocates int32 + int64 (+ optional radix WS) from workspace
- Replaces per-step `torch.full` / `.to(int64)`
- A/B deferred until Phase A longrun completes (no restart mid-soak)

## KEEP gate

TPS **+2%** and accept unchanged and no memory regression; else **REVERT**.
