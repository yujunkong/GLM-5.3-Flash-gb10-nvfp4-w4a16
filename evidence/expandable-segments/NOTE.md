# Phase D — expandable_segments (prep)

## Current (Golden)

`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (compose default + live container).

## A/B plan (after longrun)

- A = current ON (baseline / Golden)
- B = OFF via compose override for one restart cycle only
- Metrics: TPS, accept, GPU/host memory, fragmentation signals
- If OFF worse or unstable → keep Golden ON (not a new KEEP)
- If OFF better by ≥+2% TPS with accept OK → document carefully; still prefer stability on GB10 UM
