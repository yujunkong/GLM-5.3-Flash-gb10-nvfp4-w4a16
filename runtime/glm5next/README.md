# glm5next overlays for SERVE_LANE

- `model.pristine.py` — stock from image `glm53-spark:2x-sm121` (no W4A16
  compressed-tensors BF16 force). Used when `SERVE_LANE=modelopt`
  (`axiomofmind/GLM-5.3-Flash-W4A16-NVFP4`).

Golden W4A16 overlay remains at `patches/glm5next_model.py`.

Switch lane via `.env` `SERVE_LANE=golden|modelopt` then `bash scripts/up.sh`.
