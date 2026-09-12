# glm5next overlays for SERVE_LANE

- `model.pristine.py` — stock from image (no SupportsEagle3 on multimodal
  wrapper). **Not** used for `SERVE_LANE=modelopt` with DFlash2.
- ModelOpt + DFlash2 uses `patches/glm5next_model.py` (Eagle3 + BF16 ignore).

Switch: `.env` `SERVE_LANE=golden|modelopt` then `bash scripts/up.sh`.
