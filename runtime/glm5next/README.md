# glm5next overlays for SERVE_LANE

- `patches/glm5next_model.py` — Golden W4A16: Eagle3 + dense/shared `quant_config=None`.
- `patches/glm5next_model.modelopt.py` — ModelOpt lane: Eagle3 + stock `quant_config`
  for dense/shared (ignore list keeps them BF16). MLA stays BF16 (`quant_config=None`).
- `model.pristine.py` — image stock reference (no `SupportsEagle3` on multimodal wrapper).

Switch: `.env` `SERVE_LANE=golden|modelopt` then `bash scripts/up.sh`.
