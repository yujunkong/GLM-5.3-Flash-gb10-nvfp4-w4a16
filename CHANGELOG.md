# Changelog

## SERVE_LANE=modelopt — 2026-09-12

- Optional checkpoint lane (same image, no rebuild): `SERVE_LANE=modelopt` →
  `axiomofmind/GLM-5.3-Flash-W4A16-NVFP4`.
- DFlash2 requires Eagle3: modelopt uses `patches/glm5next_model.py` (not image
  stock `runtime/glm5next/model.pristine.py`, which lacks `SupportsEagle3`).
- Switch via `.env` + `bash scripts/up.sh`; materializes `.env.lane` for worker.
- Default production remains `SERVE_LANE=golden` / `v1.0.0-golden`.

## v1.0.0-golden — 2026-09-12

- **W4A16 + DFlash2 production configuration finalized** for 2× DGX Spark / GB10 / SM121 (TP=2).
- Golden knobs frozen: K=7, TOP_K=32, WALK=edge, PROBE=0, marlin, eager, async, FP8 KV, clocks 2400, expandable ON, logits 512, pristine spin-wait, `GLM53_SM121_MLA=0`, `APPLY_GATE_LINEAR=0`.
- Performance experiments completed (clocks, spin-wait, logits 64, expandable OFF, sparse buffer reuse, accept ceiling, #54282).
- Rejected runtime patches documented (do not enable): spin-wait 0.002, logits 64, expandable OFF, FlashInfer SM121 rebuild chase, CUBLAS workspace tuning, etc.
- Stability configuration finalized; optional `scripts/tps_drift_logger.py` for ops telemetry.
- Docs: `docs/golden-configuration.md`, `docs/performance.md`, README / RESULTS aligned.

## Earlier

See git history (`docs/final-improvement-report.md`, `evidence/final/`) for pre-tag work (ACC probe, APC/KV/SM90 overlays, image bring-up).
