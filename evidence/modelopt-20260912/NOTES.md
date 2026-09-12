# ModelOpt speed A/B — 2026-09-12

## Baseline (current golden overlays + marlin)

- Model: `axiomofmind/GLM-5.3-Flash-W4A16-NVFP4`
- glm5next: `patches/glm5next_model.py` (Eagle3 + BF16 force)
- MoE: marlin (NVFP4 weight-only on GB10; no native FP4)
- Soak 5m: med **21.84** tok/s, mean 22.86, accept **0.386**
- Golden canada-quant band for comparison: ~34.7–35.5 tok/s, accept ~0.43–0.44

## Experiment A (this redeploy)

- glm5next: `patches/glm5next_model.modelopt.py` (Eagle3 + quant_config for dense/shared)
- `APPLY_GATE_LINEAR=1` (DSV3 / (6144,256) shapes)
- `MOE_BACKEND=flashinfer_b12x` (SM121 NVFP4 opt-in; excluded from auto-select)

If boot fails on b12x → fall back `MOE_BACKEND=marlin` + gate_linear only.

## Attempt flashinfer_b12x
- FAIL: swiglu_limit=10.0 not supported by flashinfer_b12x
- Retry: MOE_BACKEND=b12x (+ gate_linear + modelopt glm5next)

## Attempt b12x
- FAIL: kernel does not support current device cuda (SM121)
- Retry: MOE_BACKEND=flashinfer_cutedsl

## Attempt flashinfer_cutedsl
- FAIL: kernel does not support current device cuda
- Fall back: MOE_BACKEND=marlin + APPLY_GATE_LINEAR=1 + modelopt glm5next

## Results (5m soak_timed, temp=0, thinking off)

| Config | med tok/s | mean | accept | notes |
|--------|-----------|------|--------|-------|
| baseline (golden glm5next + marlin, ~2h uptime) | **21.84** | 22.86 | **0.386** | long-uptime |
| exp (modelopt glm5next + gate_linear + marlin, fresh) | **31.59** | 32.45 | **0.429** | +44.6% med / +0.043 accept |

### Backend attempts (all FAIL on SM121 except marlin)
- `flashinfer_b12x`: swiglu_limit not supported
- `b12x`: kernel does not support current device cuda
- `flashinfer_cutedsl`: kernel does not support current device cuda

### KEEP (modelopt lane, tentative)
- `patches/glm5next_model.modelopt.py` (Eagle3 + quant_config dense/shared)
- `APPLY_GATE_LINEAR=1`
- `MOE_BACKEND=marlin` (only working NVFP4 MoE path on GB10)

Caveat: baseline was long-uptime; isolate gate vs overlay in a follow-up if needed.
Golden canada-quant band remains ~34.7–35.5 / accept ~0.43–0.44.
