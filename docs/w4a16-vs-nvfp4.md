# W4A16 vs NVFP4 — code-based comparison

Sources (no speculation):

- This repo: `/home/yujunkong/workspace/docker/GLM-5.3-Flash-gb10-nvfp4-w4a16`
- NVFP4 recipe: `/home/yujunkong/workspace/docker/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark`
- Image stock: `glm53-spark:2x-sm121` vLLM packages

## 1. Target / Draft structure

| | W4A16 (this repo) | NVFP4 (sibling recipe) |
|--|-------------------|-------------------------|
| Target weights | Hub `canada-quant/glm-5.3-w4a16-mtp` (default); optional `SERVE_LANE=modelopt` → `axiomofmind/GLM-5.3-Flash-W4A16-NVFP4` | Host path `RedHatAI/GLM-5.3-Flash-NVFP4` style (`MODEL_HOST_PATH`, refuse ModelOpt unless override) |
| Draft | Hub `incoai/GLM-5.3-Flash-DFlash2` | Local `/models/dflash2-draft` bind (same DFlash2 family) |
| Spec config | `method=dflash`, `num_speculative_tokens=7` | Same JSON shape in `launch-glm53-vllm-tp2-dflash2.sh` |
| Target model code | Overlay `patches/glm5next_model.py` → `vllm/models/glm5next/nvidia/model.py` | Day-0 image `glm5_next` + recipe overlays |

## 2. Acceptance path

Both use vLLM V1:

`DFlash2Speculator.propose` → `RejectionSampler._verify` → accept prefix until first mismatch.

| Piece | W4A16 | NVFP4 overlay |
|-------|-------|---------------|
| Speculator overlay | `patches/dflash2_speculator.py` | `overlay-dflash2/dflash2/speculator.py` |
| Walk default | `DFLASH_WALK_MODE=edge` | Stock edge walk (NVFP4 overlay = SM121 gumbel port; no ACC probe in that tree) |
| Probe | `DFLASH2_ACC_PROBE` (this repo) | Not present in NVFP4 overlay files reviewed |

Acceptance **algorithm** (greedy verify) is the shared vLLM rejection sampler; overlays do not replace sampler math.

## 3. Selector / Walker

| | W4A16 | NVFP4 |
|--|-------|-------|
| Candidate pool | `qwen3_dflash2.compute_candidates` top-k | Same module family |
| `selector_top_k` | Overridable via `DFLASH_SELECTOR_TOP_K` (`patches/qwen3_dflash2.py`) | Checkpoint `draft_config["selector_top_k"]` unless similarly patched |
| Edge walk | Triton `_selector_walk_kernel` + `candidate_selector` | Same pattern in NVFP4 `overlay-dflash2` |
| Unary walk | `DFLASH_WALK_MODE=unary` in this repo | Not in NVFP4 overlay reviewed |
| Gumbel | Stock import + `IS_DRAFTING=True` (#54282) after sync | Local SM121-ported `gumbel_noised_argmax` in overlay (older fork style) |

## 4. Sampler

| | Both |
|--|------|
| Verify | `vllm/.../rejection_sampler.py` (image) |
| Temp=0 benches | Plain argmax path (no Gumbel) |
| Draft vs target noise | `#54282` salts draft stream when `temp≠0` |

No separate W4A16-only rejection sampler file in this repo.

## 5. KV cache

| | W4A16 (`.env.example`) | NVFP4 launch script |
|--|------------------------|---------------------|
| dtype | `fp8_e4m3` | `fp8_e4m3` |
| pin bytes | `9663676416` (9 GiB) | `6442450944` (6 GiB) |
| `block_size` | `2304` | `2304` |
| Drafter group | `runtime/kv/kv_cache_utils.patched.py` (`DFLASH2-DRAFTER-GROUP`) | `overlay-dflash2/patch_glm5_drafter_group.py` (same problem class) |

## 6. Quantization

| | W4A16 | NVFP4 |
|--|-------|-------|
| Weight quant | W4A16-MTP (canada-quant) or ModelOpt W4A16_NVFP4 via `SERVE_LANE=modelopt` | NVFP4 (RedHatAI path; script rejects ModelOpt) |
| MoE backend | `marlin` | `marlin` |
| Dense MLP | Overlay forces BF16 path in `glm5next_model.py` (W4A16 KEEP) | NVFP4 native path on day-0 image |

Quantization difference is **weight format / checkpoint**, not a different DFlash accept formula in code.

## 7. Diff that is code-visible (qwen3_dflash2)

`diff` W4A16 vs NVFP4 `qwen3_dflash2.py`: W4A16 adds `_selector_top_k_from_config` + `DFLASH_SELECTOR_TOP_K` env override. NVFP4 uses checkpoint `selector_top_k` directly.

## 8. What this does **not** claim

- Does not claim NVFP4 accept % from docs apply to W4A16.
- Does not claim identical tokens / logits across quants.
- Measured W4A16 soak accept remains ~0.43–0.45 under Golden (see `docs/benchmark.md`).
