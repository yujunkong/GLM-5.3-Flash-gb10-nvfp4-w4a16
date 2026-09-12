# Upstream sync audit (DFlash2 / GLM Flash)

Date: 2026-09-11  
Image: `glm53-spark:2x-sm121` → vLLM `0.28.1rc1.dev580+g385dce36b`  
Compare: GitHub `vllm-project/vllm` **main**, local clone, NVFP4 overlay repo.

Golden knobs were **not** changed.

## Task 1 — Speculative / DFlash2

| File | Image stock vs GitHub main | Action |
|------|----------------------------|--------|
| `.../dflash2/speculator.py` | **Identical** (217 lines) | No stock replace |
| `.../dflash/speculator.py` | Main adds PCP (`prefill_context_parallel_size`, `gather_block_tables`) | **Not applied** — this stack uses CP=1 / no PCP path; risk without gain |
| `.../rejection_sampler.py` | Checked | No accept-path delta required for Golden |
| `.../sample/gumbel.py` | Includes `#54282` `IS_DRAFTING` salt | Already in image |

### Applied to our overlay `patches/dflash2_speculator.py`

Our mount previously **forked** an older local `gumbel_noised_argmax` **without** `IS_DRAFTING` (pre-#54282).

**Synced (code-only):**

1. Import stock `gumbel_noised_argmax` from `vllm.v1.worker.gpu.sample.gumbel`.
2. Walk kernel uses `sample_pos = P-1` + `IS_DRAFTING=True` (matches image stock / #54282).
3. Kept: `DFLASH_WALK_MODE` edge/unary, `DFLASH_SELECTOR_TOP_K` override, ACC probe (PROBE=0 = no-op).

**Note:** Golden benches use `temperature=0` → Gumbel branch idle; soak accept path unchanged by #54282. Sampling (`temp≠0`) now matches upstream draft noise salt.

## Task 2 — GLM tokenizer / parsers / SM121

| Component | Image | GitHub main | Action |
|-----------|-------|-------------|--------|
| `tool_parsers/glm47_moe_tool_parser.py` | Present | Present | **None** — already used (`--tool-call-parser glm47`) |
| `reasoning/glm47_moe_reasoning_parser.py` | Present | Present | **None** |
| `reasoning` alias `glm45` | Mapped in `__init__.py` | Same pattern | **None** — entrypoint keeps `--reasoning-parser glm45` |
| `parser/glm47_moe.py` | Present | Present | **None** |
| Chat template | Repo `chat_template_mm.jinja` | N/A | Keep mount |
| SM121 / GB10 | Image `image/patches/*` | Upstream 4× recipe | Already baked; no extra pull |

Open PR `#53906` (GLM-5.3-Flash model support) is **not merged** into a drop-in we can mount without weighing model/API risk; image already serves `canada-quant` W4A16 via overlays.

## Conclusion

- **DFlash2 core on main == image.** Only our overlay lagged on #54282 → **fixed**.
- **GLM parsers already in image**; API flags unchanged.
- Parent `dflash` PCP diffs **intentionally skipped**.
