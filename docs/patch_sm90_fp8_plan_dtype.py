#!/usr/bin/env python3
"""Map fp8 MLA storage dtype for FlashInfer SM90 plan (uint8 -> float8_e4m3fn).

Upstream Builder passes ``kv_cache_spec.dtype`` (uint8 for fp8_e4m3 packed
storage) into ``_SM90State`` / ``wrapper.plan(kv_data_type=...)``. FlashInfer
rejects uint8; it accepts float16/bfloat16/float8_e4m3fn.

Radixark sm121-v11-dflash2 already maps:
  ``torch.float8_e4m3fn if use_fp8_kv_cache else torch.bfloat16``.

This patch applies the same mapping on the upstream Builder path.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

P = Path(
    os.environ.get(
        "GLM53_SM90_MLA_PY",
        "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm90.py",
    )
)
MARK = "# [glm53-sm90-fp8-plan-dtype]"

OLD = """        self.state = _SM90State(
            device,
            impl.num_heads,
            kv_cache_spec.dtype,
            vllm_config.scheduler_config.max_num_batched_tokens,
            topk_indices_buffer.shape[1],
            kv_lora_rank=impl.kv_lora_rank,
            qk_rope_head_dim=impl.qk_rope_head_dim,
            sm_scale=impl.scale,
        )
"""

NEW = """        # """ + MARK + """ fp8 packed storage is uint8; FlashInfer plan needs
        # float8_e4m3fn (same mapping as radixark SM90 Impl).
        plan_kv_dtype = (
            torch.float8_e4m3fn if impl.use_fp8_kv_cache else torch.bfloat16
        )
        self.state = _SM90State(
            device,
            impl.num_heads,
            plan_kv_dtype,
            vllm_config.scheduler_config.max_num_batched_tokens,
            topk_indices_buffer.shape[1],
            kv_lora_rank=impl.kv_lora_rank,
            qk_rope_head_dim=impl.qk_rope_head_dim,
            sm_scale=impl.scale,
        )
"""


def main() -> int:
    if not P.is_file():
        raise SystemExit(f"missing {P}")
    text = P.read_text()
    if MARK in text:
        print(f"{P.name}: {MARK} already present — skipping")
        return 0
    n = text.count(OLD)
    if n != 1:
        raise SystemExit(f"{P}: expected one SM90State ctor target, found {n}")
    P.write_text(text.replace(OLD, NEW, 1))
    print(f"patched {P.name} ({MARK})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
