#!/usr/bin/env python3
"""Port radixark DFLASH2-DRAFTER-GROUP into upstream kv_cache_utils.py.

Root cause (verified against glm53-spark:2x-sm121 vs radixark sm121-v11-dflash2):

Upstream ``_get_kv_cache_groups_glm5_next`` requires every non-mamba/non-tail
spec to be ``MLAAttentionSpec``. DFlash2 adds plain ``SlidingWindowSpec``
drafter layers, so the GLM fast path returns None and
``unify_kv_cache_spec_page_size`` raises on ``indexer.k_cache`` (MLA cannot
pad to the drafter page).

Radixark already partitions ``type(v) is SlidingWindowSpec`` into a separate
drafter group (exact-fit slot-share or standalone tensors). This patch ports
that logic onto the upstream tree, using ``tokens_per_state`` (upstream) in
place of ``compress_ratio`` (radixark).

Fail-closed: any missing anchor aborts without writing a half-patched file.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

P = Path(
    os.environ.get(
        "GLM53_KV_CACHE_UTILS_PY",
        "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_utils.py",
    )
)
MARK = "# [glm53-dflash-kv-groups]"

# --- replacement bodies (tokens_per_state = upstream field) ----------------

GROUPS_NEW = '''def _get_kv_cache_groups_glm5_next(
    vllm_config: VllmConfig,
    kv_cache_spec: dict[str, KVCacheSpec],
) -> list[KVCacheGroupSpec] | None:
    """Build GLM-5.3-Flash groups with Mamba/MLA and tail/indexer aliasing.

    ''' + MARK + ''' DFlash2 adds plain SlidingWindowSpec drafter layers.
    Partition them out (exact type — KpoolTailSpec subclasses SlidingWindowSpec)
    so they do not disqualify the GLM fast path; append as one extra group.
    Port of radixark DFLASH2-DRAFTER-GROUP, adapted to tokens_per_state.
    """
    mamba_specs = {
        name: spec
        for name, spec in kv_cache_spec.items()
        if isinstance(spec, MambaSpec)
    }
    tail_specs = {
        name: spec
        for name, spec in kv_cache_spec.items()
        if isinstance(spec, KpoolTailSpec)
    }
    draft_specs = {
        name: spec
        for name, spec in kv_cache_spec.items()
        if type(spec) is SlidingWindowSpec
    }
    attn_specs = {
        name: spec
        for name, spec in kv_cache_spec.items()
        if not isinstance(spec, (MambaSpec, KpoolTailSpec))
        and type(spec) is not SlidingWindowSpec
    }
    if not mamba_specs or not all(
        type(spec) is MLAAttentionSpec for spec in attn_specs.values()
    ):
        return None

    mla_specs = cast(dict[str, MLAAttentionSpec], attn_specs)
    idx_pages = {
        spec.page_size_bytes for spec in mla_specs.values() if spec.tokens_per_state > 1
    }
    if not idx_pages:
        return None

    assert all(spec.page_size_padded is None for spec in mla_specs.values())
    assert len(idx_pages) == 1
    mla_names = [name for name, spec in mla_specs.items() if spec.tokens_per_state == 1]
    mla_pages = {mla_specs[name].page_size_bytes for name in mla_names}
    assert len(mla_pages) == 1
    mla_page = mla_pages.pop()
    uniform_spec = UniformTypeKVCacheSpecs.from_specs(attn_specs)
    assert uniform_spec is not None

    tail_group: KVCacheGroupSpec | None = None
    if tail_specs:
        idx_page = next(iter(idx_pages))
        padded_tail_specs: dict[str, KVCacheSpec] = {
            name: replace(spec, page_size_padded=idx_page)
            for name, spec in tail_specs.items()
        }
        tail_uniform = UniformTypeKVCacheSpecs.from_specs(padded_tail_specs)
        assert tail_uniform is not None
        tail_group = KVCacheGroupSpec(list(padded_tail_specs), tail_uniform)

    any_mamba = next(iter(mamba_specs.values()))
    assert all(spec == any_mamba for spec in mamba_specs.values())
    if any_mamba.real_page_size_bytes > mla_page:
        raise ValueError(
            f"the mamba state page ({any_mamba.real_page_size_bytes} bytes) "
            f"does not fit the MLA page ({mla_page} bytes); increase tensor "
            "parallelism or use a wider KV cache dtype"
        )
    padded_specs: dict[str, KVCacheSpec] = {
        name: replace(any_mamba, page_size_padded=mla_page) for name in mamba_specs
    }
    num_groups = _pp_balanced_mamba_group_count(
        vllm_config, list(mamba_specs), mla_names
    )
    if num_groups is None:
        raise ValueError(
            "a pipeline stage has mamba layers but no MLA layer to share "
            "slots with; realign the stage boundaries (VLLM_PP_LAYER_PARTITION)"
        )
    mamba_grouped_names: list[list[str]] = [[] for _ in range(num_groups)]
    for index, name in enumerate(mamba_specs):
        mamba_grouped_names[index % num_groups].append(name)

    # Drafter group: NEVER page_size_padded (strided view breaks under kernel
    # block splits). Exact-fit slot-shares MLA tensors; else standalone.
    draft_group: KVCacheGroupSpec | None = None
    if draft_specs:
        any_draft = next(iter(draft_specs.values()))
        assert all(spec == any_draft for spec in draft_specs.values()), (
            "drafter SlidingWindowSpec layers must share one spec"
        )
        draft_bytes_per_token = any_draft.page_size_bytes // any_draft.block_size
        mla_block = mla_specs[mla_names[0]].block_size
        fit_block = (
            mla_page // draft_bytes_per_token
            if mla_page % draft_bytes_per_token == 0
            else 0
        )
        if (
            fit_block
            and fit_block % 64 == 0
            and (fit_block % mla_block == 0 or mla_block % fit_block == 0)
            and len(draft_specs) <= len(mla_names)
        ):
            new_draft_specs: dict[str, KVCacheSpec] = {
                name: replace(spec, block_size=fit_block)
                for name, spec in draft_specs.items()
            }
        else:
            new_draft_specs = dict(draft_specs)
        draft_uniform = UniformTypeKVCacheSpecs.from_specs(new_draft_specs)
        assert draft_uniform is not None
        draft_group = KVCacheGroupSpec(list(new_draft_specs), draft_uniform)

    return (
        [KVCacheGroupSpec(list(attn_specs), uniform_spec)]
        + ([tail_group] if tail_group is not None else [])
        + create_kv_cache_group_specs(padded_specs, mamba_grouped_names)
        + ([draft_group] if draft_group is not None else [])
    )


'''

LAYOUT_NEW = '''def _glm5_next_tensor_layout(
    kv_cache_groups: list[KVCacheGroupSpec],
) -> (
    tuple[
        KVCacheGroupSpec,
        list[KVCacheGroupSpec],
        list[str],
        list[str],
        int,
        int,
        list[str],
        int,
        KVCacheGroupSpec | None,
    ]
    | None
):
    """Recognize GLM-5.3-Flash grouping after optional PP projection.

    ''' + MARK + ''' Returns draft_group as 9th element (None if no DFlash).
    """
    uniform_groups = [
        group
        for group in kv_cache_groups
        if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
    ]
    mamba_groups = [
        group for group in kv_cache_groups if isinstance(group.kv_cache_spec, MambaSpec)
    ]
    attn_group: KVCacheGroupSpec | None = None
    tail_group: KVCacheGroupSpec | None = None
    draft_group: KVCacheGroupSpec | None = None
    for group in uniform_groups:
        inner = cast(UniformTypeKVCacheSpecs, group.kv_cache_spec).kv_cache_specs
        if all(type(spec) is MLAAttentionSpec for spec in inner.values()):
            attn_group = group
        elif all(isinstance(spec, KpoolTailSpec) for spec in inner.values()):
            tail_group = group
        elif inner and all(type(spec) is SlidingWindowSpec for spec in inner.values()):
            draft_group = group
    if attn_group is None or not mamba_groups:
        return None
    if len(uniform_groups) + len(mamba_groups) != len(kv_cache_groups):
        return None

    attn_uniform = cast(UniformTypeKVCacheSpecs, attn_group.kv_cache_spec)
    mla_inner = cast(dict[str, MLAAttentionSpec], attn_uniform.kv_cache_specs)
    if not all(
        type(spec) is MLAAttentionSpec and spec.page_size_padded is None
        for spec in mla_inner.values()
    ):
        return None
    mla_names = [
        name for name in attn_group.layer_names if mla_inner[name].tokens_per_state == 1
    ]
    idx_names = [
        name for name in attn_group.layer_names if mla_inner[name].tokens_per_state > 1
    ]
    mla_pages = {mla_inner[name].page_size_bytes for name in mla_names}
    idx_pages = {mla_inner[name].page_size_bytes for name in idx_names}
    if len(mla_pages) != 1 or len(idx_pages) != 1:
        return None
    mla_page = mla_pages.pop()
    idx_page = idx_pages.pop()
    if any(group.kv_cache_spec.page_size_bytes != mla_page for group in mamba_groups):
        return None

    if draft_group is not None:
        draft_inner = cast(
            UniformTypeKVCacheSpecs, draft_group.kv_cache_spec
        ).kv_cache_specs
        draft_pages = {spec.page_size_bytes for spec in draft_inner.values()}
        if len(draft_pages) != 1:
            return None
        if any(spec.page_size_padded is not None for spec in draft_inner.values()):
            return None
        if draft_pages.pop() == mla_page and len(draft_group.layer_names) > len(
            mla_names
        ):
            return None

    tail_names: list[str] = []
    tail_page = 0
    if tail_group is not None:
        tail_names = list(tail_group.layer_names)
        tail_inner = cast(
            UniformTypeKVCacheSpecs, tail_group.kv_cache_spec
        ).kv_cache_specs
        tail_pages = {
            cast(KpoolTailSpec, spec).unpadded_page_size_bytes
            for spec in tail_inner.values()
        }
        if len(tail_pages) != 1 or len(tail_names) != len(idx_names):
            return None
        tail_page = tail_pages.pop()
        if tail_page > idx_page:
            return None

    return (
        attn_group,
        mamba_groups,
        mla_names,
        idx_names,
        mla_page,
        idx_page,
        tail_names,
        tail_page,
        draft_group,
    )


'''


def replace_once(text: str, old: str, new: str, label: str) -> str:
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"{P}: expected one {label} target, found {n}")
    return text.replace(old, new, 1)


def extract_def(text: str, name: str) -> str:
    needle = f"def {name}("
    start = text.find(needle)
    if start < 0:
        raise SystemExit(f"{P}: missing {name}")
    # next top-level def after this one
    nxt = text.find("\ndef ", start + 1)
    if nxt < 0:
        raise SystemExit(f"{P}: cannot find end of {name}")
    return text[start:nxt]


def main() -> int:
    if not P.is_file():
        raise SystemExit(f"missing {P}")
    text = P.read_text()
    if MARK in text:
        print(f"{P.name}: {MARK} already present — skipping")
        return 0

    groups_old = extract_def(text, "_get_kv_cache_groups_glm5_next")
    layout_old = extract_def(text, "_glm5_next_tensor_layout")
    text = replace_once(text, groups_old, GROUPS_NEW, "_get_kv_cache_groups_glm5_next")
    text = replace_once(text, layout_old, LAYOUT_NEW, "_glm5_next_tensor_layout")

    # Call-site updates: 8-tuple -> 9-tuple + draft accounting.
    bytes_old = """    if (glm5_layout := _glm5_next_tensor_layout(kv_cache_groups)) is not None:
        _, _, mla_names, idx_names, mla_page, idx_page, _, _ = glm5_layout
        return len(mla_names) * mla_page + len(idx_names) * idx_page
"""
    bytes_new = """    if (glm5_layout := _glm5_next_tensor_layout(kv_cache_groups)) is not None:
        _, _, mla_names, idx_names, mla_page, idx_page, _, _, draft_group = glm5_layout
        per_block = len(mla_names) * mla_page + len(idx_names) * idx_page
        if draft_group is not None:  """ + MARK + """
            draft_page = next(
                iter(
                    cast(
                        UniformTypeKVCacheSpecs, draft_group.kv_cache_spec
                    ).kv_cache_specs.values()
                )
            ).page_size_bytes
            if draft_page != mla_page:
                per_block += len(draft_group.layer_names) * draft_page
        return per_block
"""
    text = replace_once(text, bytes_old, bytes_new, "bytes_per_block glm5")

    config_old = """    if (glm5_layout := _glm5_next_tensor_layout(kv_cache_groups)) is not None:
        (
            attn_group,
            mamba_groups,
            mla_names,
            idx_names,
            mla_page,
            idx_page,
            tail_names,
            _,
        ) = glm5_layout
        bytes_per_block = len(mla_names) * mla_page + len(idx_names) * idx_page
        num_blocks = may_override_num_blocks(
            vllm_config, available_memory // bytes_per_block
        )
        size = bytes_per_block * num_blocks
        attn_specs = cast(
            UniformTypeKVCacheSpecs, attn_group.kv_cache_spec
        ).kv_cache_specs

        kv_cache_tensors: list[KVCacheTensor] = []

        def add_tensor(layer_name: str, spec: KVCacheSpec, offset: int) -> None:
            kv_cache_tensors.append(
                KVCacheTensor(
                    size=size,
                    layers=[layer_name],
                    layer_stride=spec.page_size_bytes * num_blocks,
                    block_stride=spec.page_size_bytes,
                    offset=offset,
                )
            )

        for index, mla_name in enumerate(mla_names):
            offset = index * mla_page * num_blocks
            add_tensor(mla_name, attn_specs[mla_name], offset)
            for group in mamba_groups:
                if index < len(group.layer_names):
                    add_tensor(group.layer_names[index], group.kv_cache_spec, offset)

        idx_base = len(mla_names) * mla_page * num_blocks
        for index, idx_name in enumerate(idx_names):
            offset = idx_base + index * idx_page * num_blocks
            add_tensor(idx_name, attn_specs[idx_name], offset)
            if tail_names:
                tail_name = tail_names[index]
                tail_group = next(
                    group for group in kv_cache_groups if tail_name in group.layer_names
                )
                tail_specs = cast(
                    UniformTypeKVCacheSpecs, tail_group.kv_cache_spec
                ).kv_cache_specs
                add_tensor(tail_name, tail_specs[tail_name], offset)

        return KVCacheConfig(
            num_blocks=num_blocks,
            kv_cache_tensors=kv_cache_tensors,
            kv_cache_groups=kv_cache_groups,
            prefix_cache_retention_interval=(
                vllm_config.cache_config.prefix_cache_retention_interval
            ),
        )
"""
    config_new = """    if (glm5_layout := _glm5_next_tensor_layout(kv_cache_groups)) is not None:
        (
            attn_group,
            mamba_groups,
            mla_names,
            idx_names,
            mla_page,
            idx_page,
            tail_names,
            _,
            draft_group,
        ) = glm5_layout
        draft_names: list[str] = []
        draft_page = 0
        draft_shared = False
        if draft_group is not None:  """ + MARK + """
            draft_names = list(draft_group.layer_names)
            draft_page = next(
                iter(
                    cast(
                        UniformTypeKVCacheSpecs, draft_group.kv_cache_spec
                    ).kv_cache_specs.values()
                )
            ).page_size_bytes
            draft_shared = draft_page == mla_page
        bytes_per_block = len(mla_names) * mla_page + len(idx_names) * idx_page
        if draft_names and not draft_shared:
            bytes_per_block += len(draft_names) * draft_page
        num_blocks = may_override_num_blocks(
            vllm_config, available_memory // bytes_per_block
        )
        size = bytes_per_block * num_blocks
        attn_specs = cast(
            UniformTypeKVCacheSpecs, attn_group.kv_cache_spec
        ).kv_cache_specs

        kv_cache_tensors: list[KVCacheTensor] = []

        def add_tensor(layer_name: str, spec: KVCacheSpec, offset: int) -> None:
            kv_cache_tensors.append(
                KVCacheTensor(
                    size=size,
                    layers=[layer_name],
                    layer_stride=spec.page_size_bytes * num_blocks,
                    block_stride=spec.page_size_bytes,
                    offset=offset,
                )
            )

        for index, mla_name in enumerate(mla_names):
            offset = index * mla_page * num_blocks
            add_tensor(mla_name, attn_specs[mla_name], offset)
            for group in mamba_groups:
                if index < len(group.layer_names):
                    add_tensor(group.layer_names[index], group.kv_cache_spec, offset)
            if draft_shared and index < len(draft_names):
                draft_specs_map = cast(
                    UniformTypeKVCacheSpecs, draft_group.kv_cache_spec
                ).kv_cache_specs
                dname = draft_names[index]
                add_tensor(dname, draft_specs_map[dname], offset)

        idx_base = len(mla_names) * mla_page * num_blocks
        for index, idx_name in enumerate(idx_names):
            offset = idx_base + index * idx_page * num_blocks
            add_tensor(idx_name, attn_specs[idx_name], offset)
            if tail_names:
                tail_name = tail_names[index]
                tail_group = next(
                    group for group in kv_cache_groups if tail_name in group.layer_names
                )
                tail_specs = cast(
                    UniformTypeKVCacheSpecs, tail_group.kv_cache_spec
                ).kv_cache_specs
                add_tensor(tail_name, tail_specs[tail_name], offset)

        if draft_names and not draft_shared:
            draft_specs_map = cast(
                UniformTypeKVCacheSpecs, draft_group.kv_cache_spec
            ).kv_cache_specs
            draft_base = idx_base + len(idx_names) * idx_page * num_blocks
            for index, dname in enumerate(draft_names):
                offset = draft_base + index * draft_page * num_blocks
                add_tensor(dname, draft_specs_map[dname], offset)

        return KVCacheConfig(
            num_blocks=num_blocks,
            kv_cache_tensors=kv_cache_tensors,
            kv_cache_groups=kv_cache_groups,
            prefix_cache_retention_interval=(
                vllm_config.cache_config.prefix_cache_retention_interval
            ),
        )
"""
    text = replace_once(text, config_old, config_new, "get_kv_cache_config_from_groups glm5")

    mem_old = """    if (glm5_layout := _glm5_next_tensor_layout(kv_cache_groups)) is not None:
        (
            attn_group,
            mamba_groups,
            mla_names,
            idx_names,
            mla_page,
            idx_page,
            tail_names,
            _,
        ) = glm5_layout
        uniform_spec = cast(UniformTypeKVCacheSpecs, attn_group.kv_cache_spec)
        total_blocks = uniform_spec.max_memory_usage_pages(vllm_config)
        total_blocks += sum(
            cdiv(
                group.kv_cache_spec.max_memory_usage_bytes(vllm_config),
                group.kv_cache_spec.page_size_bytes,
            )
            for group in mamba_groups
        )
        if tail_names:
            total_blocks += 1
        return total_blocks * (len(mla_names) * mla_page + len(idx_names) * idx_page)
"""
    mem_new = """    if (glm5_layout := _glm5_next_tensor_layout(kv_cache_groups)) is not None:
        (
            attn_group,
            mamba_groups,
            mla_names,
            idx_names,
            mla_page,
            idx_page,
            tail_names,
            _,
            draft_group,
        ) = glm5_layout
        uniform_spec = cast(UniformTypeKVCacheSpecs, attn_group.kv_cache_spec)
        total_blocks = uniform_spec.max_memory_usage_pages(vllm_config)
        total_blocks += sum(
            cdiv(
                group.kv_cache_spec.max_memory_usage_bytes(vllm_config),
                group.kv_cache_spec.page_size_bytes,
            )
            for group in mamba_groups
        )
        if tail_names:
            total_blocks += 1
        per_block = len(mla_names) * mla_page + len(idx_names) * idx_page
        if draft_group is not None:  """ + MARK + """
            draft_uniform = cast(UniformTypeKVCacheSpecs, draft_group.kv_cache_spec)
            total_blocks += draft_uniform.max_memory_usage_pages(vllm_config)
            draft_page = next(iter(draft_uniform.kv_cache_specs.values())).page_size_bytes
            if draft_page != mla_page:
                per_block += len(draft_group.layer_names) * draft_page
        return total_blocks * per_block
"""
    text = replace_once(text, mem_old, mem_new, "_max_memory_usage_bytes_from_groups glm5")

    P.write_text(text)
    print(f"patched {P.name} ({MARK}: DFlash drafter group on GLM5 KV path)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
