# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
from collections import deque
from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.logger import init_logger
from vllm.triton_utils import tl, tldevice, triton
# Local gumbel_noised_argmax (pre-#54282 walk wiring).
# Image gumbel.py has #54282; overlaying IS_DRAFTING into this walk was
# MEASURED_DISCARD (soak/accept collapse). See benchmarks/pr54282-RESULTS.md.
from vllm.v1.worker.gpu.sample.gumbel import tl_rand32, tl_rand64
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator

logger = init_logger(__name__)


# Analysis only when DFLASH2_ACC_PROBE=1; Golden path unchanged at 0.
# Observation only. Does not change scores, walk, or rejection.
# lm_head_topk = compute_candidates() pool (NOT edge-score re-ranking).
#
# Reject stages (first-reject):
#   B0 = lm_head_topk_miss (alias A)
#   B1 = b_unary_would_hit (in pool; unary#1 == target; edge overrode)
#   B2 = b_unary_also_miss (in pool; unary#1 != target)
_PROBE_ENV = "DFLASH2_ACC_PROBE"
# Prefer /logs (compose LOG_DIR mount). /cache is legacy; not mounted here.
_PROBE_PATHS = (
    "/logs/dflash2-acc-state.json",
    "/tmp/dflash2-acc-state.json",
    "/cache/dflash2-acc-state.json",
)
_PROBE_CSV_PATH = "/logs/dflash2-acc.csv"
_PROBE_CSV_FIELDS = (
    "ts",
    "draft_rounds",
    "accepted_tokens",
    "drafted_tokens",
    "rejected_rounds",
    "reject_stage_b0_total",
    "reject_stage_b1_total",
    "reject_stage_b2_total",
    "accept_ratio_recent",
    "accept_ratio_window_128",
    "accept_ratio_window_256",
    "lm_head_topk_miss",
    "lm_head_topk_hit_walk_miss",
    "walk_hit_verify_reject",
    "b_unary_would_hit",
    "b_unary_also_miss",
)
# Prometheus gauges registered only when probe is on (debug boot).
_PROBE_PROM: dict[str, Any] | None = None


def _probe_enabled() -> bool:
    if os.environ.get(_PROBE_ENV, "0").strip() != "1":
        return False
    try:
        import torch.distributed as dist

        if dist.is_initialized() and dist.get_rank() != 0:
            return False
    except Exception:
        pass
    return True


def _empty_probe_cum(num_spec: int) -> dict[str, Any]:
    return {
        "draft_rounds": 0,
        "accepted_tokens": 0,
        "drafted_tokens": 0,
        "rejected_rounds": 0,
        "all_accepted_rounds": 0,
        # A / B0: target not in lm_head top-k at the first rejected draft position.
        "lm_head_topk_miss": 0,
        "reject_stage_b0_total": 0,
        # B: target in lm_head top-k but walk picked a different candidate.
        "lm_head_topk_hit_walk_miss": 0,
        # C: walk token == verifier target but still rejected (should be ~0 at temp=0).
        "walk_hit_verify_reject": 0,
        # Aliases for bench_acceptance.py (first-reject coverage).
        "rejected": 0,
        "in_topk": 0,
        "not_in_topk": 0,
        "tgt_rank_sum": 0,
        "tgt_rank_n": 0,
        "sel_rank_sum": 0,
        "sel_rank_n": 0,
        "pos_accept": [0] * num_spec,
        "reject_at_pos": [0] * num_spec,
        # B1 / B2 (subset of B).
        "b_unary_would_hit": 0,
        "reject_stage_b1_total": 0,
        "b_unary_also_miss": 0,
        "reject_stage_b2_total": 0,
        "b_tgt_unary_rank0": 0,
        "b_tgt_unary_rank_le2": 0,
        "b_tgt_unary_rank_le7": 0,
        "b_tgt_edge_rank_sum": 0,
        "b_tgt_edge_rank_n": 0,
        "accept_ratio_recent": 0.0,
        "accept_ratio_window_128": 0.0,
        "accept_ratio_window_256": 0.0,
    }


def _token_rank(candidates_row, token_id: int) -> int | None:
    """0-based index in lm_head top-k (index 0 = highest unary logit)."""
    hits = (candidates_row == token_id).nonzero(as_tuple=False)
    if hits.numel() == 0:
        return None
    return int(hits[0].item())


def _higher_is_better_rank(scores_row, idx: int) -> int:
    """0 = best. scores_row is the K edge scores walk actually argmax'd."""
    return int((scores_row > scores_row[idx]).sum().item())


def _selector_top_k_from_config(draft_config: dict) -> int:
    """Must match CandidateSelector.top_k in qwen3_dflash2.py.

    selector_rank is a trained codebook dimension and is never overridden.
    """
    raw = os.environ.get("DFLASH_SELECTOR_TOP_K")
    if raw is None or raw.strip() == "":
        return int(draft_config["selector_top_k"])
    value = int(raw)
    if value < 1:
        raise ValueError(f"DFLASH_SELECTOR_TOP_K must be >= 1, got {value}")
    return value


@triton.jit
def gumbel_noised_argmax(
    logits,
    keys,
    mask,
    seed,
    pos,
    temp,
    USE_FP64: tl.constexpr,
    APPLY_TEMPERATURE: tl.constexpr = True,
):
    """Argmax under Gumbel-max, or plain argmax at temp 0. (pre-#54282 control)"""
    if temp != 0.0 and APPLY_TEMPERATURE:
        logits = logits / temp

    if USE_FP64:
        logits = logits.to(tl.float64)
    if temp != 0.0:
        gumbel_seed = tl.randint(seed, pos)
        if USE_FP64:
            u = tl_rand64(gumbel_seed, keys, includes_zero=False)
            gumbel_noise = -tl.log(-tl.log(u))
        else:
            u = tl_rand32(gumbel_seed, keys, includes_zero=False)
            gumbel_noise = -tl.log(-tldevice.log1p(-u))
        logits = tl.where(mask, logits + gumbel_noise, float("-inf"))

    return tl.max(logits, axis=0, return_indices=True)


@triton.jit
def _selector_walk_kernel(
    scores_ptr,
    candidate_ptr,
    sample_pos_ptr,
    req_state_ptr,
    temperature_ptr,
    seeds_ptr,
    tokens_ptr,
    realized_scores_ptr,
    num_steps: tl.constexpr,
    top_k: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SAMPLE_PROBABILISTIC: tl.constexpr,
    USE_FP64: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_K)
    mask = offsets < top_k
    req_state = tl.load(req_state_ptr + row * num_steps)
    valid = req_state >= 0
    temperature = tl.load(temperature_ptr + req_state, mask=valid, other=0.0)
    seed = tl.load(seeds_ptr + req_state, mask=valid, other=0)
    previous = 0
    for step in range(num_steps):
        flat = row * num_steps + step
        score_base = (flat * top_k + previous) * top_k
        scores = tl.load(
            scores_ptr + score_base + offsets,
            mask=mask & valid,
            other=float("-inf"),
        ).to(tl.float64 if USE_FP64 else tl.float32)
        candidate_base = flat * top_k
        candidates = tl.load(
            candidate_ptr + candidate_base + offsets,
            mask=mask & valid,
            other=0,
        )

        # pre-#54282 control (no IS_DRAFTING)
        position = tl.load(sample_pos_ptr + flat) - 1
        _, index = gumbel_noised_argmax(
            scores,
            candidates,
            mask & valid,
            seed,
            position,
            temperature if SAMPLE_PROBABILISTIC else 0.0,
            USE_FP64=USE_FP64,
        )

        tl.store(
            realized_scores_ptr + candidate_base + offsets,
            scores,
            mask=mask & valid,
        )
        token = tl.load(candidate_ptr + candidate_base + index, mask=valid, other=0)
        tl.store(tokens_ptr + flat, token, mask=valid)
        previous = index


@triton.jit
def _cache_draft_logits_kernel(
    draft_logits_ptr,
    cached_candidate_ptr,
    candidate_ptr,
    scores_ptr,
    req_state_ptr,
    draft_logits_stride_0,
    draft_logits_stride_1,
    num_steps: tl.constexpr,
    top_k: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    flat = tl.program_id(0)
    req_state = tl.load(req_state_ptr + flat)
    step = flat % num_steps
    offsets = tl.arange(0, BLOCK_K)
    mask = (req_state >= 0) & (offsets < top_k)
    candidate_base = flat * top_k
    cache_base = (req_state * num_steps + step) * top_k
    old_token_ids = tl.load(cached_candidate_ptr + cache_base + offsets, mask=mask)
    logits_base = (
        draft_logits_ptr
        + req_state * draft_logits_stride_0
        + step * draft_logits_stride_1
    )
    tl.store(logits_base + old_token_ids, -float("inf"), mask=mask)
    token_ids = tl.load(candidate_ptr + candidate_base + offsets, mask=mask)
    scores = tl.load(scores_ptr + candidate_base + offsets, mask=mask)
    tl.store(logits_base + token_ids, scores, mask=mask)
    tl.store(cached_candidate_ptr + cache_base + offsets, token_ids, mask=mask)


class DFlash2Speculator(DFlashSpeculator):
    _speculator_name = "DFlash2"

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        draft_config = self.draft_model_config.hf_config.dflash_config
        self.selector_top_k = _selector_top_k_from_config(draft_config)
        logger.info(
            "%s selector_top_k=%s (checkpoint=%s)",
            self._speculator_name,
            self.selector_top_k,
            draft_config["selector_top_k"],
        )
        self._anchor_indices = (
            torch.arange(self.max_num_reqs, dtype=torch.int64, device=device)
            * self.num_query_per_req
        )
        self._selector_scores = torch.empty(
            self.max_num_reqs,
            self.num_speculative_steps,
            self.selector_top_k,
            dtype=torch.float32,
            device=device,
        )
        self._cached_candidate_ids = torch.zeros(
            self._selector_scores.shape, dtype=torch.int64, device=device
        )
        # Probe stash: last propose's lm_head top-k ids, packed like the draft batch.
        self._probe_candidates: torch.Tensor | None = None
        self._probe_unary: torch.Tensor | None = None
        self._probe_edge_scores: torch.Tensor | None = None
        self._probe_req_states: torch.Tensor | None = None
        self._probe_n = 0
        self._probe_cum = _empty_probe_cum(self.num_speculative_steps)
        self._probe_error_logged = False
        # Rolling (accepted, drafted) per round — PROBE=1 only uses these.
        self._probe_win_128: deque[tuple[int, int]] = deque(maxlen=128)
        self._probe_win_256: deque[tuple[int, int]] = deque(maxlen=256)
        self._probe_recent: tuple[int, int] = (0, 0)
        walk = os.environ.get("DFLASH_WALK_MODE", "edge").strip().lower()
        # edge = trained bilinear path (production). unary = independent
        # argmax of lm_head top-k logits (B1 A/B). selector_rank unused by unary.
        if walk not in ("edge", "unary"):
            raise ValueError(f"DFLASH_WALK_MODE must be edge|unary, got {walk}")
        self.walk_mode = walk
        logger.info("%s walk_mode=%s", self._speculator_name, self.walk_mode)
        if _probe_enabled():
            self._install_rejection_probe()

    def draft_logits_spec(self, vllm_config: VllmConfig) -> tuple[torch.dtype, float]:
        # fp32 so the walk and the rejection that checks it read the same
        # distribution; -inf because the cache kernel writes only the K
        # candidates.
        return torch.float32, -float("inf")

    def _sample_path(
        self,
        candidate_ids: torch.Tensor,
        scores: torch.Tensor,
        num_reqs: int,
    ) -> None:
        block_k = triton.next_power_of_2(self.selector_top_k)
        _selector_walk_kernel[(num_reqs,)](
            scores.contiguous(),
            candidate_ids.contiguous(),
            self.sample_pos,
            self.sample_idx_mapping,
            self.temperature,
            self.seeds,
            self.draft_tokens,
            self._selector_scores,
            num_steps=self.num_speculative_steps,
            top_k=self.selector_top_k,
            BLOCK_K=block_k,
            SAMPLE_PROBABILISTIC=self.draft_logits is not None,
            USE_FP64=self.use_fp64_gumbel,
            num_warps=1,
        )

    def _sample_path_unary(
        self,
        candidate_ids: torch.Tensor,
        unary_logits: torch.Tensor,
        num_reqs: int,
    ) -> None:
        """Independent per-position argmax over the lm_head top-k pool.

        Does not use predecessor/successor codebooks. Recovers B1 (unary
        would have hit) and cannot recover B2 or A.
        """
        idx = unary_logits.argmax(dim=-1, keepdim=True)
        self.draft_tokens[:num_reqs] = candidate_ids.gather(-1, idx).squeeze(-1)
        self._selector_scores[:num_reqs].copy_(unary_logits.float())

    def _cache_draft_logits(self, candidate_ids: torch.Tensor, num_sample: int) -> None:
        draft_logits = self.draft_logits
        assert draft_logits is not None
        block_k = triton.next_power_of_2(self.selector_top_k)
        _cache_draft_logits_kernel[(num_sample,)](
            draft_logits,
            self._cached_candidate_ids,
            candidate_ids,
            self._selector_scores,
            self.sample_idx_mapping,
            draft_logits.stride(0),
            draft_logits.stride(1),
            num_steps=self.num_speculative_steps,
            top_k=self.selector_top_k,
            BLOCK_K=block_k,
            num_warps=1,
        )

    def _generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        last_hidden_states = self._run_model(
            num_tokens_padded,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
        )
        num_sample = num_reqs * self.num_speculative_steps
        hidden_states = last_hidden_states[self.sample_indices[:num_sample]].view(
            num_reqs, self.num_speculative_steps, -1
        )
        candidate_ids, unary_logits = self.model.compute_candidates(
            hidden_states.flatten(0, 1)
        )
        candidate_ids = candidate_ids.view(
            num_reqs, self.num_speculative_steps, self.selector_top_k
        )
        unary_logits = unary_logits.view_as(candidate_ids)
        if self.walk_mode == "unary":
            self._sample_path_unary(candidate_ids, unary_logits, num_reqs)
        else:
            anchor_token_ids = self.input_buffers.input_ids[
                self._anchor_indices[:num_reqs]
            ]
            scores = self.model.model.candidate_selector(
                candidate_ids,
                unary_logits,
                hidden_states,
                anchor_token_ids,
            )
            self._sample_path(candidate_ids, scores, num_reqs)
        # Observation stash only when the A/B/C probe is on. Production
        # (DFLASH2_ACC_PROBE=0) skips this so decode does not keep extra refs.
        if _probe_enabled():
            self._probe_candidates = candidate_ids
            self._probe_unary = unary_logits
            self._probe_edge_scores = self._selector_scores[:num_reqs]
            self._probe_req_states = self.sample_idx_mapping[
                : num_reqs * self.num_speculative_steps : self.num_speculative_steps
            ]
            self._probe_n = num_reqs
        if self.draft_logits is not None:
            self._cache_draft_logits(candidate_ids, num_sample)

    def _install_rejection_probe(self) -> None:
        """Wrap RejectionSampler._verify after it returns. Algorithm unchanged."""
        from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler

        RejectionSampler._dflash2_acc_speculator = self  # type: ignore[attr-defined]
        if getattr(RejectionSampler, "_dflash2_acc_probe_installed", False):
            return
        orig = RejectionSampler._verify

        def _verify_with_probe(sampler_self, *args, **kwargs):
            out = orig(sampler_self, *args, **kwargs)
            spec = getattr(RejectionSampler, "_dflash2_acc_speculator", None)
            if spec is not None and _probe_enabled():
                try:
                    spec._observe_rejection(args, kwargs, out)
                except Exception:
                    if not spec._probe_error_logged:
                        spec._probe_error_logged = True
                        logger.warning(
                            "DFlash2 acceptance probe failed; further errors suppressed.",
                            exc_info=True,
                        )
            return out

        RejectionSampler._verify = _verify_with_probe  # type: ignore[method-assign]
        RejectionSampler._dflash2_acc_probe_installed = True  # type: ignore[attr-defined]
        logger.info(
            "DFlash2 acceptance probe on (lm_head_topk_hit / walk_target_hit / "
            "verified_accept). Set %s=0 to disable.",
            _PROBE_ENV,
        )

    def _observe_rejection(self, args, kwargs, out) -> None:
        """Classify first-reject: A=topk miss, B=walk miss, C=verify reject."""
        if self._probe_n <= 0 or self._probe_candidates is None:
            return
        # _verify(logits, draft_logits, draft_sampled, pos, cu_num_logits, idx_mapping, ...)
        draft_sampled = kwargs.get("draft_sampled", args[2] if len(args) > 2 else None)
        cu_num_logits = kwargs.get("cu_num_logits", args[4] if len(args) > 4 else None)
        idx_mapping = kwargs.get("idx_mapping", args[5] if len(args) > 5 else None)
        processed_logits, sampled, num_sampled = out
        del processed_logits
        if draft_sampled is None or cu_num_logits is None or idx_mapping is None:
            return

        num_spec = self.num_speculative_steps
        ns = num_sampled.detach().cpu()
        smp = sampled.detach().cpu()
        imap = idx_mapping.detach().cpu()
        cu = cu_num_logits.detach().cpu()
        ds = draft_sampled.detach().cpu()
        cands = self._probe_candidates.detach().cpu()
        unary = (
            self._probe_unary.detach().cpu() if self._probe_unary is not None else None
        )
        edge = (
            self._probe_edge_scores.detach().cpu()
            if self._probe_edge_scores is not None
            else None
        )
        req_states = self._probe_req_states.detach().cpu() if self._probe_req_states is not None else None
        if req_states is None:
            return

        # packed row -> req_state from last propose
        row_by_state = {
            int(req_states[i].item()): i for i in range(min(self._probe_n, req_states.numel()))
        }
        cum = self._probe_cum
        n_req = int(ns.numel())
        for r in range(n_req):
            accepted_drafts = max(int(ns[r].item()) - 1, 0)
            accepted_drafts = min(accepted_drafts, num_spec)
            cum["draft_rounds"] += 1
            cum["accepted_tokens"] += accepted_drafts
            cum["drafted_tokens"] += num_spec
            for p in range(accepted_drafts):
                cum["pos_accept"][p] += 1
            if accepted_drafts >= num_spec:
                cum["all_accepted_rounds"] += 1
                self._probe_recent = (accepted_drafts, num_spec)
                self._probe_win_128.append((accepted_drafts, num_spec))
                self._probe_win_256.append((accepted_drafts, num_spec))
                continue

            reject_pos = accepted_drafts
            cum["rejected_rounds"] += 1
            cum["rejected"] += 1
            cum["reject_at_pos"][reject_pos] += 1
            self._probe_recent = (accepted_drafts, num_spec)
            self._probe_win_128.append((accepted_drafts, num_spec))
            self._probe_win_256.append((accepted_drafts, num_spec))
            start = int(cu[r].item())
            draft_idx = start + reject_pos + 1
            if draft_idx >= int(ds.numel()) or reject_pos >= int(smp.shape[1]):
                continue
            walk_token = int(ds[draft_idx].item())
            target_token = int(smp[r, reject_pos].item())
            req_state = int(imap[r].item())
            row = row_by_state.get(req_state)
            if row is None or row >= cands.shape[0] or walk_token < 0:
                continue
            pool = cands[row, reject_pos]
            tgt_rank = _token_rank(pool, target_token)
            sel_rank = _token_rank(pool, walk_token)
            lm_head_topk_hit = tgt_rank is not None
            walk_target_hit = walk_token == target_token
            if sel_rank is not None:
                cum["sel_rank_sum"] += sel_rank
                cum["sel_rank_n"] += 1
            if tgt_rank is not None:
                cum["tgt_rank_sum"] += tgt_rank
                cum["tgt_rank_n"] += 1
            if not lm_head_topk_hit:
                # A / B0: verifier target is outside the lm_head top-k pool.
                cum["lm_head_topk_miss"] += 1
                cum["reject_stage_b0_total"] += 1
                cum["not_in_topk"] += 1
            else:
                cum["in_topk"] += 1
                if not walk_target_hit:
                    # B: target was a candidate; walk chose someone else.
                    cum["lm_head_topk_hit_walk_miss"] += 1
                    unary_hit = tgt_rank == 0
                    if unary is not None and row < unary.shape[0]:
                        unary_hit = (
                            int(pool[int(unary[row, reject_pos].argmax().item())].item())
                            == target_token
                        )
                    if unary_hit:
                        cum["b_tgt_unary_rank0"] += 1
                        cum["b_unary_would_hit"] += 1
                        cum["reject_stage_b1_total"] += 1
                    else:
                        cum["b_unary_also_miss"] += 1
                        cum["reject_stage_b2_total"] += 1
                    if tgt_rank is not None and tgt_rank <= 2:
                        cum["b_tgt_unary_rank_le2"] += 1
                    if tgt_rank is not None and tgt_rank <= 7:
                        cum["b_tgt_unary_rank_le7"] += 1
                    if (
                        edge is not None
                        and row < edge.shape[0]
                        and tgt_rank is not None
                    ):
                        cum["b_tgt_edge_rank_sum"] += _higher_is_better_rank(
                            edge[row, reject_pos], tgt_rank
                        )
                        cum["b_tgt_edge_rank_n"] += 1
                else:
                    # C: walk emitted the target token; greedy verify still rejected.
                    cum["walk_hit_verify_reject"] += 1
        # Refresh ratio fields once per verify batch.
        def _ratio(pairs) -> float:
            a = sum(x[0] for x in pairs)
            d = sum(x[1] for x in pairs)
            return float(a) / float(d) if d else 0.0

        cum["accept_ratio_recent"] = (
            _ratio([self._probe_recent]) if self._probe_recent[1] else 0.0
        )
        cum["accept_ratio_window_128"] = _ratio(self._probe_win_128)
        cum["accept_ratio_window_256"] = _ratio(self._probe_win_256)
        self._flush_probe()

    def _ensure_probe_prom(self) -> dict[str, Any] | None:
        """Best-effort same-process gauges. Cross-process: see .prom file."""
        global _PROBE_PROM
        if _PROBE_PROM is not None:
            return _PROBE_PROM
        try:
            from prometheus_client import Gauge
            from vllm.v1.metrics.prometheus import get_prometheus_registry

            registry = get_prometheus_registry()
            names = {
                "reject_stage_b0_total": "reject_stage_b0_total",
                "reject_stage_b1_total": "reject_stage_b1_total",
                "reject_stage_b2_total": "reject_stage_b2_total",
                "accept_ratio_recent": "accept_ratio_recent",
                "accept_ratio_window_128": "accept_ratio_window_128",
                "accept_ratio_window_256": "accept_ratio_window_256",
                "lm_head_topk_miss": "dflash2_acc_lm_head_topk_miss",
                "lm_head_topk_hit_walk_miss": "dflash2_acc_walk_miss",
                "walk_hit_verify_reject": "dflash2_acc_verify_reject",
                "rejected_rounds": "dflash2_acc_rejected_rounds",
                "accepted_tokens": "dflash2_acc_accepted_tokens",
                "drafted_tokens": "dflash2_acc_drafted_tokens",
                "draft_rounds": "dflash2_acc_draft_rounds",
            }
            gauges: dict[str, Any] = {}
            for key, metric in names.items():
                try:
                    gauges[key] = Gauge(
                        metric,
                        f"DFlash2 accept-ceiling probe cumulative ({key})",
                        registry=registry,
                    )
                except ValueError:
                    # Already registered on re-init in this process — skip.
                    pass
            _PROBE_PROM = gauges
            return _PROBE_PROM
        except Exception:
            _PROBE_PROM = {}
            return _PROBE_PROM

    def _flush_probe_csv(self) -> None:
        """Append one cumulative snapshot. Probe ON only (/logs mount)."""
        import time as _time

        path = _PROBE_CSV_PATH
        directory = os.path.dirname(path)
        if directory and not os.path.isdir(directory):
            return
        cum = self._probe_cum
        row = {
            "ts": f"{_time.time():.3f}",
            "draft_rounds": cum.get("draft_rounds", 0),
            "accepted_tokens": cum.get("accepted_tokens", 0),
            "drafted_tokens": cum.get("drafted_tokens", 0),
            "rejected_rounds": cum.get("rejected_rounds", 0),
            "reject_stage_b0_total": cum.get("reject_stage_b0_total", 0),
            "reject_stage_b1_total": cum.get("reject_stage_b1_total", 0),
            "reject_stage_b2_total": cum.get("reject_stage_b2_total", 0),
            "accept_ratio_recent": cum.get("accept_ratio_recent", 0.0),
            "accept_ratio_window_128": cum.get("accept_ratio_window_128", 0.0),
            "accept_ratio_window_256": cum.get("accept_ratio_window_256", 0.0),
            "lm_head_topk_miss": cum.get("lm_head_topk_miss", 0),
            "lm_head_topk_hit_walk_miss": cum.get("lm_head_topk_hit_walk_miss", 0),
            "walk_hit_verify_reject": cum.get("walk_hit_verify_reject", 0),
            "b_unary_would_hit": cum.get("b_unary_would_hit", 0),
            "b_unary_also_miss": cum.get("b_unary_also_miss", 0),
        }
        try:
            need_header = not os.path.isfile(path) or os.path.getsize(path) == 0
            with open(path, "a") as fh:
                if need_header:
                    fh.write(",".join(_PROBE_CSV_FIELDS) + "\n")
                fh.write(",".join(str(row[k]) for k in _PROBE_CSV_FIELDS) + "\n")
        except OSError:
            return

    def _flush_probe_prom_file(self) -> None:
        """Write Prometheus textfile under /logs (API process cannot see worker gauges)."""
        path = "/logs/dflash2-acc.prom"
        directory = os.path.dirname(path)
        if directory and not os.path.isdir(directory):
            return
        cum = self._probe_cum
        mapping = (
            ("accept_ratio_recent", "accept_ratio_recent"),
            ("accept_ratio_window_128", "accept_ratio_window_128"),
            ("accept_ratio_window_256", "accept_ratio_window_256"),
            ("reject_stage_b0_total", "reject_stage_b0_total"),
            ("reject_stage_b1_total", "reject_stage_b1_total"),
            ("reject_stage_b2_total", "reject_stage_b2_total"),
            ("dflash2_acc_draft_rounds", "draft_rounds"),
            ("dflash2_acc_accepted_tokens", "accepted_tokens"),
            ("dflash2_acc_drafted_tokens", "drafted_tokens"),
            ("dflash2_acc_rejected_rounds", "rejected_rounds"),
            ("dflash2_acc_lm_head_topk_miss", "lm_head_topk_miss"),
            ("dflash2_acc_walk_miss", "lm_head_topk_hit_walk_miss"),
            ("dflash2_acc_verify_reject", "walk_hit_verify_reject"),
            ("dflash2_acc_b_unary_would_hit", "b_unary_would_hit"),
            ("dflash2_acc_b_unary_also_miss", "b_unary_also_miss"),
        )
        lines = [
            "# HELP dflash2_acc_probe DFlash2 accept-ceiling debug (PROBE=1 only).",
            "# TYPE dflash2_acc_draft_rounds gauge",
        ]
        for metric, key in mapping:
            lines.append(f"{metric} {float(cum.get(key, 0))}")
        blob = "\n".join(lines) + "\n"
        try:
            tmp = path + ".tmp"
            with open(tmp, "w") as fh:
                fh.write(blob)
            os.replace(tmp, path)
        except OSError:
            return

    def _flush_probe_prom(self) -> None:
        gauges = self._ensure_probe_prom()
        cum = self._probe_cum
        if gauges:
            for key, gauge in gauges.items():
                try:
                    gauge.set(float(cum.get(key, 0)))
                except Exception:
                    continue
        # Always write textfile so host/diag can read without worker→API multiproc.
        self._flush_probe_prom_file()

    def _flush_probe(self) -> None:
        # Analysis only when DFLASH2_ACC_PROBE=1; Golden path unchanged at 0.
        payload = {
            "schema": "dflash2-acc-probe-v1",
            "topk_definition": (
                "lm_head_topk_hit: target in compute_candidates() pool "
                f"(DFLASH_SELECTOR_TOP_K={self.selector_top_k}). "
                "walk_target_hit: draft token == verifier target at first reject. "
                "verified_accept: draft positions accepted by rejection_sample."
            ),
            "selector_top_k": self.selector_top_k,
            "num_spec": self.num_speculative_steps,
            "cum": self._probe_cum,
        }
        blob = json.dumps(payload)
        for path in _PROBE_PATHS:
            try:
                directory = os.path.dirname(path)
                if directory and not os.path.isdir(directory):
                    continue
                tmp = path + ".tmp"
                with open(tmp, "w") as fh:
                    fh.write(blob)
                os.replace(tmp, path)
            except OSError:
                continue
        self._flush_probe_csv()
        self._flush_probe_prom()
