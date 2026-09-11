# Upstream analysis — mmastrac/glm-5.3-flash-4x-gx10

Phase 1 artifact. Source: https://github.com/mmastrac/glm-5.3-flash-4x-gx10 (main).

## Decision summary

| File | Purpose | Relevant to 2× Spark? | Copy / Rewrite / Drop | Reason |
|------|---------|------------------------|------------------------|--------|
| `image/Dockerfile` | Base `vllm/vllm-openai:glm53-flash-arm64-cu130` + FlashInfer/NCCL/CuTeDSL pins + SM121 patches + mentat | Yes | **Copy** (done) | GB10/SM121 correctness; same base required |
| `image/patches/*` | SM121 MLA, top-k, plugin, mem cap/trace, CUDA headers, verify, thinking_budget | Yes | **Copy** (done) | Required for NoPE sparse MLA on SM121 |
| `image/entrypoint.sh` | Fabric discovery, mentat/ray join, `vllm serve` | Yes | **Rewrite** | Keep fabric/mentat/serve logic; allow HF Hub `MODEL` (no revision pin); defaults for TP=2 Spark |
| `image/chat-template.jinja` | Multimodal-capable chat template | Yes | **Copy** (done) | Checkpoint text-only template breaks images |
| `image/self-test.py` | Post-boot probe | Yes | **Copy** (done) | Health gate |
| `image/status-server.py` | :8082 status / MCP | Yes | **Copy** (done) | Bring-up visibility |
| `compose/glm53.yaml` | Single service, ROLE/NODE_RANK, host net, RoCE | Yes (already TP=2 pair) | **Rewrite** | HF cache mounts; 2× DGX Spark env; image tag `glm53-spark:2x-sm121` |
| `compose/dflash2-full-override.yaml` | Host-mounted DFlash draft + vLLM file binds | Partial | **Rewrite** | Hub draft id + EXTRA_ARGS; no host `pp-patches` dependency |
| `compose/pp-mtp-override.yaml` | MTP/PP experiments | No (baseline MTP off / DFlash preferred) | **Drop** | Not part of validated production profile |
| `compose/spin-wait-override.yaml` | Spin-wait tuning | Optional | **Drop** for now | Can re-add after bring-up |
| `compose/mentatd.yaml` | Host mentat daemon (Ray replacement control plane) | Yes | **Rewrite** | 2-node peers only |
| `compose/mentatd-serve.yaml` | OpenAI front door :6381 | Optional | **Copy** (light rewrite) | Useful on head; not required for direct :8002 |
| `scripts/up.sh` | SSH orchestrate mentatd + compose | Yes | **Rewrite** | 2-node from `.env`; repo-local compose paths |
| `scripts/down.sh` | Stop containers | Yes | **Rewrite** | Stop glm53 (+ optional mentat); never delete HF cache |
| `.env.example` | Cluster knobs | Yes | **Rewrite** | TP=2, HF Hub models, 192.168.100.x, production baseline |

## Architecture notes from upstream

- Upstream **serving compose is already a 2-node TP=2 pair** (`ROLE=head|worker`). The 4-node story is mostly `.env` / orchestration scale-out, not a different image.
- Multi-node TP **requires mentat** (in-image Ray shim + host `mentatd`). Real Ray OOMs unified memory.
- Model load in upstream is **local `MODEL_DIR` path**. This repo policy uses **Hugging Face Hub id + standard HF cache mount** instead of `/var/tmp` or baked weights.

## Imported (image)

- SM121 patch, GB10 top-k fallback, GB10 plugin backend, worker memory cap, memory trace, CUDA header support, thinking_budget_guard, verify.py
