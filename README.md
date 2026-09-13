# GLM-5.3-Flash — 2× DGX Spark (GB10 / SM121)

**Version:** [`v1.0.0-golden`](VERSION) · Canonical config: [`docs/golden-configuration.md`](docs/golden-configuration.md)

vLLM serving recipe for **two NVIDIA DGX Spark** nodes (GB10 / SM121, TP=2).  
This is **not** a mechanically downsized 4-node GX10 recipe — topology, memory, launcher, and runtime knobs are built for 2× Spark from the start.

| Item | Value |
|------|--------|
| GPU | NVIDIA GB10 ×1 per node |
| Arch | SM121 / compute capability 12.1a |
| Nodes / TP | 2 / TP=2 |
| Engine | vLLM only (`glm53-spark:2x-sm121`) |
| vLLM / Torch / CUDA | `0.28.1rc1.dev580+g385dce36b` / `2.13.0+cu130` / `13.0` |
| FlashInfer | `flashinfer-python 0.6.18.dev20260819` |
| Draft | `incoai/GLM-5.3-Flash-DFlash2` (DFlash2, K=7) |
| Default target | `canada-quant/glm-5.3-w4a16-mtp` (`SERVE_LANE=golden`) |
| Optional target | `axiomofmind/GLM-5.3-Flash-W4A16-NVFP4` (`SERVE_LANE=modelopt`) |
| Upstream reference | [mmastrac/glm-5.3-flash-4x-gx10](https://github.com/mmastrac/glm-5.3-flash-4x-gx10) |

---

## Status

| | Aspirational gate | Golden (measured) |
|--|-------------------|-------------------|
| Single-stream decode | 40–50 tok/s | **~34.7–35.5** soak (60m floor **~33.5**) |
| Spec accept (`/metrics`) | 50–70% | **~0.43–0.44** (B2 draft mismatch ceiling) |
| Concurrent C6 | stretch | **~81–87** tok/s aggregate |

Production prioritizes **stability and reproducibility** over chasing 40–60 tok/s with unproven patches.  
Rejected on this stack: spin-wait `0.002`, logits `64`, expandable OFF, TOP_K=48, ACC probe in prod, `#54282` walk, FlashInfer SM121 rebuild chase.

Details: [`docs/performance.md`](docs/performance.md) · [`docs/benchmark.md`](docs/benchmark.md) · [`benchmarks/RESULTS.md`](benchmarks/RESULTS.md)

---

## Quick start

```bash
cp .env.example .env
# Same HEAD_HOST on both nodes. Set ROLE / NODE_RANK / MENTAT_* per node.
docker build -t glm53-spark:2x-sm121 image/   # each node, or save/load
bash scripts/up.sh
bash scripts/status.sh
bash scripts/self-test.sh
```

Stop:

```bash
bash scripts/down.sh                 # glm53 only
STOP_MENTAT=1 bash scripts/down.sh   # also mentatd
```

`up.sh`: apply serve lane → generate APC/KV/SM90 overlays → lock clocks → sync to worker → compose up → wait `/health`.

API: `http://<HEAD_HOST>:8000` · served name default `glm-5.3-flash`  
Status page: `http://<HEAD_HOST>:8082` (polls `/v1/models` + `/metrics`; not on the inference hot path)

---

## SERVE_LANE (checkpoint switch, no image rebuild)

| Lane | `.env` | Target Hub id | glm5next overlay |
|------|--------|---------------|------------------|
| **golden** (default) | `SERVE_LANE=golden` | `canada-quant/glm-5.3-w4a16-mtp` | `patches/glm5next_model.py` |
| **modelopt** | `SERVE_LANE=modelopt` | `axiomofmind/GLM-5.3-Flash-W4A16-NVFP4` | `patches/glm5next_model.modelopt.py` |

`scripts/apply-serve-lane.sh` writes gitignored `.env.lane` for head+worker compose. Draft stays `DRAFT_MODEL` (default `incoai/GLM-5.3-Flash-DFlash2`).

ModelOpt lane notes:

- Needs Eagle3 on the multimodal wrapper (DFlash2); stock image multimodal lacks `SupportsEagle3`.
- Optional speed KEEP on modelopt: `APPLY_GATE_LINEAR=1` → `compose/overrides/gate-linear.yaml` via `up.sh`.
- NVFP4 MoE on GB10 uses **Marlin** (weight-only). `b12x` / FlashInfer NVFP4 backends are not supported on this SM121 path.
- Evidence: [`evidence/modelopt-20260912/`](evidence/modelopt-20260912/)

See [`docs/deployment.md`](docs/deployment.md).

---

## Hardware & network

| Role | Default IP |
|------|------------|
| HEAD (rank 0) | `192.168.100.10` |
| WORKER (rank 1) | `192.168.100.20` |

ConnectX-7 RoCE. IPs and NCCL knobs live in `.env` only (not hardcoded in Python/Dockerfile).

```text
NCCL_SOCKET_IFNAME=enp1s0f0np0
NCCL_IB_HCA=rocep1s0f0
NCCL_IB_GID_INDEX=3
```

Both nodes must use the **same** `HEAD_HOST` (worker must not set itself as head).

---

## Golden knobs (`v1.0.0-golden`)

```text
SERVE_LANE=golden
DFLASH_TOKENS=7
DFLASH_SELECTOR_TOP_K=32
DFLASH_WALK_MODE=edge
DFLASH2_ACC_PROBE=0

MOE_BACKEND=marlin
ENFORCE_EAGER=1
CUDA_GRAPHS=0
ASYNC_SCHEDULING=1
DISABLE_FLASHINFER_AUTOTUNE=1
APPLY_APC_PATCH=1
GLM53_SM121_MLA=0
APPLY_GATE_LINEAR=0          # golden default; modelopt may set 1

LOCK_CLOCKS=1
CLOCK_MHZ=2400               # stability / low drift; Auto boost ≈ same TPS
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=512
SPINWAIT_PATCH_HOST=../runtime/spinwait/shm_broadcast.pristine.py
KV_CACHE_DTYPE=fp8_e4m3
KV_CACHE_MEMORY=9663676416
MAX_MODEL_LEN=1048576
MAX_NUM_SEQS=6
MAX_NUM_BATCHED_TOKENS=8192
GPU_MEM_UTIL=0.85
```

Full KEEP / REJECTED table: [`docs/golden-configuration.md`](docs/golden-configuration.md)

### Models & revisions

- Hub ids only — **do not pin** Hugging Face revisions.
- Do not use `/var/tmp` for weights. Host cache: `${HOST_CACHE:-~/.cache}` → container `/root/.cache`.

| Variable | Default |
|----------|---------|
| `MODEL` / lane | set by `SERVE_LANE` |
| `DRAFT_MODEL` | `incoai/GLM-5.3-Flash-DFlash2` |
| `SERVED_MODEL_NAME` | `glm-5.3-flash` |
| `SPEC_METHOD` | `dflash` |

---

## Repository layout

```text
.
├── README.md
├── VERSION / CHANGELOG.md / .env.example
├── compose/glm53.yaml
├── compose/overrides/          # gate-linear, logits64, etc. (optional)
├── image/                      # Dockerfile, entrypoint, GB10/SM121 patches
├── patches/                    # runtime overlays (glm5next, DFlash2, kpool, …)
├── runtime/                    # generated APC/KV/SM90, spinwait, jit, glm5next
├── scripts/                    # up/down/status/self-test, apply-serve-lane, …
├── bench/                      # soak / C1–C6 / accept helpers
├── benchmarks/                 # labeled result dirs
├── evidence/                   # A/B artifacts
└── docs/                       # architecture, deployment, performance, …
```

---

## Build

```bash
docker build -t glm53-spark:2x-sm121 image/
# or: docker compose -f compose/glm53.yaml build
```

| Pin | Value |
|-----|--------|
| Base | `vllm/vllm-openai:glm53-flash-arm64-cu130` |
| Mentat artifacts | `mmastrac/mentat-artifacts:0.6.0` |
| FlashInfer | `0.6.18.dev20260819` |
| NCCL | `nvidia-nccl-cu13==2.30.7` |

`image/patches/verify.py` checks SM121 / GB10 patches at build time (text-based; no GPU import).

---

## Benchmarks (snapshot)

2× Spark, clocks 2400, Golden `PROBE=0`:

| Label | Soak med tok/s | Accept | C6 | Notes |
|-------|----------------|--------|-----|-------|
| `upstream-baseline-clocks2400` | **35.87** | **0.430** | 81.0 | Golden band high |
| `checklist-final-spinwait-2400` | **35.09** | 0.421 | **87.2** | spin-wait later reverted |
| `20260912-longrun-60m` | **33.50** | 0.400 | — | sustained floor |
| sparse workspace reuse | **+0.70%** | flat | — | KEEP in kpool patch |

Re-run: `bash bench/bench_config.sh <label>` · `python3 bench/soak_timed.py …`

Accept ceiling (`PROBE=1`): reject mostly **B2** (~90% of B) → knobs alone cannot reach 0.50+. Real headroom: **target-aligned draft** or a different NVFP4 recipe — not more W4A16 micro-patches.

---

## Design rules

1. Do not copy unrelated 2× Spark recipes; pull from upstream only what is validated for this topology.
2. No TP=4 / 4-node launchers, host lists, or memory numbers.
3. vLLM only — no SGLang / Ray serving / extra inference server.
4. No new CUDA kernels, full FlashInfer/CUDA rebuilds, weight edits, or revision pins without explicit approval.
5. Secrets stay out of git (`.env` gitignored; ship `.env.example`).

---

## Troubleshooting (short)

| Symptom | Check |
|---------|--------|
| Worker never joins | Same `HEAD_HOST` on both nodes |
| NCCL / RoCE | `NCCL_SOCKET_IFNAME`, `NCCL_IB_HCA`, `NCCL_IB_GID_INDEX` |
| OOM / unified memory | `GPU_MEM_UTIL`, `KV_CACHE_MEMORY`, `MAX_NUM_SEQS`, `MAX_NUM_BATCHED_TOKENS` |
| Model re-download | Host `HOST_CACHE` mount (not `/var/tmp`) |
| Slow first request | Triton/JIT warmup — exclude from benchmarks |
| Vision / images in OpenCode | Declare `modalities.input: ["text","image"]` on the custom provider model |
| Noisy `/v1/models` from `127.0.0.1` | `status-server.py` on `:8082` (page auto-refresh ~5s) |

Full guide: [`docs/troubleshooting.md`](docs/troubleshooting.md)

---

## Known limitations

- **2× Spark / TP=2 only** — not a 4-node recipe.
- Golden gates (40–50 tok/s, 50–70% accept) **not met**; configuration frozen for stability.
- DFlash2 draft is trained for **BF16 zai GLM-5.3-Flash**; quantized targets inherit an accept ceiling.
- Hangul often feels slower: denser tokenization + lower draft accept — not a separate “Korean mode” knob.
- `DFLASH2_ACC_PROBE=1` is analysis-only (~−10% soak).
- `MAX_MODEL_LEN=1048576` is configured; ultra-long soak is a separate measurement.
- Draft license: `incoai/GLM-5.3-Flash-DFlash2` is **CC BY-NC-ND 4.0** (see upstream card).

---

## Docs index

| Doc | Contents |
|-----|----------|
| [`docs/golden-configuration.md`](docs/golden-configuration.md) | Production KEEP / REJECTED |
| [`docs/performance.md`](docs/performance.md) | Experiment ledger |
| [`docs/deployment.md`](docs/deployment.md) | Bring-up, SERVE_LANE |
| [`docs/architecture.md`](docs/architecture.md) | Control / data plane |
| [`docs/benchmark.md`](docs/benchmark.md) | Gates & tables |
| [`docs/applied-stack.md`](docs/applied-stack.md) | What is applied |
| [`docs/w4a16-vs-nvfp4.md`](docs/w4a16-vs-nvfp4.md) | Code-level comparison |
| [`CHANGELOG.md`](CHANGELOG.md) | Version history |

---

## Credits & license

- Upstream topology / GB10 patches: [mmastrac/glm-5.3-flash-4x-gx10](https://github.com/mmastrac/glm-5.3-flash-4x-gx10)
- Related: mmastrac, MiaAI-Lab, tonyd2wild, incoai, canada-quant, axiomofmind
- Runtime: vLLM, FlashInfer

Preserve upstream copyright / license headers on imported files. See repo `LICENSE` and each Hub model card for model terms.
