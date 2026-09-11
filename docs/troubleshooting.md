# Troubleshooting

| Area | Check |
|------|--------|
| Worker join | Same `HEAD_HOST` on every node; `MENTAT_PEERS` points at the other node; `mentatd` up on both |
| Worker uses wrong head | Worker must not set `HEAD_HOST` to its own IP |
| NCCL / RoCE | `NCCL_SOCKET_IFNAME` exists (`up.sh` check); leave `NCCL_IB_GID_INDEX` empty if unsure (site pin `3` is optional) |
| Model missing | Hub id in `MODEL`; weights under `HOST_CACHE/huggingface`; optional `HF_TOKEN` |
| SM121 / MLA | Image built from this repo's `image/` (verify.py passed); `VLLM_GLM53_CUDA_SPARSE_MLA` if using plugin |
| top-k | Built-in `gb10_topk_fallback` — do not add new CUDA kernels |
| OOM / UMA | `GPU_MEM_UTIL=0.88`, `MAX_NUM_SEQS=6`, `worker_memory_cap`; do not jump to 0.90 |
| Re-download | Confirm `HOST_CACHE` mount; never use `/var/tmp` |
| Slow first request | Cold Triton/JIT — exclude from benchmark |
| Build fail | `image/patches/verify.py` markers after base tag move |

Logs: `./scripts/logs.sh` (head) or `WORKER=1 ./scripts/logs.sh`.
