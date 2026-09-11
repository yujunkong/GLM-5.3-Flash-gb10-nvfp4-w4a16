# Troubleshooting

| Area | Check |
|------|--------|
| Worker join | Same `HEAD_HOST` on every node; worker must not use its own IP as head |
| NCCL / RoCE | `NCCL_SOCKET_IFNAME`, `NCCL_IB_HCA`, `NCCL_IB_GID_INDEX`; iface exists (`up.sh`) |
| SM121 / MLA | SM121 patch, plugin backend, `VLLM_GLM53_CUDA_SPARSE_MLA` |
| top-k | `gb10_topk_fallback` only — no new CUDA kernels |
| OOM | `worker_memory_cap`, KV / `GPU_MEM_UTIL`, `MAX_NUM_SEQS`, `MAX_NUM_BATCHED_TOKENS` |
| Re-download | Host HF cache mount present |
| Slow first request | Cold Triton/JIT warmup — not a benchmark sample |
| Build fail | `image/verify.py` patch markers |

*(Add real log snippets after bring-up.)*
