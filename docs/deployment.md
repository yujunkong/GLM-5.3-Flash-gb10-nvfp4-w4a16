# Deployment — 2× DGX Spark

## Prerequisites

- Two DGX Spark nodes (GB10 / SM121), RoCE connectivity
- Docker / Compose
- Host Hugging Face cache at `~/.cache/huggingface` (mounted into container)

## Quick path (target)

```bash
cp .env.example .env
# edit HEAD_HOST / WORKER_HOST / NCCL iface names
./scripts/up.sh
./scripts/status.sh
./scripts/self-test.sh
```

## HF cache policy

- Host: `~/.cache/huggingface`
- Container: `/root/.cache/huggingface`
- No `/var/tmp` model cache, no revision pinning

*(Expand with compose profiles and per-node ROLE steps after launcher implementation.)*
