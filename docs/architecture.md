# Architecture — 2× DGX Spark / TP=2

## Topology

```
192.168.100.10  HEAD / rank 0
        │ RoCE (ConnectX-7)
192.168.100.20  WORKER / rank 1
TP=2  NNODES=2  engine=vLLM (mentat Ray shim)
```

## Control plane

| Component | Where | Role |
|-----------|--------|------|
| `mentatd` | every node (`compose/mentatd.yaml`) | Placement / `ray start` target :6379 |
| `mentatd-serve` | head optional (`compose/mentatd-serve.yaml`) | Front door :6381 |
| `glm53` | every node (`compose/glm53.yaml`) | TP rank (head serves :8002) |

## Data plane

- Weights: Hugging Face Hub → host `HOST_CACHE/huggingface` → container `/root/.cache/huggingface`
- No revision pin; no `/var/tmp` model cache
- NCCL over RoCE (`/dev/infiniband`, `network_mode: host`)

## Image

- Tag: `glm53-spark:2x-sm121`
- Base: `vllm/vllm-openai:glm53-flash-arm64-cu130` (+ upstream SM121 patch stack)

See `docs/upstream-analysis.md` for Copy/Rewrite/Drop decisions.
