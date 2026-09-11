# Architecture — 2× DGX Spark / TP=2

## Topology

- Node 0 `HEAD` @ `192.168.100.10` → rank 0
- Node 1 `WORKER` @ `192.168.100.20` → rank 1
- Network: ConnectX-7 RoCE
- Engine: vLLM only (`tensor_parallel_size=2`)

## Design notes

- Not a mechanical shrink of the 4-node upstream recipe
- Memory / KV / batching sized for unified memory on two Sparks
- All nodes share the same `HEAD_HOST`

## Patches (planned import)

See `image/patches/` and README § Architecture.

*(Fill with concrete diagrams and data-plane notes after upstream analysis.)*
