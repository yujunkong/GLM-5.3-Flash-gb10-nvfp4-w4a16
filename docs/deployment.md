# Deployment — 2× DGX Spark

## Prerequisites

- Same repo checkout on both nodes (`REMOTE_ROOT`, default = this path)
- Docker + NVIDIA runtime; image `glm53-spark:2x-sm121` present on both
- SSH key auth as `SSH_USER`
- RoCE fabric addresses on `CLUSTER_SUBNET` (default `192.168.100.`)
- Host cache at `HOST_CACHE` (default `~/.cache`) including Hugging Face Hub downloads

## Configure

```bash
cp .env.example .env
# Edit ROLE/NODE_RANK/MENTAT_* per node; HEAD_HOST identical on both
```

Worker node `.env` differences:

```bash
ROLE=worker
NODE_RANK=1
MENTAT_NODE_IP=192.168.100.20
MENTAT_PEERS=192.168.100.10:6379
```

## Build image (once per node, or build once and docker save/load)

```bash
docker build -t glm53-spark:2x-sm121 image/
```

## Start / stop

```bash
./scripts/up.sh
./scripts/status.sh
./scripts/self-test.sh
./scripts/down.sh                 # glm53 only
STOP_MENTAT=1 ./scripts/down.sh   # also mentatd
```

`up.sh` order: config → SSH → docker → iface → image → mentatd → mentatd-serve → glm53 worker+head → `/v1/models` → self-test.

## Cold start

First boot may spend a long time on Triton/JIT. Treat the first completion as warmup (excluded from benchmarks).
