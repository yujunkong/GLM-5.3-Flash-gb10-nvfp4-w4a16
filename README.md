# GLM-5.3-Flash — 2× DGX Spark (GB10 / SM121)

NVIDIA DGX Spark 2대 전용 GLM-5.3-Flash serving recipe (vLLM, TP=2).

This repository is specifically designed for **2× NVIDIA DGX Spark / GB10 / SM121**.  
It is **not** a mechanically reduced 4-node recipe. The distributed topology, memory configuration, launcher, and runtime configuration are designed specifically for two DGX Spark nodes.

| Item | Value |
|------|--------|
| GPU | NVIDIA GB10 |
| Architecture | SM121 / compute capability 12.1a |
| Nodes | 2 (1 GPU per node) |
| Tensor Parallel | TP=2 |
| Inference engine | vLLM |
| Upstream reference | [mmastrac/glm-5.3-flash-4x-gx10](https://github.com/mmastrac/glm-5.3-flash-4x-gx10) |

Upstream은 4× ASUS GX10 / GB10, TP=4, 4-node topology 대상이다. 본 repo는 4대→2대 축소가 아니라, 처음부터 2× DGX Spark / GB10 / SM121 / TP=2 전용으로 재구성한다.

---

## 1. Project overview

- **목표**: `./scripts/up.sh` 한 번으로 2× DGX Spark에서 GLM-5.3-Flash serving을 기동
- **엔진**: vLLM만 사용 (SGLang / Ray serving / 별도 custom inference server 미사용)
- **기준 upstream**: `mmastrac/glm-5.3-flash-4x-gx10`의 GB10/SM121 해결 방법을 참고하되, 4-node 전용 요소는 제거
- **모델 경로**: Hugging Face Hub → `~/.cache/huggingface` → container `/root/.cache/huggingface` → vLLM

### 핵심 설계 원칙

1. **기존 다른 2× DGX Spark recipe를 복사하지 않는다** — upstream(`mmastrac/glm-5.3-flash-4x-gx10`) 기준으로 필요한 기능만 검증 후 가져온다.
2. **4-node 설정을 그대로 쓰지 않는다** — TP=4, 4-node launcher/host list/topology/readiness, switchless 4-node 설정, 4-node memory·benchmark·routing 제거. runtime config는 전부 2-node 기준으로 재작성.
3. **vLLM만 사용** — upstream mentat 관련 코드는 distributed orchestration에 필요한 부분만 검토하고, 2-node에서 불필요하면 제거.

### 정상 기동 상태 (목표)

```
HEAD        OK
WORKER      OK
GPU/RANK 0  OK
GPU/RANK 1  OK
TP          2
NNODES      2
MODEL       canada-quant/glm-5.3-w4a16-mtp
DRAFT       incoai/GLM-5.3-Flash-DFlash2
DFlash2     enabled
DFlash k    7
KV CACHE    OK
HEALTH      OK
OPENAI API  OK
```

---

## 2. Hardware

| Role | Host | IP | Notes |
|------|------|-----|--------|
| HEAD | Node 0 | `192.168.100.10` | rank 0 |
| WORKER | Node 1 | `192.168.100.20` | rank 1 |

| Spec | Value |
|------|--------|
| GPU | NVIDIA GB10 × 1 per node |
| Total GPU | 2 |
| Architecture | SM121 / compute capability 12.1a |
| Tensor Parallel | 2 |
| Network | ConnectX-7, RoCE |

IP는 Python/Dockerfile에 하드코딩하지 않는다. `.env` / `.env.example`로만 관리한다.

---

## 3. Architecture

```
192.168.100.10
HEAD / rank 0
        │
        │ RoCE
        │
192.168.100.20
WORKER / rank 1

TP=2
NNODES=2
```

- DGX Spark 1대당 GPU 1개 → Node 0 = rank 0, Node 1 = rank 1
- `TP_SIZE=2`, `NNODES=2`, vLLM `tensor_parallel_size=2`
- 모든 노드가 동일한 `HEAD_HOST`를 사용한다 (Worker가 자기 IP를 `HEAD_HOST`로 쓰지 않음)

```
HEAD:
  HEAD_HOST=192.168.100.10
  ROLE=head

WORKER:
  HEAD_HOST=192.168.100.10
  ROLE=worker
```

### 목표 Repository 구조

필요하지 않은 파일은 만들지 않는다.

```text
.
├── README.md
├── LICENSE
├── .gitignore
├── .env.example
│
├── compose/
│   ├── glm53.yaml
│   └── overrides/
│       ├── dflash2.yaml
│       └── production.yaml
│
├── image/
│   ├── Dockerfile
│   ├── entrypoint.sh
│   ├── verify.py
│   └── patches/
│       ├── glm53-flash_SM121.py
│       ├── gb10_topk_fallback.py
│       ├── gb10_plugin_backend.py
│       ├── worker_memory_cap.py
│       ├── spark_mem_trace.py
│       └── link_cuda_headers.sh
│
├── scripts/
│   ├── up.sh
│   ├── down.sh
│   ├── status.sh
│   ├── logs.sh
│   └── self-test.sh
│
└── docs/
    ├── architecture.md
    ├── deployment.md
    ├── troubleshooting.md
    └── benchmark.md
```

### Upstream에서 가져올 핵심 코드

`mmastrac/glm-5.3-flash-4x-gx10`을 **실제 코드 기준**으로 분석한다 (README만 보고 구현하지 않음). 우선 검토 대상:

| Patch / script | Purpose |
|----------------|---------|
| `glm53-flash_SM121.py` | GLM-5.3-Flash sparse MLA가 GB10/SM121에서 동작하도록 SM121 patch. 단순 복사가 아니라: vLLM 코드 확인 → patch 대상 확인 → 적용 → build-time 검증 → runtime import 검증 |
| `gb10_topk_fallback.py` | GB10 `persistent_topk` 문제 시 upstream fallback. **새 top-k CUDA kernel 작성 금지** |
| `gb10_plugin_backend.py` | SM121에서 GLM sparse MLA backend 선택. `VLLM_GLM53_CUDA_SPARSE_MLA` 및 SM121 backend 경로 확인 |
| `worker_memory_cap.py` | unified memory에서 worker/indexer/KV 외 allocation 증가로 인한 exhaustion 방지 |
| `spark_mem_trace.py` | 메모리 추적. production에서 과도한 verbose logging 지양 |
| `link_cuda_headers.sh` | DFlash2/Triton/JIT용 CUDA headers 접근 확인 후 필요 시 적용 |

### `verify.py` (build-time)

Docker build 시 patch 적용 여부를 **text/function/path 기반**으로 자동 검증한다.  
Driver 없는 build 단계에서 vLLM CUDA platform import로 검사하는 방식은 사용하지 않는다.

최소 확인: SM121 patch, GB10 top-k fallback, GB10 plugin backend, worker memory cap, CUDA headers, DFlash2 support.  
필수 patch 누락 시 Docker build 실패.

### Docker image

Upstream의 GB10/SM121 지원 방식을 우선 사용한다. 확인 항목: CUDA / vLLM / PyTorch / FlashInfer / Triton version, DFlash2 support, SM121 compatibility.  
가능하면 upstream image 구조를 활용하되, 4-node 전용 dependency·불필요 service는 제거한다.

### SM121 native rebuild (금지에 가까움)

다음만으로 **수행하지 않는다**: full CUDA rebuild, FlashInfer full rebuild, `_C.abi3.so` / `_moe_C.abi3.so` custom replacement, new Marlin kernel, new sparse MLA kernel.  
먼저 runtime profiler로 병목을 확인하고, native SM121 build는 필요성이 확인될 때만 별도 작업으로 진행한다.

### Profiler

기본 production에서는 profiler 비활성화. 필요 시 debug 설정 예:

```text
profiler=torch
torch_profiler_dir=/cache/step4
```

프로파일 결과는 production model cache와 분리한다.

---

## 4. Model

| Variable | Default |
|----------|---------|
| `MODEL` | `canada-quant/glm-5.3-w4a16-mtp` |
| `DRAFT_MODEL` | `incoai/GLM-5.3-Flash-DFlash2` |

### Revision 고정 금지

특정 Hugging Face revision/hash를 사용하지 않는다.

사용하지 않음: `MODEL_REVISION=`, `--revision`, revision hash, commit hash 고정.

```bash
vllm serve "$MODEL"
```

형태로만 지정하여, Hub에 새 revision이 올라와도 코드 수정 없이 사용할 수 있게 한다.  
README에 모델 revision hash를 기록하지 않는다.

### DFlash2

기본 활성화.

| Variable | Default |
|----------|---------|
| `DFLASH_TOKENS` | `7` (`num_speculative_tokens=7`) |
| `DFLASH_SELECTOR_TOP_K` | `32` |
| `DFLASH_WALK_MODE` | `edge` |
| `DFLASH2_ACC_PROBE` | `0` |

DFlash2 관련 설정은 `compose/overrides/dflash2.yaml`에 분리한다.

### Production baseline (재 A/B 테스트하지 않음)

이미 검증된 기본 production profile:

```text
DFLASH_SELECTOR_TOP_K=32
DFLASH_WALK_MODE=edge
DFLASH2_ACC_PROBE=0
DFLASH_TOKENS=7

MOE_BACKEND=marlin

ENFORCE_EAGER=1
DISABLE_FLASHINFER_AUTOTUNE=1
ASYNC_SCHEDULING=1
APPLY_APC_PATCH=1
```

### Context / batching / KV

| Variable | Default | Notes |
|----------|---------|--------|
| `MAX_MODEL_LEN` | `1048576` | bring-up 시 `262144` → `524288` → `1048576` 순 검증 권장. 1M이 불안정하면 README에 실제 검증된 최대값 기록 |
| `MAX_NUM_SEQS` | `6` | 이후 concurrency benchmark 가능하도록 |
| `MAX_NUM_BATCHED_TOKENS` | `8192` | 4-node upstream `16384`를 무조건 가져오지 않음. 2× unified memory 안정성 우선 |
| `KV_CACHE_MEMORY` | (환경변수) | 4-node upstream 값 복사 금지. 2× DGX Spark unified memory 기준으로 재설정 |
| `GPU_MEM_UTIL` | `0.88` (예) | 무조건 최대 메모리 할당하지 않음 |

---

## 5. Hugging Face cache

| Policy | Value |
|--------|--------|
| Model | `canada-quant/glm-5.3-w4a16-mtp` |
| Revision | **not pinned** |
| Cache | standard Hugging Face cache |
| Host default | `~/.cache/huggingface` |
| Container (root) | `/root/.cache/huggingface` |

### 사용하지 않는 경로

`/var/tmp`, `/var/tmp/model`, `/var/tmp/huggingface`, 프로젝트 내부 `model/`, 별도 custom model cache.

### Docker cache persistence

container 삭제 후에도 재다운로드하지 않도록 host HF cache를 mount한다.

```yaml
volumes:
  - ${HF_CACHE:-$HOME/.cache/huggingface}:/root/.cache/huggingface
```

Docker Compose expansion이 정상 동작하는지 검증한다. 필요 시 `.env`에 `HF_CACHE=...` 명시 가능.

### `HF_HOME`

가능하면 `HF_HOME`을 강제로 변경하지 않고 Hugging Face 기본 cache 동작을 유지한다 (`~/.cache/huggingface`).

### HF Token

공개 모델이면 `HF_TOKEN`을 필수로 요구하지 않는다. 필요할 때만 optional로 제공한다. Token을 Docker image에 넣지 않는다.

---

## 6. Network

기본 NCCL / RoCE 환경변수 (절대 고정값이 아님 — `.env.example`에 변경 방법 명시, `up.sh`에서 interface 존재 검사):

```text
NCCL_IB_GID_INDEX=3
NCCL_SOCKET_IFNAME=enp1s0f0np0
NCCL_IB_HCA=rocep1s0f0
```

물리 링크: ConnectX-7, RoCE.

---

## 7. Requirements

- 2× NVIDIA DGX Spark (GB10 / SM121)
- Docker / Docker Compose
- ConnectX-7 RoCE 네트워크 (노드 간)
- Hugging Face Hub 접근 (공개 모델이면 token optional)
- Host Hugging Face cache (`~/.cache/huggingface`)

### 금지사항 (승인 없이 수행하지 않음)

- 새 CUDA kernel 작성
- CUDA / NCCL / FlashInfer 전체 rebuild
- 모델 weight·quantization·DFlash2 weight 수정
- TP4 설정 추가
- Ray / SGLang / 새 inference server 도입
- revision pinning
- `/var/tmp` model cache
- Docker image 내부에 model weight 포함

### 코드 품질

- 스크립트: `set -euo pipefail`
- 환경변수: 기본값 + validation 명확화
- 잘못된 IP / interface / model은 startup 전 실패
- secret은 Git에 commit하지 않음 (`.env` → `.gitignore`, `.env.example`만 포함)

### Static validation (실기 전)

- `docker compose config`
- YAML / Dockerfile validation
- shellcheck
- environment validation
- patch verification

구현 전 upstream의 `compose/`, `image/`, `scripts/`, `patches/`, Dockerfile, environment·DFlash2·memory configuration을 **실제 코드로** 확인한다.

---

## 8. Build

Image는 upstream GB10/SM121 지원 방식을 우선 사용하고, `image/patches/` + `verify.py`로 patch 적용을 build 시 검증한다.

(구현 후 구체적인 `docker build` / compose build 명령을 이 섹션에 추가한다.)

---

## 9. Configuration

`.env.example` 최소 항목 (Revision 관련 변수는 만들지 않음):

```bash
MODEL=canada-quant/glm-5.3-w4a16-mtp
DRAFT_MODEL=incoai/GLM-5.3-Flash-DFlash2

HEAD_HOST=192.168.100.10
WORKER_HOST=192.168.100.20

TP_SIZE=2
NNODES=2

MAX_MODEL_LEN=1048576
MAX_NUM_SEQS=6
MAX_NUM_BATCHED_TOKENS=8192

DFLASH_TOKENS=7
DFLASH_SELECTOR_TOP_K=32
DFLASH_WALK_MODE=edge
DFLASH2_ACC_PROBE=0

MOE_BACKEND=marlin
ENFORCE_EAGER=1
DISABLE_FLASHINFER_AUTOTUNE=1
ASYNC_SCHEDULING=1
APPLY_APC_PATCH=1

NCCL_IB_GID_INDEX=3
NCCL_SOCKET_IFNAME=enp1s0f0np0
NCCL_IB_HCA=rocep1s0f0
```

Compose:

- 기본: `compose/glm53.yaml`
- override: `compose/overrides/dflash2.yaml`, `compose/overrides/production.yaml`

---

## 10. Start

```bash
./scripts/up.sh
```

실행 순서:

1. configuration validation  
2. host connectivity  
3. Docker check  
4. network/interface check  
5. image check  
6. worker preparation  
7. head preparation  
8. distributed vLLM startup  
9. rank readiness  
10. `/health`  
11. `/v1/models`  
12. inference self-test  

실패한 단계는 명확히 표시되어야 한다.

종료:

```bash
./scripts/down.sh
```

관련 container/process를 안전하게 종료한다. **Hugging Face cache는 삭제하지 않는다.**

---

## 11. Health check

```bash
./scripts/status.sh
```

표시 항목:

HEAD, WORKER, GPU/rank, TP, NNODES, MODEL, DRAFT MODEL, vLLM version, CUDA version, FlashInfer version, DFlash2, DFlash tokens, DFlash selector top-k, DFlash walk mode, MoE backend, KV cache, `MAX_MODEL_LEN`, `MAX_NUM_SEQS`, `MAX_NUM_BATCHED_TOKENS`, NCCL interface, RoCE GID, health, OpenAI API.

```bash
./scripts/self-test.sh
```

순서: `/health` → `/v1/models` → simple completion → reasoning request → tool-call parser → DFlash2 warmup → second completion.  
첫 요청은 Triton/JIT warmup일 수 있으므로 benchmark에서 제외한다.

로그:

```bash
./scripts/logs.sh
```

---

## 12. First inference

### Cold start vs warm inference

첫 실행은 Triton/JIT compile 때문에 오래 걸릴 수 있다. **cold startup**과 **warm inference**를 구분한다.  
첫 warmup 요청은 latency/throughput 측정에서 제외한다.

(엔드포인트·예제 curl은 구현 후 이 섹션에 추가한다.)

---

## 13. Benchmark

최소 concurrency: **C1, C2, C4, C6**.

측정 항목: TTFT, Prefill tok/s, Decode tok/s, Output tok/s, DFlash acceptance, GPU utilization, Memory.

- 첫 warmup 요청은 제외
- 결과는 `docs/benchmark.md`에 기록
- **실제 측정하지 않은 숫자는 README에 기재하지 않는다**

---

## 14. Troubleshooting

상세 가이드: `docs/troubleshooting.md` (구현 시 작성).

자주 확인할 항목:

| 증상 / 영역 | 점검 |
|-------------|------|
| Worker가 head에 못 붙음 | `HEAD_HOST`가 모든 노드에서 head IP인지, Worker가 자기 IP를 head로 쓰지 않는지 |
| NCCL / RoCE | `NCCL_SOCKET_IFNAME`, `NCCL_IB_HCA`, `NCCL_IB_GID_INDEX`, interface 존재 여부 (`up.sh` 검사) |
| SM121 / sparse MLA | SM121 patch, `gb10_plugin_backend`, `VLLM_GLM53_CUDA_SPARSE_MLA` |
| top-k | `gb10_topk_fallback` (새 CUDA kernel 금지) |
| OOM / unified memory | `worker_memory_cap`, KV/`GPU_MEM_UTIL`, `MAX_NUM_SEQS`, `MAX_NUM_BATCHED_TOKENS` |
| 모델 재다운로드 | host `~/.cache/huggingface` mount 여부 (`/var/tmp` 미사용) |
| Cold start 지연 | Triton/JIT warmup — warm과 구분, benchmark에서 제외 |
| Patch 누락 | `verify.py` build 실패 여부 |

---

## 15. Known limitations

- 본 recipe는 **2× DGX Spark / TP=2 전용**. 4-node / TP=4는 지원하지 않는다.
- `MAX_MODEL_LEN=1048576`은 목표값이며, bring-up 시 단계적 검증이 필요하다. 실제 검증된 최대 context는 측정 후 기록한다.
- KV cache / memory util은 2-node unified memory 기준으로 보수적으로 시작한다. 무조건 최대 할당하지 않는다.
- SM121 native cubin 부재만으로 full CUDA/FlashInfer rebuild나 custom `.so` 교체를 하지 않는다.
- Production profiler는 기본 off.
- 측정하지 않은 benchmark 수치는 문서에 넣지 않는다. 미실행 검증은 `NOT RUN`으로만 표시한다.

---

## 16. Upstream credits / License

주요 참고·관련:

- [mmastrac/glm-5.3-flash-4x-gx10](https://github.com/mmastrac/glm-5.3-flash-4x-gx10)
- mmastrac, MiaAI-Lab, tonyd2wild, incoai, canada-quant
- vLLM, FlashInfer

upstream에서 코드를 가져올 때 파일별 source와 해당 프로젝트 LICENSE / attribution을 확인한다.  
원본 copyright/license header가 필요한 파일은 유지한다.

---

## Appendix A — Implementation phases (구축 순서)

1. **Upstream 분석** — 코드 작성 전 구조 분석. 표: File / Purpose / Relevant to 2× Spark? / Copy·Rewrite·Drop / Reason  
2. **2-node Architecture** — HEAD/WORKER, TP=2, RoCE topology 재설계  
3. **Image** — 필요한 SM121/GB10 patch만 적용  
4. **Compose** — 2-node 전용 compose  
5. **Launcher** — `up.sh` / `down.sh` / `status.sh` / `logs.sh` / `self-test.sh`  
6. **Configuration** — `.env.example` (위 Configuration 섹션)  
7. **Documentation** — `docs/architecture.md`, `deployment.md`, `troubleshooting.md`, `benchmark.md`  
8. **Static validation** — compose config, shellcheck, YAML/Dockerfile, patch, environment  
9. **Git** — 권장 commit 예:  
   - `feat: add 2-node DGX Spark base recipe`  
   - `feat: add GB10 and SM121 runtime patches`  
   - `feat: add DFlash2 TP2 configuration`  
   - `feat: add 2-node launcher and health checks`  
   - `docs: add deployment and troubleshooting guide`

## Appendix B — Completion report template

작업 완료 보고 시 아래 형식을 사용한다. 추측으로 PASS를 적지 않는다. 미실행은 `NOT RUN`.

```text
## Repository
<repository name>

## Upstream
mmastrac/glm-5.3-flash-4x-gx10

## Architecture
2× DGX Spark / TP=2 / NNODES=2

## Model
MODEL=
DRAFT_MODEL=
Revision: NOT PINNED
HF Cache: ~/.cache/huggingface

## Imported
- SM121 patch
- GB10 top-k fallback
- GB10 plugin backend
- worker memory cap
- memory trace
- CUDA header support

## Rewritten
- compose
- launcher
- distributed topology
- memory configuration
- network configuration
- health checks

## Removed
- TP4
- 4-node launcher
- 4-node topology
- unnecessary mentat components
- unnecessary experimental code

## Validation
docker compose config: PASS/FAIL
shellcheck: PASS/FAIL
patch verification: PASS/FAIL
Docker build: PASS/FAIL
2-node startup: PASS/FAIL/NOT RUN
inference: PASS/FAIL/NOT RUN

## Git
commit:
push:

## Remaining
(실제 DGX Spark에서 추가 검증할 항목만)
```
