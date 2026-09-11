# GLM-5.3-Flash-gb10-nvfp4-w4a16
좋아. 아래 내용을 Cursor에게 처음 전달하는 최종 작업 지시서로 그대로 넣으면 된다.
기존 repo나 이전 작업을 전제로 하지 않고, mmastrac/glm-5.3-flash-4x-gx10을 참고해 새 repo를 2× DGX Spark 전용으로 구축하는 내용으로 정리했다.

GLM-5.3-Flash — 2× DGX Spark 전용 신규 Repository 구축

1. 작업 목표

현재 repository를 NVIDIA DGX Spark 2대 전용 GLM-5.3-Flash serving recipe로 구축한다.

주요 참고 upstream:

https://github.com/mmastrac/glm-5.3-flash-4x-gx10

이 repository의 구조와 실제 GB10/SM121 관련 해결 방법을 적극적으로 참고한다.

하지만 upstream은:

4× ASUS GX10 / GB10
TP=4
4-node topology

를 대상으로 한다.

이번 repository는 이를 단순히 4대 → 2대로 줄이는 것이 아니다.

처음부터 2× DGX Spark / GB10 / SM121 / TP=2 전용으로 재구성한다.

---

2. 대상 하드웨어

Node 0 — HEAD

DGX Spark / GB10
IP: 192.168.100.10
Role: head

Node 1 — WORKER

DGX Spark / GB10
IP: 192.168.100.20
Role: worker

환경:

GPU: NVIDIA GB10
Architecture: SM121 / compute capability 12.1a
GPU count: 1 GPU per node
Total GPU: 2
Tensor Parallel: 2
Nodes: 2

네트워크:

ConnectX-7
RoCE

---

3. 핵심 설계 원칙

다음 원칙을 반드시 지킨다.

3.1 기존 다른 2× DGX Spark repository를 복사하지 않는다

이 repository는 다음 upstream을 기준으로 새롭게 구성한다.

mmastrac/glm-5.3-flash-4x-gx10

기존에 존재하는 다른 GLM recipe의 실험적인 설정이나 불필요한 패치는 가져오지 않는다.

필요한 기능만 검증해서 가져온다.

---

3.2 4-node 설정을 그대로 사용하지 않는다

다음 4-node 전용 요소는 제거한다.

TP=4
4-node launcher
4-node host list
4-GPU topology
4-node readiness
switchless 4-node 전용 설정
4-node memory 계산
4-node benchmark configuration
4-node routing

모든 runtime configuration은 2-node 기준으로 다시 작성한다.

---

3.3 vLLM을 사용한다

Inference engine은 vLLM이다.

사용하지 않는다:

SGLang
Ray 기반 serving
별도 custom inference server
새로운 inference engine

upstream의 mentat 관련 코드는 필요한 distributed orchestration 부분만 검토한다.

2-node 환경에서 불필요하다면 제거한다.

---

4. 모델

기본 모델:

MODEL=canada-quant/glm-5.3-w4a16-mtp

Draft model:

DRAFT_MODEL=incoai/GLM-5.3-Flash-DFlash2

중요 — Revision 고정 금지

특정 Hugging Face revision/hash를 사용하지 않는다.

다음은 사용하지 않는다.

MODEL_REVISION=
--revision
revision hash
commit hash 고정

즉:

vllm serve "$MODEL"

형태로 모델을 지정한다.

Hugging Face에 새 모델 revision이 올라오면 별도의 코드 수정 없이 사용할 수 있는 구조로 만든다.

---

5. Hugging Face Cache

Hugging Face의 기본 cache 경로를 사용한다.

모델을 다음 경로에 저장하지 않는다.

/var/tmp
/var/tmp/model
/var/tmp/huggingface
프로젝트 내부 model/
별도 custom model cache

기본 cache:

~/.cache/huggingface

Docker container 내부에서는 root로 실행되는 경우:

/root/.cache/huggingface

를 사용한다.

---

5.1 Docker cache persistence

container를 삭제해도 모델을 다시 다운로드하지 않도록 host의 Hugging Face cache를 container에 mount한다.

예:

volumes:
  - ${HF_CACHE:-$HOME/.cache/huggingface}:/root/.cache/huggingface

Docker Compose에서 실제 expansion이 정상적으로 동작하는지 검증한다.

필요하다면 ".env"에서:

HF_CACHE=/root/.cache/huggingface

등으로 명시할 수 있다.

중요한 것은 Hugging Face 기본 cache를 사용하고 "/var/tmp" 등을 사용하지 않는 것이다.

---

5.2 HF_HOME

가능하면 "HF_HOME"을 강제로 변경하지 않는다.

Hugging Face의 기본 cache 동작을 유지한다.

즉:

~/.cache/huggingface

를 기본으로 사용한다.

---

5.3 HF Token

공개 모델이면 HF_TOKEN을 필수로 요구하지 않는다.

필요한 경우에만:

HF_TOKEN=

을 optional로 제공한다.

Token은 Docker image에 넣지 않는다.

---

6. Repository 구조

다음과 같은 구조를 목표로 한다.

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

실제로 필요하지 않은 파일은 만들지 않는다.

---

7. Upstream에서 가져올 핵심 코드

"mmastrac/glm-5.3-flash-4x-gx10"을 분석해서 다음 기능을 우선 검토한다.

7.1 SM121 GLM patch

glm53-flash_SM121.py

목적:

GLM-5.3-Flash의 sparse MLA 경로가 GB10/SM121에서 정상 동작하도록 하는 upstream의 방법을 사용한다.

단순 복사가 아니라:

1. 현재 사용할 vLLM 코드 확인
2. patch 대상 확인
3. patch 적용
4. build-time 검증
5. runtime import 검증

순서로 처리한다.

---

7.2 GB10 top-k fallback

gb10_topk_fallback.py

GB10에서 "persistent_topk" 경로 문제가 발생하는 경우 upstream fallback을 사용하는 구조를 적용한다.

새로운 top-k CUDA kernel을 만들지 않는다.

---

7.3 GB10 plugin backend

gb10_plugin_backend.py

SM121에서 GLM sparse MLA backend 선택 문제가 발생하지 않도록 upstream 방식을 사용한다.

특히:

VLLM_GLM53_CUDA_SPARSE_MLA

와 SM121 backend 선택 경로를 확인한다.

---

7.4 Unified Memory worker cap

worker_memory_cap.py

DGX Spark는 unified memory 구조이므로 이 부분을 중요하게 검토한다.

목적:

worker allocation 증가
indexer allocation 증가
KV cache 외 allocation 증가
unified memory exhaustion

을 방지한다.

---

7.5 Memory trace

spark_mem_trace.py

메모리 문제를 추적할 수 있도록 유지한다.

Production에서는 필요 이상으로 verbose한 logging을 하지 않도록 한다.

---

7.6 CUDA headers

link_cuda_headers.sh

DFlash2/Triton 또는 runtime JIT에 필요한 CUDA headers가 base image에서 정상적으로 접근 가능한지 확인한다.

필요한 경우 적용한다.

---

8. verify.py

Docker build 시 patch가 실제로 적용됐는지 자동 검증한다.

Driver가 없는 Docker build 단계에서 vLLM CUDA platform을 import해서 검사하는 방식은 사용하지 않는다.

text/function/path 기반으로 검사한다.

최소 확인:

SM121 patch
GB10 top-k fallback
GB10 plugin backend
worker memory cap
CUDA headers
DFlash2 support

필수 patch가 누락되면 Docker build가 실패하도록 한다.

---

9. Tensor Parallel

반드시 2-node 구성:

TP_SIZE=2
NNODES=2

vLLM:

tensor_parallel_size=2

를 사용한다.

DGX Spark 한 대당 GPU 하나이므로:

Node 0 → rank 0
Node 1 → rank 1

구조를 기준으로 한다.

---

10. Node configuration

".env.example":

HEAD_HOST=192.168.100.10
WORKER_HOST=192.168.100.20

TP_SIZE=2
NNODES=2

IP를 Python 코드나 Dockerfile에 하드코딩하지 않는다.

---

11. HEAD_HOST 주의

모든 node가 head 주소를 동일하게 사용한다.

HEAD:

HEAD_HOST=192.168.100.10
ROLE=head

WORKER:

HEAD_HOST=192.168.100.10
ROLE=worker

Worker가 자신의 IP를 HEAD_HOST로 사용하지 않도록 한다.

---

12. NCCL / RoCE

현재 환경을 기본값으로 지원한다.

NCCL_IB_GID_INDEX=3
NCCL_SOCKET_IFNAME=enp1s0f0np0
NCCL_IB_HCA=rocep1s0f0

단, interface 이름은 하드코딩된 절대값으로 간주하지 않는다.

".env.example"에 변경 방법을 명시한다.

실제 시스템에서 interface가 존재하는지 "up.sh"가 검사할 수 있도록 한다.

---

13. DFlash2

DFlash2를 기본 활성화한다.

기본값:

DFLASH_TOKENS=7

vLLM speculative decoding:

num_speculative_tokens=7

을 사용한다.

DFlash2 관련 configuration은:

compose/overrides/dflash2.yaml

에 분리한다.

---

14. 이미 검증된 Production 설정

다음 값은 기본 production profile로 적용한다.

DFLASH_SELECTOR_TOP_K=32
DFLASH_WALK_MODE=edge
DFLASH2_ACC_PROBE=0
DFLASH_TOKENS=7

MOE_BACKEND=marlin

ENFORCE_EAGER=1
DISABLE_FLASHINFER_AUTOTUNE=1
ASYNC_SCHEDULING=1
APPLY_APC_PATCH=1

이 값들은 새 repository에서 다시 A/B 테스트하지 않는다.

이미 검증된 baseline이다.

---

15. Context

초기 안정성 검증과 최대 context 검증을 분리한다.

기본 설정:

MAX_MODEL_LEN=1048576

단, 실제 bring-up 시 다음 순서로 검증할 수 있도록 한다.

262144
524288
1048576

1M context가 실제로 안정적으로 동작하지 않는다면 README에 실제 검증된 최대값을 기록한다.

---

16. KV Cache

KV cache는 4-node upstream 값을 그대로 복사하지 않는다.

2× DGX Spark unified memory를 기준으로 다시 설정한다.

처음에는 안정성을 우선한다.

환경변수로 조절 가능하게 한다.

예:

KV_CACHE_MEMORY=
GPU_MEM_UTIL=0.88

단, 실제 사용할 값은 upstream과 현재 2-node 환경의 메모리 요구량을 확인한 뒤 결정한다.

무조건 최대 메모리를 할당하지 않는다.

---

17. max-num-seqs

기본:

MAX_NUM_SEQS=6

으로 시작한다.

이후 concurrency benchmark를 별도로 수행할 수 있도록 한다.

---

18. max-num-batched-tokens

기본:

MAX_NUM_BATCHED_TOKENS=8192

으로 시작한다.

4-node upstream의 "16384"를 무조건 가져오지 않는다.

2× DGX Spark unified memory 환경에서 안정성을 우선한다.

---

19. Docker Image

Docker image는 upstream의 실제 GB10/SM121 지원 방식을 우선 사용한다.

확인할 것:

CUDA version
vLLM version
PyTorch version
FlashInfer version
Triton version
DFlash2 support
SM121 compatibility

가능하면 upstream image 구조를 활용한다.

하지만 4-node 전용 dependency나 불필요한 service는 제거한다.

---

20. SM121 native rebuild

현재 SM121 native cubin이 일부 없다는 이유만으로 다음 작업을 하지 않는다.

full CUDA rebuild
FlashInfer full rebuild
_C.abi3.so custom replacement
_moe_C.abi3.so custom replacement
new Marlin kernel
new sparse MLA kernel

먼저 실제 runtime profiler를 통해 병목을 확인한다.

native SM121 build는 실제 필요성이 확인될 때 별도 작업으로 진행한다.

---

21. Profiler

기본 production에서는 profiler를 비활성화한다.

필요하면 다음과 같은 debug configuration을 지원한다.

profiler=torch
torch_profiler_dir=/cache/step4

프로파일 결과는 production model cache와 분리한다.

---

22. up.sh

다음 명령으로 전체 cluster를 시작할 수 있어야 한다.

./scripts/up.sh

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
10. /health
11. /v1/models
12. inference self-test

실패한 단계가 명확하게 표시되어야 한다.

---

23. down.sh

./scripts/down.sh

로 모든 관련 container/process를 안전하게 종료할 수 있어야 한다.

Hugging Face cache는 삭제하지 않는다.

---

24. status.sh

./scripts/status.sh

실행 시 다음 정보를 보여준다.

HEAD
WORKER
GPU/rank
TP
NNODES
MODEL
DRAFT MODEL
vLLM version
CUDA version
FlashInfer version
DFlash2
DFlash tokens
DFlash selector top-k
DFlash walk mode
MoE backend
KV cache
MAX_MODEL_LEN
MAX_NUM_SEQS
MAX_NUM_BATCHED_TOKENS
NCCL interface
RoCE GID
health
OpenAI API

---

25. self-test.sh

./scripts/self-test.sh

다음 순서:

/health
/v1/models
simple completion
reasoning request
tool-call parser
DFlash2 warmup
second completion

첫 번째 요청은 Triton/JIT warmup일 수 있으므로 benchmark에서 제외한다.

---

26. Cold Start

첫 실행에서 Triton/JIT compile 때문에 오래 걸릴 수 있다.

따라서:

cold startup

과:

warm inference

를 구분한다.

README에 이를 명확히 설명한다.

---

27. Benchmark

최소 다음 concurrency를 지원한다.

C1
C2
C4
C6

측정:

TTFT
Prefill tok/s
Decode tok/s
Output tok/s
DFlash acceptance
GPU utilization
Memory

첫 warmup 요청은 benchmark에서 제외한다.

결과는:

docs/benchmark.md

에 기록한다.

실제 측정하지 않은 숫자를 README에 기재하지 않는다.

---

28. README

README는 다음 구조로 작성한다.

1. Project overview
2. Hardware
3. Architecture
4. Model
5. Hugging Face cache
6. Network
7. Requirements
8. Build
9. Configuration
10. Start
11. Health check
12. First inference
13. Benchmark
14. Troubleshooting
15. Known limitations
16. Upstream credits

특히 다음을 명시한다.

This repository is specifically designed for
2× NVIDIA DGX Spark / GB10 / SM121.

It is not a mechanically reduced 4-node recipe.
The distributed topology, memory configuration,
launcher and runtime configuration are designed
specifically for two DGX Spark nodes.

---

29. Hugging Face 정책을 README에 명시

다음 정책을 명확하게 기록한다.

Model:
canada-quant/glm-5.3-w4a16-mtp

Revision:
not pinned

Cache:
standard Hugging Face cache

Default host cache:
~/.cache/huggingface

Container cache:
/root/.cache/huggingface

모델 revision hash를 README에 기록하지 않는다.

---

30. License / Credits

upstream에서 코드를 가져오는 경우 파일별 source를 확인한다.

특히:

mmastrac
MiaAI-Lab
tonyd2wild
incoai
canada-quant
vLLM
FlashInfer

관련 코드를 사용할 경우 해당 프로젝트의 LICENSE와 attribution 조건을 확인한다.

원본 copyright/license header가 필요한 파일은 유지한다.

---

31. 금지사항

사용자 승인 없이 다음을 하지 않는다.

새 CUDA kernel 작성
CUDA 전체 rebuild
NCCL rebuild
FlashInfer 전체 rebuild
모델 weight 수정
모델 quantization 변경
DFlash2 weight 변경
TP4 설정 추가
Ray 도입
SGLang 도입
새 inference server 작성
revision pinning
/var/tmp model cache
Docker image 내부 model weight 포함

---

32. 코드 품질

스크립트는 다음 원칙을 따른다.

set -euo pipefail

환경변수는 기본값과 validation을 명확하게 한다.

잘못된 IP/interface/model 설정은 startup 전에 실패하도록 한다.

secret은 Git에 commit하지 않는다.

".env"는 ".gitignore"에 포함한다.

".env.example"만 Git에 포함한다.

---

33. Validation

실제 DGX Spark에서 실행하기 전에 다음을 수행한다.

docker compose config

그리고:

YAML validation
Dockerfile validation
shellcheck
environment validation
patch verification

을 수행한다.

---

34. 검색을 통한 upstream 검증

작업 전에 upstream repository의 현재 파일을 직접 확인한다.

특히:

compose/
image/
scripts/
patches/
Dockerfile
environment configuration
DFlash2 configuration
memory configuration

을 실제 코드 기준으로 분석한다.

README만 보고 구현하지 않는다.

---

35. 작업 단계

Cursor는 반드시 다음 순서로 작업한다.

Phase 1 — Upstream 분석

먼저 upstream 전체 구조를 분석한다.

아직 코드를 작성하지 않는다.

다음 표를 만든다.

File
Purpose
Relevant to 2× Spark?
Copy / Rewrite / Drop
Reason

---

Phase 2 — 2-node Architecture

다음 topology로 재설계한다.

192.168.100.10
HEAD / rank 0
        │
        │ RoCE
        │
192.168.100.20
WORKER / rank 1

TP=2
NNODES=2

---

Phase 3 — Image

필요한 SM121/GB10 patch만 적용한다.

---

Phase 4 — Compose

2-node 전용 compose를 작성한다.

---

Phase 5 — Launcher

다음 구현:

up.sh
down.sh
status.sh
logs.sh
self-test.sh

---

Phase 6 — Configuration

".env.example" 작성.

최소:

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

Revision 관련 변수는 만들지 않는다.

---

Phase 7 — Documentation

다음 문서를 작성한다.

docs/architecture.md
docs/deployment.md
docs/troubleshooting.md
docs/benchmark.md

---

Phase 8 — Static Validation

docker compose config
shellcheck
YAML validation
Dockerfile validation
patch verification
environment validation

---

Phase 9 — Git

현재 신규 repository에 commit/push한다.

권장 commit:

feat: add 2-node DGX Spark base recipe
feat: add GB10 and SM121 runtime patches
feat: add DFlash2 TP2 configuration
feat: add 2-node launcher and health checks
docs: add deployment and troubleshooting guide

---

36. 최종 목표

최종적으로:

./scripts/up.sh

한 번으로 2× DGX Spark에서 GLM-5.3-Flash가 올라오도록 한다.

정상 상태:

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

모델은:

Hugging Face Hub
        ↓
~/.cache/huggingface
        ↓
/root/.cache/huggingface
        ↓
vLLM

구조로 사용한다.

---

37. 최종 보고

작업 완료 후 다음 형식으로 보고한다.

## Repository

<repository name>

## Upstream

mmastrac/glm-5.3-flash-4x-gx10

## Architecture

2× DGX Spark
TP=2
NNODES=2

## Model

MODEL=
DRAFT_MODEL=

Revision:
NOT PINNED

HF Cache:
~/.cache/huggingface

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

실제 DGX Spark에서 추가로 검증해야 하는 항목만 작성한다.

실제 실행하지 않은 항목은 반드시 "NOT RUN"으로 표시한다.

추측해서 PASS라고 보고하지 않는다.
