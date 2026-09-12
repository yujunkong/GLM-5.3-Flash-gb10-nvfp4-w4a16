# Cursor 구현 지시문 — W4A16 accept ceiling 분석/디버그 (Golden 유지)

이 문서를 Cursor에 **그대로** 전달해 구현한다.  
대상 레포: `/home/yujunkong/workspace/docker/GLM-5.3-Flash-gb10-nvfp4-w4a16`  
(참고만: 구 검증 레포 `glm-5.3-flash-w4a16-2x-DGX-Sparks` — launcher/compose 복사 금지)

## 0. 목적 / 비목적

**목적**

- W4A16 + `incoai/GLM-5.3-Flash-DFlash2` 의 soak `/metrics` accept 천장(~0.43–0.44) **원인 분석**용 관측을 완성한다.
- 다음 단계(accept ≥ 0.50: draft/target 정렬 등)를 위한 **디버그 모드**를 준비한다.

**비목적**

- 서빙 TPS / Golden 성능 튜닝 아님.
- `TOP_K` / `WALK` / `TOKENS` / graphs / autotune / unary 등 **노브 실험·변경 금지**.
- NVFP4 타깃으로 전환하거나 이미지를 갈아엎는 작업 아님.

## 1. 하드 제약 (위반 시 실패)

1. **Golden 서빙 동작·TPS에 영향 없어야 함.**  
   `DFLASH2_ACC_PROBE=0`(기본)이면 probe 설치·stash·`.cpu()`·파일 I/O·CSV·추가 Prometheus gauge **전부 스킵**. 디코드 경로가 현재와 **완전히 동일**.
2. 디버그 기능은 **환경변수로만** ON/OFF. 코드 상수로 켜지 말 것.
3. **추가** A/B/C 관측용 `/metrics` gauge 및 **CSV 로그**는 디버그 모드(`DFLASH2_ACC_PROBE=1`)에서만 활성화.  
   (기존 vLLM `vllm:spec_decode_*` 카운터는 엔진 기본값 — 끄지 말 것.)
4. **변경 금지**
   - `compose/glm53.yaml`, `compose/overrides/*`, `compose/mentatd*.yaml`
   - `reference-notes/start.sh` 및 구 레포 `start.sh` 스타일 런처 재도입
   - Hugging Face 캐시 경로 / `HOST_CACHE` / `~/.cache/huggingface` 레이아웃
5. **Golden 노브 값 수정 금지** (기본값·문서 예시·`.env.example` 권장값 포함)

   | 변수 | Golden |
   |---|---|
   | `DFLASH_SELECTOR_TOP_K` | `32` |
   | `DFLASH_WALK_MODE` | `edge` |
   | `DFLASH_TOKENS` | `7` |
   | `DFLASH2_ACC_PROBE` | `0` (운영 기본) |
   | `ENFORCE_EAGER` | `1` |
   | `CUDA_GRAPHS` | `0` |
   | `DISABLE_FLASHINFER_AUTOTUNE` | `1` |
   | `MOE_BACKEND` | `marlin` |
   | `KV_CACHE_DTYPE` | `fp8_e4m3` |
   | `SPEC_METHOD` | `dflash` |
   | `MODEL` / `DRAFT_MODEL` | `canada-quant/...` / `incoai/...` |

6. compose에 **새 environment 키를 추가하지 말 것.**  
   이미 주입되는 `DFLASH2_ACC_PROBE` 하나만 마스터 스위치로 쓴다.  
   (선택 서브옵션이 필요하면 `DFLASH2_ACC_PROBE=1`일 때만 읽는 **선택 env**는 문서화하되, compose 미등록 → `docker compose run -e` / 일시 `.env` 주입으로만 쓰는 절차를 분석 문서에 적는다. 운영 `.env` 기본은 건드리지 않음.)

## 2. 현재 상태 (구현 전 사실)

- `patches/dflash2_speculator.py` 에 A/B/C probe 골격 있음 (`DFLASH2_ACC_PROBE`).
- 상태 파일 경로가 `/cache/dflash2-acc-state.json` + `/tmp/...` 인데, **이 레포 compose는 `/cache`를 마운트하지 않음** (`LOG_DIR` → `/logs`, `HOST_CACHE` → `/root/.cache`만).
- 그 결과 `bench/bench_reject_split.py` 가 호스트에서 probe를 못 읽어 A/B/C가 **전부 0**으로 나옴 (baseline `benchmarks/upstream-baseline-clocks2400/reject_split.json`).
- 운영은 `DFLASH2_ACC_PROBE=0` 유지가 맞음 (구 레포: probe ON soak ≈32.1 vs OFF ≈35.5).

## 3. 구현 범위 — 파일과 수정 위치

### 3.1 `patches/dflash2_speculator.py` (핵심)

| 위치 | 작업 |
|---|---|
| `_PROBE_PATHS` (~L28–31) | **운영 마운트에 맞춤.** 우선순위: `/logs/dflash2-acc-state.json`, 그다음 `/tmp/dflash2-acc-state.json`. `/cache/...` 는 호환용으로 남겨도 되나 **필수 경로가 아니게**. |
| `_probe_enabled()` | 기존 유지: env `=="1"` 이고 rank0만. |
| `generate` stash 블록 (~L393–402) | `if _probe_enabled():` 가드 유지. `PROBE=0`이면 candidate/unary/edge **참조 보관 금지**. |
| `_install_rejection_probe` | `PROBE=0`이면 `__init__`에서 호출하지 않음 (기존). monkey-patch는 probe ON 부팅에서만. |
| `_observe_rejection` | A/B/C·B-detail 분류 로직 유지. `.cpu()` 는 probe ON에서만. |
| `_flush_probe` | JSON atomic write를 `/logs` 우선으로. |
| **신규** `_flush_probe_csv` (또는 `_flush_probe` 내부) | `DFLASH2_ACC_PROBE=1`일 때만 `/logs/dflash2-acc.csv` append. 스키마 예: `ts,draft_rounds,accepted,drafted,rejected,A,B,C,b_unary_would_hit,b_unary_also_miss`. 헤더는 파일 없을 때 1회. |
| **신규** Prometheus (디버그 전용) | `PROBE=1`일 때만 A/B/C·rejected_rounds 등을 gauge/counter로 등록해 **기존** vLLM `/metrics`에 노출. 이름 예: `dflash2_acc_lm_head_topk_miss`, `dflash2_acc_walk_miss`, `dflash2_acc_verify_reject`. `PROBE=0`이면 import/register/update **금지**. 등록이 엔진 구조상 과도하면: (1) JSON+CSV를 1차 산출로 하고 (2) `docs/accept-ceiling-debug.md`에 “Prometheus 미연결 사유”를 명시 — 단, 가능하면 `/metrics` 노출을 우선 시도. |

주석: `# Analysis only when DFLASH2_ACC_PROBE=1; Golden path unchanged at 0.`

### 3.2 `bench/bench_reject_split.py`

| 위치 | 작업 |
|---|---|
| `PROBE_CANDIDATES` (~L37–40) | 호스트 경로 추가·수정. **이 레포 기본**: `$LOG_DIR/dflash2-acc-state.json` (`LOG_DIR` 기본 `~/logs/glm53`), compose와 동일 마운트. 구 경로(`/var/tmp/...`, `/tmp/...`)는 fallback 유지. |
| docstring | probe ON 진단 부팅에서만 A/B/C가 non-zero라고 명시. |
| 출력 | `first_reject_split` 요약에 A/B/C 비율(%) 및 B2(`b_unary_*`) 힌트 유지/보강. |

compose·HF 캐시 경로는 **수정하지 않음**. docker cp / SSH로 `/logs` 파일을 읽는 헬퍼가 필요하면 bench 쪽에만 추가.

### 3.3 `bench/acceptance_ratio.py` (소폭)

- 디버그 gauge가 `/metrics`에 있으면 함께 파싱해 출력.
- 없으면 기존 `accepted/drafted`만 (회귀 없음).

### 3.4 신규 `bench/run_accept_ceiling_diag.sh`

진단 **전용** 스크립트 (운영 `up.sh` 기본 경로에 넣지 말 것).

1. 사전조건 출력: Golden 노브 변경 금지, **일시** `DFLASH2_ACC_PROBE=1` 재기동 필요.
2. (수동/안내) head+worker를 probe ON으로 올린 뒤:
   - `python3 bench/bench_reject_split.py ...`
   - `python3 bench/acceptance_ratio.py`
   - `/logs/dflash2-acc-state.json` · `dflash2-acc.csv` 존재 확인
3. 결과를 `benchmarks/accept-ceiling-diag-<ts>/` 에 저장.
4. 끝나면 **반드시** `DFLASH2_ACC_PROBE=0` 으로 되돌리라고 출력.
5. `scripts/up.sh` / compose 기본값을 probe ON으로 바꾸지 말 것.

### 3.5 신규 `docs/accept-ceiling-debug.md`

포함할 내용:

- 왜 W4A16 soak accept가 ~0.43인지 (draft–target 불일치, B≈88–91% 가설) — 측정 전제와 재현 절차.
- 디버그 ON/OFF 절차 (`DFLASH2_ACC_PROBE`, compose **변경 없이** `.env` 일시 수정 → `scripts/up.sh`).
- 산출물: JSON / CSV / (있으면) `/metrics` gauge 이름.
- Golden 복귀 체크리스트.
- 다음 단계 후보만 나열 (구현하지 말 것): W4A16용 draft 재학습, NVFP4 타깃 실험 레시피 분리 등.

### 3.6 `.env.example`

- `DFLASH2_ACC_PROBE=0` 유지.
- 주석 2–3줄: `=1` 은 분석 부팅 전용, soak −~10%, CSV·A/B/C는 `/logs`.

### 3.7 건드리면 안 되는 파일 (명시)

- `compose/**` (내용·환경키·볼륨·HF 경로)
- `image/entrypoint.sh` 의 Golden argv / TOP_K·WALK·TOKENS 기본 조립 (probe 관련 print 한 줄 추가는 허용하되 기본값 변경 금지)
- `scripts/up.sh` 의 기본 `DFLASH2_ACC_PROBE` 강제 ON
- `clocks.sh`, SM90/APC/kv-groups 패치 생성 스크립트
- `patches/glm5next_model.py`, `qwen3_dflash2.py`, kpool, warmup (accept 분석과 무관하면 수정 금지)

## 4. 구현 후 필수 제출물

구현이 끝나면 Cursor는 **한국어로** 아래를 제출한다.

### 4.1 변경 파일 목록

경로 + 한 줄 요지.

### 4.2 diff 요약

파일별 핵심 변경 (노브/compose/HF 경로를 안 건드렸는지 체크 포함).

### 4.3 오버헤드(성능 영향) 확인 결과

최소 검증:

1. **`DFLASH2_ACC_PROBE=0` (Golden)**  
   - 서버 기동 후 self-test PASS.  
   - soak 또는 동등 decode 벤치 1회 이상: tok/s 가 baseline `benchmarks/upstream-baseline-clocks2400` (soak median ≈35.87) 대비 **유의미한 하락 없음** (목표: 변동 ≤ ~3% 또는 측정 노이즈 내).  
   - `/logs/dflash2-acc.csv` **생성되지 않음**.  
   - probe JSON이 갱신되지 않거나, 쓰이더라도 decode 경로에 `.cpu()` stash가 없음(코드 가드 확인).  
   - `reject_split` A/B/C = 0 은 probe OFF에서 **정상**.

2. **`DFLASH2_ACC_PROBE=1` (진단 부팅, 짧게)**  
   - 동일 워크로드에서 A/B/C non-zero.  
   - CSV·JSON이 `/logs`에 생김.  
   - (구현 시) `/metrics`에 디버그 gauge 출현.  
   - soak가 구 레포와 같이 떨어질 수 있음 — **기대 동작**, 운영 기본으로 두지 말 것.  
   - 측정 후 **반드시 PROBE=0 복귀**.

3. 회귀 체크: Golden 노브 문자열/기본값이 디프에 안 들어갔는지 `rg`로 확인.

### 4.4 분석 해석 (짧게)

진단 부팅에서 나온 A/B/C 비율을 한 단락으로 해석하고, “서빙 노브로 0.50 불가 / draft–target 정렬 필요” 여부를 **데이터에 근거해** 적는다. 추측으로 완료 선언 금지.

## 5. 완료 기준 (Definition of Done)

- [ ] `PROBE=0` 경로 = 기존와 동일 (성능·동작).
- [ ] `PROBE=1` 에서만 JSON(`/logs`) + CSV + (가능 시) `/metrics` A/B/C.
- [ ] compose / start.sh / HF 캐시 경로 **무변경**.
- [ ] Golden 노브 **무변경**.
- [ ] `bench_reject_split` 이 이 레포 `LOG_DIR` 마운트에서 probe를 읽음.
- [ ] 제출물 4.1–4.4 완비.

## 6. 작업 순서 권장

1. `patches/dflash2_speculator.py` 경로·CSV·(옵션) metrics  
2. `bench/bench_reject_split.py` + `acceptance_ratio.py`  
3. `bench/run_accept_ceiling_diag.sh` + `docs/accept-ceiling-debug.md` + `.env.example` 주석  
4. PROBE=0 오버헤드 확인 → PROBE=1 스모크 → PROBE=0 복귀  
5. 제출물 작성

커밋은 사용자 요청 시에만. push 금지.
