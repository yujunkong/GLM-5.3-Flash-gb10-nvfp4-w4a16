# W4A16 accept-ceiling debug (Golden 유지)

분석 전용. 서빙 노브(`TOP_K=32`, `WALK=edge`, `TOKENS=7`, `ENFORCE_EAGER=1` 등)는 바꾸지 않는다.

## 배경

- Soak `/metrics` accept ≈ **0.43–0.44** (`canada-quant` W4A16 + `incoai` DFlash2).
- 업스트림 NVFP4 문서의 0.6–0.8은 **타깃 정합이 다른 스택** — 이 레포 Golden과 동일시하지 말 것.
- First-reject는 주로 **B (pool hit, walk miss)**; B2가 크면 TOP_K/walk 노브로 0.50에 못 감.
- `DFLASH2_ACC_PROBE=1` 은 GPU→CPU sync로 soak ≈ **−10%** (구 레포 32.1 vs 35.5). **운영 기본은 0.**

## ON / OFF

| 모드 | env | 동작 |
|------|-----|------|
| Golden (운영) | `DFLASH2_ACC_PROBE=0` | stash / `.cpu()` / CSV / 디버그 gauge **없음** |
| 진단 부팅 | `DFLASH2_ACC_PROBE=1` | `/logs` JSON+CSV, A/B/C, (가능 시) `/metrics` `dflash2_acc_*` |

compose / HF 캐시 경로(`HOST_CACHE` → `~/.cache`)는 변경하지 않는다.  
이미 compose에 있는 `DFLASH2_ACC_PROBE` 만 사용한다.

```bash
# 진단 부팅
# .env: DFLASH2_ACC_PROBE=1
bash scripts/up.sh
bash bench/run_accept_ceiling_diag.sh

# 복귀 (필수)
# .env: DFLASH2_ACC_PROBE=0
bash scripts/up.sh
```

## Metrics (`PROBE=1`)

File: `$LOG_DIR/dflash2-acc.prom` (API `/metrics` cannot see worker gauges without multiproc).

| Name | Meaning |
|------|---------|
| `accept_ratio_recent` | last draft round accepted/drafted |
| `accept_ratio_window_128` | rolling 128 rounds |
| `accept_ratio_window_256` | rolling 256 rounds |
| `reject_stage_b0_total` | first-reject target ∉ lm_head top-k (alias A) |
| `reject_stage_b1_total` | in pool; unary would have hit |
| `reject_stage_b2_total` | in pool; unary also miss |

Also JSON/CSV under `$LOG_DIR`. See `docs/accept-ceiling-debug.md`.

## Golden 복귀 체크리스트

- [ ] `.env` 에 `DFLASH2_ACC_PROBE=0`
- [ ] `scripts/up.sh` 재기동
- [ ] `/health` + self-test PASS
- [ ] `$LOG_DIR/dflash2-acc.csv` 가 **새 decode에서 갱신되지 않음** (또는 파일 삭제 후 미생성)
- [ ] soak /metrics accept ~0.43 밴드, tok/s baseline 근처

## 다음 단계 (이 문서 범위 밖)

- W4A16용 draft 재학습 또는 NVFP4 타깃 실험 레시피 분리
- 서빙 노브 재튜닝으로 0.50을 기대하지 말 것
