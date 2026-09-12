#!/usr/bin/env bash
# Generate SM90 fp8 plan-dtype overlay from IMAGE.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
[[ -f .env ]] && { set -a; source .env; set +a; }

IMAGE="${IMAGE:-glm53-spark:2x-sm121}"
WORK="${SM90_WORK_DIR:-$ROOT/runtime/sm90}"
SCRIPT="${PATCH_SM90_SCRIPT:-$ROOT/docs/patch_sm90_fp8_plan_dtype.py}"
mkdir -p "$WORK"

echo "[sm90] extract flashinfer_mla_sparse_sm90 from $IMAGE"
cid="$(docker create "$IMAGE")"
docker cp "$cid:/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm90.py" \
  "$WORK/flashinfer_mla_sparse_sm90.pristine.py"
docker rm "$cid" >/dev/null

cp "$WORK/flashinfer_mla_sparse_sm90.pristine.py" "$WORK/flashinfer_mla_sparse_sm90.patched.py"
GLM53_SM90_MLA_PY="$WORK/flashinfer_mla_sparse_sm90.patched.py" python3 "$SCRIPT"

if ! grep -q "glm53-sm90-fp8-plan-dtype" "$WORK/flashinfer_mla_sparse_sm90.patched.py"; then
  echo "[sm90] ERROR: patch markers missing" >&2
  exit 1
fi
python3 -m py_compile "$WORK/flashinfer_mla_sparse_sm90.patched.py"
echo "[sm90] ok → $WORK/flashinfer_mla_sparse_sm90.patched.py"
