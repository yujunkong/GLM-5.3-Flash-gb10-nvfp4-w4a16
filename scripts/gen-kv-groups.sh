#!/usr/bin/env bash
# Generate DFlash+GLM5 KV-group overlay from IMAGE + docs/patch_dflash_glm5_kv_groups.py.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
[[ -f .env ]] && { set -a; source .env; set +a; }

IMAGE="${IMAGE:-glm53-spark:2x-sm121}"
WORK="${KV_GROUPS_WORK_DIR:-$ROOT/runtime/kv}"
SCRIPT="${PATCH_KV_GROUPS_SCRIPT:-$ROOT/docs/patch_dflash_glm5_kv_groups.py}"
mkdir -p "$WORK"

echo "[kv-groups] extract kv_cache_utils from $IMAGE"
cid="$(docker create "$IMAGE")"
docker cp "$cid:/usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_utils.py" \
  "$WORK/kv_cache_utils.pristine.py"
docker rm "$cid" >/dev/null

cp "$WORK/kv_cache_utils.pristine.py" "$WORK/kv_cache_utils.patched.py"
GLM53_KV_CACHE_UTILS_PY="$WORK/kv_cache_utils.patched.py" python3 "$SCRIPT"

if ! grep -q "glm53-dflash-kv-groups" "$WORK/kv_cache_utils.patched.py"; then
  echo "[kv-groups] ERROR: patch markers missing" >&2
  exit 1
fi
python3 -m py_compile "$WORK/kv_cache_utils.patched.py"
echo "[kv-groups] ok → $WORK/kv_cache_utils.patched.py"
