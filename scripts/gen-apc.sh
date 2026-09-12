#!/usr/bin/env bash
# Generate hybrid APC overlay from IMAGE + docs/patch_hybrid_prefix_hit.py.
# Fail-closed: exits 1 if anchors drift (do not mount a half-patched file).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
[[ -f .env ]] && { set -a; source .env; set +a; }

IMAGE="${IMAGE:-glm53-spark:2x-sm121}"
APC_WORK="${APC_WORK_DIR:-$ROOT/runtime/apc}"
APC_SCRIPT="${PATCH_APC_SCRIPT:-$ROOT/docs/patch_hybrid_prefix_hit.py}"
mkdir -p "$APC_WORK"

echo "[apc] extract coordinator from $IMAGE"
cid="$(docker create "$IMAGE")"
docker cp "$cid:/usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_coordinator.py" \
  "$APC_WORK/coordinator.pristine.py"
docker rm "$cid" >/dev/null

cp "$APC_WORK/coordinator.pristine.py" "$APC_WORK/coordinator.patched.py"
# Comment: patch mutates file in place via GLM53_KV_COORDINATOR_PY
GLM53_KV_COORDINATOR_PY="$APC_WORK/coordinator.patched.py" python3 "$APC_SCRIPT"

if ! grep -q "glm53-hybrid-apc" "$APC_WORK/coordinator.patched.py"; then
  echo "[apc] ERROR: patch markers missing — anchors likely drifted" >&2
  exit 1
fi
echo "[apc] ok → $APC_WORK/coordinator.patched.py"
