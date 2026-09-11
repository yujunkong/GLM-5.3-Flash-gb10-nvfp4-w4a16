#!/usr/bin/env bash
# Self-test against head OpenAI endpoint.
# Order: /health|/v1/models → simple completion → (optional) more probes.
# First request may be Triton/JIT warmup — not for benchmarks.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if [[ -f .env ]]; then
  # shellcheck disable=SC1091
  set -a && source .env && set +a
fi

HEAD_HOST="${HEAD_HOST:?Set HEAD_HOST}"
API_PORT="${API_PORT:-8002}"
BASE="http://${HEAD_HOST}:${API_PORT}"
SERVED="${SERVED_NAME:-glm53}"

echo "[self-test] ${BASE}"

code="$(curl -s -o /dev/null -w '%{http_code}' -m 10 "${BASE}/v1/models")"
[[ "$code" == "200" ]] || { echo "FAIL /v1/models HTTP ${code}" >&2; exit 1; }
echo "  /v1/models OK"

# Warmup (may be slow — JIT)
echo "  warmup completion..."
curl -sf -m 600 "${BASE}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"${SERVED}\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}],\"max_tokens\":8}" \
  >/tmp/glm53-selftest-warmup.json \
  || { echo "FAIL warmup completion" >&2; exit 1; }
echo "  warmup OK (exclude from benchmark)"

echo "  second completion..."
curl -sf -m 300 "${BASE}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"${SERVED}\",\"messages\":[{\"role\":\"user\",\"content\":\"Say OK\"}],\"max_tokens\":16}" \
  >/tmp/glm53-selftest-2.json \
  || { echo "FAIL second completion" >&2; exit 1; }
echo "  second completion OK"
echo "[self-test] PASS"
