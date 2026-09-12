#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
set -a; source "$ROOT/.env"; set +a
BASE="http://${HEAD_HOST}:${API_PORT:-8000}"
SERVED="${SERVED_MODEL_NAME:-glm-5.3-flash}"
curl -sf -m 10 "$BASE/health" >/dev/null
curl -sf -m 10 "$BASE/v1/models" >/dev/null
curl -sf -m 300 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$SERVED\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}],\"max_tokens\":8,\"chat_template_kwargs\":{\"enable_thinking\":false}}" >/dev/null
out=$(curl -sf -m 180 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$SERVED\",\"messages\":[{\"role\":\"user\",\"content\":\"Say OK\"}],\"max_tokens\":16,\"chat_template_kwargs\":{\"enable_thinking\":false}}")
python3 -c 'import sys,json; print(json.load(sys.stdin)["choices"][0]["message"].get("content"))' <<<"$out"
echo PASS
