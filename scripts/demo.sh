#!/usr/bin/env bash
# Start the TokLab demo: API in the background, Streamlit UI, open the browser.
#
#   bash scripts/demo.sh
#   TOKLAB_DEFAULT_RUN=<hash> bash scripts/demo.sh   # another stored run
#   API_PORT=8001 UI_PORT=8502 bash scripts/demo.sh
#   NO_OPEN=1 bash scripts/demo.sh                    # do not open a browser
#
# Display only: reads runs/<hash>/; never runs an experiment. Ctrl-C stops
# both servers.
set -euo pipefail

cd "$(dirname "$0")/.."

API_PORT="${API_PORT:-8000}"
UI_PORT="${UI_PORT:-8501}"
RUNS_ROOT="${RUNS_ROOT:-runs}"
API_URL="http://127.0.0.1:${API_PORT}"
UI_URL="http://localhost:${UI_PORT}"
API_WAIT_SECS=20

die() {
  echo "ERROR: $*" >&2
  exit 1
}

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  [[ -f .venv/bin/activate ]] \
    || die "no .venv found. Install first: python3.11 -m venv .venv && .venv/bin/pip install -e '.[dev]'"
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi
command -v toklab >/dev/null || die "toklab not installed in the venv (pip install -e '.[dev]')"

RUN_HASH="${TOKLAB_DEFAULT_RUN:-}"
if [[ -z "$RUN_HASH" ]]; then
  RUN_HASH="$(python -c "import yaml; print((yaml.safe_load(open('configs/demo.yaml')) or {}).get('demo_run') or '')")"
fi
[[ -n "$RUN_HASH" ]] || die "no demo run: set TOKLAB_DEFAULT_RUN or demo_run in configs/demo.yaml"
[[ -d "$RUNS_ROOT/$RUN_HASH" ]] \
  || die "run directory not found: $RUNS_ROOT/$RUN_HASH (run the pipeline or unzip demo_bundle.zip; see README)"
[[ -f "$RUNS_ROOT/$RUN_HASH/report.md" ]] || die "no report.md in $RUNS_ROOT/$RUN_HASH"
export TOKLAB_DEFAULT_RUN="$RUN_HASH"
export TOKLAB_API_URL="$API_URL"
echo "demo run: $RUN_HASH"

TOKENIZER="$(python -c "import json; print(json.load(open('$RUNS_ROOT/$RUN_HASH/config.json'))['model']['tokenizer_name'])")"
if HF_HUB_OFFLINE=1 python -c "
import sys
from transformers import AutoTokenizer
AutoTokenizer.from_pretrained(sys.argv[1], local_files_only=True)
" "$TOKENIZER" >/dev/null 2>&1; then
  export HF_HUB_OFFLINE=1
  echo "tokenizer $TOKENIZER cached: HF_HUB_OFFLINE=1"
else
  echo "WARNING: tokenizer not cached, Preview needs internet once ($TOKENIZER)" >&2
fi

if curl -s -o /dev/null "$API_URL/health"; then
  die "something is already listening on port $API_PORT (stop it or set API_PORT)"
fi

API_PID=""
UI_PID=""
cleanup() {
  [[ -n "$UI_PID" ]] && kill "$UI_PID" 2>/dev/null || true
  [[ -n "$API_PID" ]] && kill "$API_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

toklab serve-api --port "$API_PORT" --runs-root "$RUNS_ROOT" &
API_PID=$!

echo "waiting for $API_URL/datasets (max ${API_WAIT_SECS}s)..."
code=""
for _ in $(seq 1 "$API_WAIT_SECS"); do
  kill -0 "$API_PID" 2>/dev/null || die "API process exited during startup (see output above)"
  code="$(curl -s -o /dev/null -w '%{http_code}' "$API_URL/datasets" || true)"
  [[ "$code" == "200" ]] && break
  sleep 1
done
[[ "$code" == "200" ]] \
  || die "API did not answer GET /datasets with 200 within ${API_WAIT_SECS}s (last status: ${code:-none})"
echo "API ready: $API_URL"

python -m streamlit run ui/app.py \
  --server.headless true \
  --server.port "$UI_PORT" \
  --browser.gatherUsageStats false &
UI_PID=$!

for _ in $(seq 1 "$API_WAIT_SECS"); do
  kill -0 "$UI_PID" 2>/dev/null || die "Streamlit exited during startup (see output above)"
  curl -s -o /dev/null "$UI_URL/_stcore/health" && break
  sleep 1
done

if [[ "${NO_OPEN:-0}" != "1" ]] && command -v open >/dev/null; then
  open "$UI_URL"
else
  echo "open $UI_URL in a browser"
fi
echo "UI: $UI_URL   API: $API_URL   (Ctrl-C to stop)"
wait "$UI_PID"
