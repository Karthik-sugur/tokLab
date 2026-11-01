#!/usr/bin/env bash
# Package a stored run so a second laptop (with the repo installed) can show
# the UI without re-running anything.
#
#   bash scripts/export_demo_bundle.sh            # run from configs/demo.yaml
#   TOKLAB_DEFAULT_RUN=<hash> bash scripts/export_demo_bundle.sh
#   MAX_FILE_MB=10 bash scripts/export_demo_bundle.sh
#
# Writes demo_bundle.zip at the repo root with repo-relative paths:
#   runs/<hash>/...                        report.md, tables, drift_table.txt,
#                                          K* vocab (for Preview); files above
#                                          MAX_FILE_MB are skipped and listed
#   data/synthetic/synth_manifest.parquet  if not above MAX_FILE_MB
#   <preview_file from configs/demo.yaml>  the pre-filled Preview snippet
#   demo_assets/report.html                if it exists
#   BUNDLE_MANIFEST.txt                    what was included / skipped
# Import: unzip -o demo_bundle.zip -d <repo root>   (see README "Demo").
set -euo pipefail

cd "$(dirname "$0")/.."

RUNS_ROOT="${RUNS_ROOT:-runs}"
OUT="${OUT:-demo_bundle.zip}"
MAX_FILE_MB="${MAX_FILE_MB:-5}"
MAX_BYTES=$((MAX_FILE_MB * 1024 * 1024))

die() {
  echo "ERROR: $*" >&2
  exit 1
}

if [[ -z "${VIRTUAL_ENV:-}" && -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

RUN_HASH="${TOKLAB_DEFAULT_RUN:-}"
if [[ -z "$RUN_HASH" ]]; then
  RUN_HASH="$(python -c "import yaml; print((yaml.safe_load(open('configs/demo.yaml')) or {}).get('demo_run') or '')")"
fi
[[ -n "$RUN_HASH" ]] || die "no run: set TOKLAB_DEFAULT_RUN or demo_run in configs/demo.yaml"
RUN_DIR="$RUNS_ROOT/$RUN_HASH"
[[ -d "$RUN_DIR" ]] || die "run directory not found: $RUN_DIR"
[[ -f "$RUN_DIR/report.md" ]] || die "no report.md in $RUN_DIR (run: toklab report --run $RUN_HASH)"

K_STAR="$(python -c "
import json, pathlib
p = pathlib.Path('$RUN_DIR/k_selection.json')
print(json.loads(p.read_text()).get('k_star') if p.is_file() else '')
")"
PREVIEW_FILE="$(python -c "import yaml; print((yaml.safe_load(open('configs/demo.yaml')) or {}).get('preview_file') or '')")"

size_of() { wc -c < "$1" | tr -d ' '; }

INCLUDED=()
SKIPPED=()
consider() {
  local f="$1"
  local n
  n="$(size_of "$f")"
  if (( n > MAX_BYTES )); then
    SKIPPED+=("$f ($n bytes > ${MAX_FILE_MB} MB)")
  else
    INCLUDED+=("$f")
  fi
}

while IFS= read -r f; do
  case "$f" in
    "$RUN_DIR"/vocab/*)
      if [[ -n "$K_STAR" && "$K_STAR" != "None" && "$(basename "$f")" == *"_k${K_STAR}.json" ]]; then
        consider "$f"
      else
        SKIPPED+=("$f (vocab not at K*=${K_STAR:-none})")
      fi
      ;;
    *) consider "$f" ;;
  esac
done < <(find "$RUN_DIR" -type f ! -name '.DS_Store' | sort)

for extra in data/synthetic/synth_manifest.parquet "$PREVIEW_FILE" demo_assets/report.html; do
  [[ -n "$extra" ]] || continue
  if [[ -f "$extra" ]]; then
    consider "$extra"
  else
    SKIPPED+=("$extra (not found)")
  fi
done

{
  echo "TokLab demo bundle"
  echo "run_hash=$RUN_HASH"
  echo "k_star=${K_STAR:-none}"
  echo "git_sha=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "created=$(date '+%Y-%m-%d %H:%M:%S %z')"
  echo
  echo "included:"
  for f in "${INCLUDED[@]}"; do echo "  $f"; done
  echo
  echo "skipped:"
  if (( ${#SKIPPED[@]} == 0 )); then echo "  (none)"; fi
  for f in "${SKIPPED[@]+"${SKIPPED[@]}"}"; do echo "  $f"; done
  echo
  echo "import: unzip -o $OUT -d <repo root>; then bash scripts/demo.sh"
  [[ "$RUN_HASH" == "$(python -c "import yaml; print((yaml.safe_load(open('configs/demo.yaml')) or {}).get('demo_run') or '')")" ]] \
    || echo "note: run differs from configs/demo.yaml; start with TOKLAB_DEFAULT_RUN=$RUN_HASH bash scripts/demo.sh"
} > BUNDLE_MANIFEST.txt

rm -f "$OUT"
zip -q -X "$OUT" BUNDLE_MANIFEST.txt "${INCLUDED[@]}"
rm -f BUNDLE_MANIFEST.txt
echo "bundle=$OUT ($(size_of "$OUT") bytes, ${#INCLUDED[@]} files, ${#SKIPPED[@]} skipped)"
