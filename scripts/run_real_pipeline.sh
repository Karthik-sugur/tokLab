#!/usr/bin/env bash
# Real-data TokLab pipeline. Stops at the first failing gate.
#
#   bash scripts/run_real_pipeline.sh
#   SKIP_SYSTEMS=1 bash scripts/run_real_pipeline.sh   # skip the systems step
#
# Steps (each logged under logs/realrun_<step>.log):
#   1. data build       -> data/processed/manifest.parquet (prints manifest_hash)
#   2. synth + validate -> data/synthetic (all splits) + validation gate
#   3. experiment       -> runs/<config_hash>/ (reads data/synthetic, K* on val)
#   4. drift            -> drift sweep at K* (pre-generated drift/p<P> files)
#   5. systems          -> latency at K* (skippable: SKIP_SYSTEMS=1)
#   6. report           -> runs/<config_hash>/report.md
set -euo pipefail

cd "$(dirname "$0")/.."

REPOS_DIR="data/repos"
PROCESSED_DIR="data/processed"
SYNTH_DIR="data/synthetic"
EXPERIMENT_CONFIG="configs/experiments/mvp_py.yaml"
RUNS_ROOT="runs"
SEED=0
LOG_DIR="logs"
SKIP_SYSTEMS="${SKIP_SYSTEMS:-0}"

mkdir -p "$LOG_DIR"

PIPELINE_START=$SECONDS
STEP_START=$SECONDS
STEP_NAME=""
DURATIONS=()

fail() {
  echo "GATE FAILED: $*" >&2
  exit 1
}

step() {
  STEP_NAME="$1"
  STEP_START=$SECONDS
  echo
  echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] $STEP_NAME ==="
}

step_done() {
  local secs=$((SECONDS - STEP_START))
  DURATIONS+=("$STEP_NAME: ${secs}s")
  echo "--- $STEP_NAME took ${secs}s"
}

step "1/6 data build"
toklab data build --repos "$REPOS_DIR" --out "$PROCESSED_DIR" --seed "$SEED" \
  2>&1 | tee "$LOG_DIR/realrun_1_data_build.log"
[[ -f "$PROCESSED_DIR/manifest.parquet" ]] \
  || fail "manifest not written: $PROCESSED_DIR/manifest.parquet"
MANIFEST_HASH="$(tr -d '[:space:]' < "$PROCESSED_DIR/manifest_hash")"
[[ -n "$MANIFEST_HASH" ]] || fail "empty manifest_hash"
echo "manifest_hash=$MANIFEST_HASH"
step_done

step "2/6 synth + validation"
rm -rf "$SYNTH_DIR"
toklab data synth --manifest "$PROCESSED_DIR/manifest.parquet" \
  --repos "$REPOS_DIR" --out "$SYNTH_DIR" --seed "$SEED" \
  2>&1 | tee "$LOG_DIR/realrun_2_synth.log"
python scripts/validate_synth.py --repos "$REPOS_DIR" --synthetic "$SYNTH_DIR" \
  2>&1 | tee "$LOG_DIR/realrun_2_synth_validate.log" \
  || fail "synthetic layer validation failed (see $LOG_DIR/realrun_2_synth_validate.log)"
SYNTH_MANIFEST_HASH="$(sed -n 's/.*manifest_hash=\([0-9a-f]*\).*/\1/p' "$LOG_DIR/realrun_2_synth.log" | tail -n 1)"
[[ "$SYNTH_MANIFEST_HASH" == "$MANIFEST_HASH" ]] \
  || fail "synth built from manifest_hash=$SYNTH_MANIFEST_HASH, expected $MANIFEST_HASH"
step_done

step "3/6 experiment"
toklab run experiment --config "$EXPERIMENT_CONFIG" \
  2>&1 | tee "$LOG_DIR/realrun_3_experiment.log"
RUN_HASH="$(sed -n 's/^config_hash=//p' "$LOG_DIR/realrun_3_experiment.log" | tail -n 1)"
RUN_STATE="$(sed -n 's/^state=//p' "$LOG_DIR/realrun_3_experiment.log" | tail -n 1)"
K_STAR="$(sed -n 's/^k_star=//p' "$LOG_DIR/realrun_3_experiment.log" | tail -n 1)"
[[ -n "$RUN_HASH" ]] || fail "experiment printed no config_hash"
[[ "$RUN_STATE" == "complete" ]] || fail "experiment state=$RUN_STATE (expected complete)"
[[ -d "$RUNS_ROOT/$RUN_HASH" ]] || fail "run directory missing: $RUNS_ROOT/$RUN_HASH"
[[ -f "$RUNS_ROOT/$RUN_HASH/k_selection.json" ]] || fail "k_selection.json missing"
echo "k_star=$K_STAR"
step_done

step "4/6 drift (K*=$K_STAR)"
toklab run drift --run "$RUN_HASH" --runs-root "$RUNS_ROOT" \
  2>&1 | tee "$LOG_DIR/realrun_4_drift.log"
[[ -f "$RUNS_ROOT/$RUN_HASH/drift_table.txt" ]] \
  || fail "drift_table.txt not written for run $RUN_HASH"
step_done

if [[ "$SKIP_SYSTEMS" == "1" ]]; then
  step "5/6 systems (skipped: SKIP_SYSTEMS=1)"
  echo "systems step skipped; report will say 'Systems stage not run'"
  step_done
else
  step "5/6 systems (K*=$K_STAR)"
  toklab run systems --run "$RUN_HASH" --runs-root "$RUNS_ROOT" \
    2>&1 | tee "$LOG_DIR/realrun_5_systems.log"
  [[ -f "$RUNS_ROOT/$RUN_HASH/systems.parquet" ]] \
    || fail "systems.parquet not written for run $RUN_HASH"
  step_done
fi

step "6/6 report"
toklab report --run "$RUN_HASH" --runs-root "$RUNS_ROOT" \
  2>&1 | tee "$LOG_DIR/realrun_6_report.log"
REPORT_PATH="$(sed -n 's/^report=//p' "$LOG_DIR/realrun_6_report.log" | tail -n 1)"
[[ -n "$REPORT_PATH" && -f "$REPORT_PATH" ]] || fail "report not written"
step_done

echo
echo "=== step durations ==="
for d in "${DURATIONS[@]}"; do
  echo "  $d"
done
echo "  total: $((SECONDS - PIPELINE_START))s"
echo
echo "manifest_hash=$MANIFEST_HASH"
echo "k_star=$K_STAR"
echo "report=$REPORT_PATH"
echo "run_hash=$RUN_HASH"
