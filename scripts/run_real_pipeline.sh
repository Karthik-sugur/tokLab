#!/usr/bin/env bash
# Real-data TokLab pipeline. Stops at the first failing gate.
#
#   bash scripts/run_real_pipeline.sh
#
# Steps (each logged under logs/realrun_<step>.log):
#   1. data build      -> data/processed/manifest.parquet
#   2. synth + validate -> data/synthetic (all splits) + validation gate
#   3. experiment      -> runs/<config_hash>/
#   4. drift           -> drift sweep for that run
#   5. report          -> runs/<config_hash>/report.md
set -euo pipefail

cd "$(dirname "$0")/.."

REPOS_DIR="data/repos"
PROCESSED_DIR="data/processed"
SYNTH_DIR="data/synthetic"
EXPERIMENT_CONFIG="configs/experiments/mvp_py.yaml"
RUNS_ROOT="runs"
SEED=0
LOG_DIR="logs"

mkdir -p "$LOG_DIR"

fail() {
  echo "GATE FAILED: $*" >&2
  exit 1
}

step() {
  echo
  echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] $1 ==="
}

step "1/5 data build"
toklab data build --repos "$REPOS_DIR" --out "$PROCESSED_DIR" --seed "$SEED" \
  2>&1 | tee "$LOG_DIR/realrun_1_data_build.log"
[[ -f "$PROCESSED_DIR/manifest.parquet" ]] \
  || fail "manifest not written: $PROCESSED_DIR/manifest.parquet"

step "2/5 synth + validation"
rm -rf "$SYNTH_DIR"
toklab data synth --manifest "$PROCESSED_DIR/manifest.parquet" \
  --repos "$REPOS_DIR" --out "$SYNTH_DIR" --seed "$SEED" \
  2>&1 | tee "$LOG_DIR/realrun_2_synth.log"
python scripts/validate_synth.py --repos "$REPOS_DIR" --synthetic "$SYNTH_DIR" \
  2>&1 | tee "$LOG_DIR/realrun_2_synth_validate.log" \
  || fail "synthetic layer validation failed (see $LOG_DIR/realrun_2_synth_validate.log)"

step "3/5 experiment"
toklab run experiment --config "$EXPERIMENT_CONFIG" \
  2>&1 | tee "$LOG_DIR/realrun_3_experiment.log"
RUN_HASH="$(sed -n 's/^config_hash=//p' "$LOG_DIR/realrun_3_experiment.log" | tail -n 1)"
RUN_STATE="$(sed -n 's/^state=//p' "$LOG_DIR/realrun_3_experiment.log" | tail -n 1)"
[[ -n "$RUN_HASH" ]] || fail "experiment printed no config_hash"
[[ "$RUN_STATE" == "complete" ]] || fail "experiment state=$RUN_STATE (expected complete)"
[[ -d "$RUNS_ROOT/$RUN_HASH" ]] || fail "run directory missing: $RUNS_ROOT/$RUN_HASH"

step "4/5 drift"
toklab run drift --run "$RUN_HASH" --runs-root "$RUNS_ROOT" \
  2>&1 | tee "$LOG_DIR/realrun_4_drift.log"
[[ -f "$RUNS_ROOT/$RUN_HASH/drift_table.txt" ]] \
  || fail "drift_table.txt not written for run $RUN_HASH"

step "5/5 report"
toklab report --run "$RUN_HASH" --runs-root "$RUNS_ROOT" \
  2>&1 | tee "$LOG_DIR/realrun_5_report.log"
REPORT_PATH="$(sed -n 's/^report=//p' "$LOG_DIR/realrun_5_report.log" | tail -n 1)"
[[ -n "$REPORT_PATH" && -f "$REPORT_PATH" ]] || fail "report not written"

echo
echo "run_hash=$RUN_HASH"
echo "report=$REPORT_PATH"
