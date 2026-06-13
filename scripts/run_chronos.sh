#!/usr/bin/env bash
# Run Chronos-2 foundation experiments for step 4 of the pipeline.
#
# The runner (src.chronos_runner) prints one `conda run -n stock-prediction ...`
# command per (stock × seed). This wrapper pipes those commands into bash so
# they actually execute, captures stdout+stderr to a timestamped log, and lets
# you scope the run via positional args:
#
#   scripts/run_chronos.sh
#       → all 4 modes × all stocks × all foundation_seeds (full step 4)
#
#   scripts/run_chronos.sh chronos2_finetuned_cov
#       → just one mode (zero_shot, zero_shot_cov, finetuned, finetuned_cov)
#
#   scripts/run_chronos.sh chronos2_finetuned_cov AAPL 42
#       → one mode, one stock, one seed (pretest)
#
# Designed to be safe under nohup:
#   nohup bash scripts/run_chronos.sh > /dev/null 2>&1 &
# The log file path is printed at the start and again at the end.

set -euo pipefail

# Repo-relative cwd so paths in the emitted commands resolve.
cd "$(dirname "$0")/.."

mkdir -p logs

ALL_MODES=(
  chronos2_zero_shot
  chronos2_zero_shot_cov
  chronos2_finetuned
  chronos2_finetuned_cov
)

# Positional args
MODE="${1:-}"
STOCK="${2:-}"
SEED="${3:-}"

if [[ -n "${MODE}" ]]; then
  MODES=("${MODE}")
else
  MODES=("${ALL_MODES[@]}")
fi

TS="$(date +%Y%m%d_%H%M%S)"
TAG="chronos"
[[ -n "${MODE}" ]]  && TAG="${TAG}_${MODE}"
[[ -n "${STOCK}" ]] && TAG="${TAG}_${STOCK}"
[[ -n "${SEED}" ]]  && TAG="${TAG}_s${SEED}"
LOG="logs/${TAG}_${TS}.log"

# Build the runner-args list once.
RUNNER_ARGS=()
[[ -n "${STOCK}" ]] && RUNNER_ARGS+=(--stock "${STOCK}")
[[ -n "${SEED}"  ]] && RUNNER_ARGS+=(--seeds "${SEED}")

{
  echo "=== run_chronos.sh start at $(date) ==="
  echo "Log:        ${LOG}"
  echo "Modes:      ${MODES[*]}"
  echo "Runner args: ${RUNNER_ARGS[*]:-(defaults)}"
  echo
} | tee -a "${LOG}"

for mode in "${MODES[@]}"; do
  echo "=== mode=${mode} start at $(date) ===" | tee -a "${LOG}"
  # The runner prints lines like "  python src/...py ...". Strip the leading
  # two-space indent and pipe straight into bash. `set -e` inside this bash
  # subshell makes any individual run failure stop the rest of THIS mode.
  if ! python -m src.chronos_runner --mode "${mode}" "${RUNNER_ARGS[@]}" \
       | grep '^  ' | sed 's/^  //' \
       | bash -e - 2>&1 | tee -a "${LOG}"; then
    echo "=== mode=${mode} FAILED at $(date); aborting wrapper ===" | tee -a "${LOG}"
    exit 1
  fi
  echo "=== mode=${mode} done at $(date) ===" | tee -a "${LOG}"
done

{
  echo
  echo "=== run_chronos.sh done at $(date) ==="
  echo "Log: ${LOG}"
} | tee -a "${LOG}"
