#!/usr/bin/env bash
# Run FinCast foundation experiments for step 4 of the pipeline.
#
# Mirrors scripts/run_chronos.sh: src.fincast_runner prints
# `conda run -n fincast_v1 ...` commands; this wrapper pipes them into bash,
# captures stdout+stderr to a timestamped log, and lets you scope the run via
# positional args:
#
#   scripts/run_fincast.sh
#       → both modes × all stocks × all foundation_seeds (full FinCast step 4)
#
#   scripts/run_fincast.sh fincast_finetuned
#       → just one mode (fincast_zero_shot or fincast_finetuned)
#
#   scripts/run_fincast.sh fincast_finetuned AAPL 42
#       → one mode, one stock, one seed (pretest)
#
# Env vars (override the defaults if your FinCast checkout lives elsewhere):
#   FINCAST_PATH      — path to <repo>/FinCast-fts/src, needed at import time
#   FINCAST_WEIGHTS   — path to <repo>/FinCast-fts/model_weights/v1.pth
#
# Designed to be safe under nohup:
#   nohup bash scripts/run_fincast.sh > /dev/null 2>&1 &

set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p logs

# Repo-relative defaults; override via env to point at a FinCast checkout elsewhere.
# After the `cd` above, $(pwd) is the repo root.
: "${FINCAST_PATH:=$(pwd)/FinCast-fts/src}"
: "${FINCAST_WEIGHTS:=$(pwd)/FinCast-fts/model_weights/v1.pth}"
export FINCAST_PATH FINCAST_WEIGHTS

if [[ ! -d "${FINCAST_PATH}" ]]; then
  echo "ERROR: FINCAST_PATH=${FINCAST_PATH} is not a directory." >&2
  echo "       Set FINCAST_PATH to <repo>/FinCast-fts/src and retry." >&2
  exit 2
fi
if [[ ! -f "${FINCAST_WEIGHTS}" ]]; then
  echo "ERROR: FINCAST_WEIGHTS=${FINCAST_WEIGHTS} is not a file." >&2
  echo "       Set FINCAST_WEIGHTS to <repo>/FinCast-fts/model_weights/v1.pth and retry." >&2
  exit 2
fi

ALL_MODES=(
  fincast_zero_shot
  fincast_finetuned
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
TAG="fincast"
[[ -n "${MODE}" ]]  && TAG="${TAG}_${MODE}"
[[ -n "${STOCK}" ]] && TAG="${TAG}_${STOCK}"
[[ -n "${SEED}" ]]  && TAG="${TAG}_s${SEED}"
LOG="logs/${TAG}_${TS}.log"

# Build the runner-args list once.
RUNNER_ARGS=()
[[ -n "${STOCK}" ]] && RUNNER_ARGS+=(--stock "${STOCK}")
[[ -n "${SEED}"  ]] && RUNNER_ARGS+=(--seeds "${SEED}")

{
  echo "=== run_fincast.sh start at $(date) ==="
  echo "Log:              ${LOG}"
  echo "Modes:            ${MODES[*]}"
  echo "Runner args:      ${RUNNER_ARGS[*]:-(defaults)}"
  echo "FINCAST_PATH:     ${FINCAST_PATH}"
  echo "FINCAST_WEIGHTS:  ${FINCAST_WEIGHTS}"
  echo
} | tee -a "${LOG}"

for mode in "${MODES[@]}"; do
  echo "=== mode=${mode} start at $(date) ===" | tee -a "${LOG}"
  if ! python -m src.fincast_runner --mode "${mode}" "${RUNNER_ARGS[@]}" \
       | grep '^  ' | sed 's/^  //' \
       | bash -e - 2>&1 | tee -a "${LOG}"; then
    echo "=== mode=${mode} FAILED at $(date); aborting wrapper ===" | tee -a "${LOG}"
    exit 1
  fi
  echo "=== mode=${mode} done at $(date) ===" | tee -a "${LOG}"
done

{
  echo
  echo "=== run_fincast.sh done at $(date) ==="
  echo "Log: ${LOG}"
} | tee -a "${LOG}"
