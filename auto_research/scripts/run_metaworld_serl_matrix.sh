#!/usr/bin/env bash
set -euo pipefail

SERL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RSYNC_ROOT="${RSYNC_ROOT:?Set RSYNC_ROOT to the Rsync repository}"
PYTHON_BIN="${PYTHON_BIN:-${SERL_ROOT}/auto_research/venv_serl/bin/python}"
MAX_UPDATES="${MAX_UPDATES:-3000}"
TIME_LIMIT_MIN="${TIME_LIMIT_MIN:-30}"
BATCH_SIZE="${BATCH_SIZE:-256}"
UTD_RATIO="${UTD_RATIO:-4}"
SEEDS="${SEEDS:-0}"
REWARD_MODES="${REWARD_MODES:-auto dense sparse}"

TASKS=(
  mw_button_press
  mw_window_open
  mw_reach_wall
  mw_plate_slide
  mw_push
  mw_coffee_push
  mw_stick_push
  mw_pick_place
)
TAUS=(fixed adaptive)

OUTPUT_ROOT="${SERL_ROOT}/auto_research/logs/metaworld_serl"
mkdir -p "${OUTPUT_ROOT}"

for seed in ${SEEDS}; do
  for task in "${TASKS[@]}"; do
    for reward in ${REWARD_MODES}; do
      data="${RSYNC_ROOT}/data/${task}/serl_${reward}.pkl"
      if [[ ! -f "${data}" ]]; then
        echo "Missing ${data}; run Rsync/scripts/run_metaworld_rm_pipeline.sh first." >&2
        exit 2
      fi
      for tau in "${TAUS[@]}"; do
        run="${task}__${reward}__${tau}__seed${seed}"
        adaptive_arg=""
        if [[ "${tau}" == "adaptive" ]]; then
          adaptive_arg="--adaptive_tau"
        fi
        echo "[RUN] ${run}"
        "${PYTHON_BIN}" "${SERL_ROOT}/auto_research/scripts/train_serl_offline_maniskill.py" \
          --data "${data}" \
          --max_updates "${MAX_UPDATES}" \
          --time_limit_min "${TIME_LIMIT_MIN}" \
          --seed "${seed}" \
          --batch_size "${BATCH_SIZE}" \
          --utd_ratio "${UTD_RATIO}" \
          --metrics_csv "${OUTPUT_ROOT}/${run}.csv" \
          ${adaptive_arg:+${adaptive_arg}} \
          2>&1 | tee "${OUTPUT_ROOT}/${run}.log"
      done
    done
  done
done
