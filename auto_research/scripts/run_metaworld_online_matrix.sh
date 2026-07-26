#!/usr/bin/env bash
set -euo pipefail

SERL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RSYNC_ROOT="${RSYNC_ROOT:?Set RSYNC_ROOT to the Rsync repository}"
PYTHON_BIN="${PYTHON_BIN:-${SERL_ROOT}/auto_research/venv_serl/bin/python}"
MAX_STEPS="${MAX_STEPS:-1000000}"
TIME_LIMIT_MIN="${TIME_LIMIT_MIN:-0}"
SEEDS="${SEEDS:-0 1 2}"

TASKS=(
  button-press
  window-open
  reach-wall
  plate-slide
  push
  coffee-push
  stick-push
  pick-place
)

rsync_name() {
  case "$1" in
    button-press) echo mw_button_press ;;
    window-open) echo mw_window_open ;;
    reach-wall) echo mw_reach_wall ;;
    plate-slide) echo mw_plate_slide ;;
    push) echo mw_push ;;
    coffee-push) echo mw_coffee_push ;;
    stick-push) echo mw_stick_push ;;
    pick-place) echo mw_pick_place ;;
  esac
}

for seed in ${SEEDS}; do
  for task in "${TASKS[@]}"; do
    task_data="$(rsync_name "${task}")"
    for reward in dense sparse; do
      demo="${RSYNC_ROOT}/data/${task_data}/serl_${reward}.pkl"
      if [[ ! -f "${demo}" ]]; then
        echo "Missing ${demo}; run the MetaWorld RM/data pipeline first." >&2
        exit 2
      fi
      for tau in fixed adaptive; do
        output="${SERL_ROOT}/auto_research/logs/metaworld_online/${task}__${reward}__${tau}__seed${seed}"
        mkdir -p "$(dirname "${output}")"
        adaptive_arg=""
        if [[ "${tau}" == "adaptive" ]]; then
          adaptive_arg="--adaptive-tau"
        fi
        echo "[RUN] ${task} reward=${reward} tau=${tau} seed=${seed}"
        "${PYTHON_BIN}" "${SERL_ROOT}/auto_research/scripts/train_serl_metaworld.py" \
          --task "${task}" \
          --reward-mode "${reward}" \
          --demo-path "${demo}" \
          --seed "${seed}" \
          --max-steps "${MAX_STEPS}" \
          --time-limit-min "${TIME_LIMIT_MIN}" \
          --output-dir "${output}" \
          ${adaptive_arg:+${adaptive_arg}} \
          2>&1 | tee "${output}.log"
      done
    done
  done
done
