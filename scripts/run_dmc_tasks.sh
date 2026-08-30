#!/usr/bin/env bash

# Sequential DMC training runner.
# Order is seed-major: run all tasks for seed 3, then all tasks for seed 4, etc.

set -u
set -o pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${REPO_ROOT}/results/runner_logs"

SEED_GROUP="default"
SEEDS_CSV=""
TASKS_CSV=""
DEVICE=""
EXP_NAME="default"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RETRY_FAILED=1
RETRY_DELAY=60
EXTRA_ARGS=()

usage() {
    cat <<'EOF'
Usage:
  bash scripts/run_dmc_tasks.sh [options] [-- extra main.py args]

Options:
  --seed-group GROUP       Seed group: default|paper|debug (default: default)
  --seeds CSV             Override seeds, e.g. 3,4,5
  --tasks CSV             Override tasks, e.g. walker-run,cheetah-run,cup-catch
  --device DEVICE          Value passed to main.py --device, e.g. 0
  --exp-name NAME          Value passed to main.py --exp_name (default: dmc_classic)
  --log-dir DIR            Runner stdout/stderr log directory (default: results/runner_logs)
  --retry-delay SECONDS    Delay before retrying a failed run (default: 60)
  --no-retry               Continue to the next run when one run fails
  -h, --help               Show this help

Examples:
  bash scripts/run_dmc_tasks.sh
  bash scripts/run_dmc_tasks.sh --seeds 3,4,5 --device 0
  bash scripts/run_dmc_tasks.sh --tasks walker-run,cheetah-run -- --lamda 0.1
EOF
}

csv_to_array() {
    local csv="$1"
    local -n out="$2"
    IFS=',' read -r -a out <<< "$csv"
}

load_seed_group() {
    local group="$1"
    case "$group" in
        default|paper)
            SEEDS=(3 4 5)
            ;;
        debug)
            SEEDS=(3)
            ;;
        *)
            echo "Unknown seed group: ${group}" >&2
            exit 2
            ;;
    esac
}

TASKS=(
    dog-stand
    dog-walk
    dog-trot
    dog-run
    humanoid-stand
    humanoid-walk
    humanoid-run
    walker-stand
    walker-walk
    walker-run
    hopper-stand
    hopper-hop
    quadruped-walk
    quadruped-run
    cheetah-run
    fish-swim
    finger-spin
    cartpole-swingup
    pendulum-swingup
    reacher-easy
    cup-catch
)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --seed-group)
            SEED_GROUP="$2"
            shift 2
            ;;
        --seeds)
            SEEDS_CSV="$2"
            shift 2
            ;;
        --tasks)
            TASKS_CSV="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --exp-name)
            EXP_NAME="$2"
            shift 2
            ;;
        --log-dir)
            LOG_DIR="$2"
            shift 2
            ;;
        --retry-delay)
            RETRY_DELAY="$2"
            shift 2
            ;;
        --no-retry)
            RETRY_FAILED=0
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            EXTRA_ARGS=("$@")
            break
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

load_seed_group "$SEED_GROUP"

if [[ -n "$SEEDS_CSV" ]]; then
    csv_to_array "$SEEDS_CSV" SEEDS
fi

if [[ -n "$TASKS_CSV" ]]; then
    csv_to_array "$TASKS_CSV" TASKS
fi

mkdir -p "$LOG_DIR"
cd "$REPO_ROOT" || exit 1

echo "Repo: ${REPO_ROOT}"
echo "Seeds: ${SEEDS[*]}"
echo "Tasks: ${TASKS[*]}"
echo "Experiment: ${EXP_NAME}"
echo "Logs: ${LOG_DIR}"
echo

run_one() {
    local seed="$1"
    local task="$2"
    local timestamp
    local log_file
    local cmd

    timestamp="$(date +%Y%m%d_%H%M%S)"
    log_file="${LOG_DIR}/${timestamp}_${task}_seed${seed}.log"
    cmd=("$PYTHON_BIN" main.py --task "$task" --seed "$seed" --exp_name "$EXP_NAME")

    if [[ -n "$DEVICE" ]]; then
        cmd+=(--device "$DEVICE")
    fi
    if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
        cmd+=("${EXTRA_ARGS[@]}")
    fi

    echo "[$(date '+%F %T')] Starting task=${task} seed=${seed}"
    echo "Command: ${cmd[*]}"
    echo "Log: ${log_file}"

    "${cmd[@]}" > "$log_file" 2>&1
}

for seed in "${SEEDS[@]}"; do
    echo "========== Seed ${seed} =========="
    for task in "${TASKS[@]}"; do
        if [[ "$RETRY_FAILED" -eq 1 ]]; then
            until run_one "$seed" "$task"; do
                status=$?
                echo "[$(date '+%F %T')] Failed task=${task} seed=${seed} status=${status}; retrying in ${RETRY_DELAY}s"
                sleep "$RETRY_DELAY"
            done
        else
            run_one "$seed" "$task"
            status=$?
            if [[ "$status" -ne 0 ]]; then
                echo "[$(date '+%F %T')] Failed task=${task} seed=${seed} status=${status}; continuing"
            fi
        fi
        echo "[$(date '+%F %T')] Finished task=${task} seed=${seed}"
        echo
    done
done

echo "All requested DMC runs finished."
