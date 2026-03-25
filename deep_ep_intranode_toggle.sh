#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$SCRIPT_DIR}"

CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
ENV_PREFIX="${ENV_PREFIX:-$REPO_ROOT/vllm-eep/foundry}"
TEST_SCRIPT="${TEST_SCRIPT:-$REPO_ROOT/vllm-eep/tools/ep_kernels/ep_kernels_workspace/DeepEP/tests/test_intranode.py}"
CUDA_CHECKPOINT_BIN="${CUDA_CHECKPOINT_BIN:-$HOME/cuda-checkpoint/bin/x86_64_Linux/cuda-checkpoint}"

NUM_PROCESSES="${NUM_PROCESSES:-4}"
HOLD_SECS="${HOLD_SECS:-60}"
WAIT_FOR_PIDS_SECS="${WAIT_FOR_PIDS_SECS:-300}"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/intranode_toggle_logs/$RUN_STAMP}"
TEST_LOGFILE="${TEST_LOGFILE:-$LOG_DIR/test_intranode.log}"
STATE_LOGFILE="${STATE_LOGFILE:-$LOG_DIR/cuda_checkpoint_states.log}"

ROOT_PID=""
CLEANUP_DONE=0

declare -A RANK_TO_PID=()
declare -A PID_TO_RANK=()
declare -a WORKER_PIDS=()

get_process_name() {
  local pid="$1"
  local proc_name=""

  if [[ -r "/proc/$pid/cmdline" ]]; then
    proc_name="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)"
  fi
  if [[ -z "$proc_name" ]] && [[ -r "/proc/$pid/comm" ]]; then
    proc_name="$(<"/proc/$pid/comm")"
  fi
  if [[ -z "$proc_name" ]]; then
    proc_name="<unknown>"
  fi

  printf '%s\n' "$proc_name"
}

kill_process_group() {
  local pid="$1"

  [[ -n "$pid" ]] || return 0
  kill -0 "$pid" >/dev/null 2>&1 || return 0

  echo "Cleaning up process group rooted at PID $pid ($(get_process_name "$pid"))"
  kill -- -"${pid}" >/dev/null 2>&1 || true
  wait "$pid" >/dev/null 2>&1 || true

  if kill -0 "$pid" >/dev/null 2>&1; then
    sleep 1
  fi
  if kill -0 "$pid" >/dev/null 2>&1; then
    kill -KILL -- -"${pid}" >/dev/null 2>&1 || true
  fi
}

cleanup() {
  [[ "$CLEANUP_DONE" == "0" ]] || return
  CLEANUP_DONE=1
  kill_process_group "${ROOT_PID:-}"
}

trap cleanup EXIT INT TERM

require_file() {
  local path="$1"
  local label="$2"

  if [[ ! -e "$path" ]]; then
    echo "$label not found: $path" >&2
    exit 1
  fi
}

activate_env() {
  require_file "$CONDA_SH" "conda init script"
  require_file "$ENV_PREFIX" "environment prefix"
  require_file "$TEST_SCRIPT" "test script"

  # shellcheck disable=SC1090
  source "$CONDA_SH"
  conda activate "$ENV_PREFIX"
}

parse_worker_pids_from_log() {
  local rank
  local pid

  while IFS=: read -r rank pid; do
    [[ -n "$rank" ]] || continue
    [[ -n "$pid" ]] || continue
    RANK_TO_PID["$rank"]="$pid"
    PID_TO_RANK["$pid"]="$rank"
  done < <(
    sed -n 's/.*\[checkpoint\] rank=\([0-9]\+\) pid=\([0-9]\+\).*/\1:\2/p' "$TEST_LOGFILE" \
      | sort -t: -k1,1n -u
  )
}

wait_for_worker_pids() {
  local deadline=$((SECONDS + WAIT_FOR_PIDS_SECS))
  local rank
  local pid

  while (( SECONDS < deadline )); do
    if ! kill -0 "$ROOT_PID" >/dev/null 2>&1; then
      echo "test_intranode exited before all worker PIDs were discovered." >&2
      tail -n 200 "$TEST_LOGFILE" || true
      exit 1
    fi

    parse_worker_pids_from_log

    WORKER_PIDS=()
    for (( rank=0; rank<NUM_PROCESSES; rank++ )); do
      pid="${RANK_TO_PID[$rank]:-}"
      [[ -n "$pid" ]] || continue
      if kill -0 "$pid" >/dev/null 2>&1; then
        WORKER_PIDS+=("$pid")
      fi
    done

    if [[ "${#WORKER_PIDS[@]}" -eq "$NUM_PROCESSES" ]]; then
      return 0
    fi

    sleep 1
  done

  echo "Timed out waiting for ${NUM_PROCESSES} worker PIDs in $TEST_LOGFILE" >&2
  tail -n 200 "$TEST_LOGFILE" || true
  exit 1
}

print_worker_pids() {
  local pid

  echo "Worker PIDs:"
  for pid in "${WORKER_PIDS[@]}"; do
    printf '  rank %s pid %s (%s)\n' "${PID_TO_RANK[$pid]}" "$pid" "$(get_process_name "$pid")"
  done
}

get_cuda_state() {
  local pid="$1"
  sudo "$CUDA_CHECKPOINT_BIN" --get-state --pid "$pid" 2>&1 || true
}

record_state_snapshot() {
  local label="$1"
  local pid
  local state

  {
    printf '=== %s ===\n' "$label"
    for pid in "${WORKER_PIDS[@]}"; do
      state="$(get_cuda_state "$pid")"
      printf 'rank=%s pid=%s state=%s\n' "${PID_TO_RANK[$pid]}" "$pid" "$state"
    done
    printf '\n'
  } | tee -a "$STATE_LOGFILE"
}

run_parallel_toggle() {
  local label="$1"
  shift
  local pid
  local bg_pid
  local -a bg_pids=()
  local rc=0

  for pid in "$@"; do
    echo "${label} rank ${PID_TO_RANK[$pid]} pid $pid"
    (
      sudo "$CUDA_CHECKPOINT_BIN" --toggle --pid "$pid"
    ) &
    bg_pids+=("$!")
  done

  for bg_pid in "${bg_pids[@]}"; do
    if ! wait "$bg_pid"; then
      rc=1
    fi
  done

  return "$rc"
}

main() {
  local rank
  local -a reversed_pids=()

  mkdir -p "$LOG_DIR"

  if [[ ! -x "$CUDA_CHECKPOINT_BIN" ]]; then
    echo "cuda-checkpoint binary not found: $CUDA_CHECKPOINT_BIN" >&2
    exit 1
  fi

  activate_env

  if ! command -v python >/dev/null 2>&1; then
    echo "python is not available after activating $ENV_PREFIX" >&2
    exit 1
  fi

  sudo -v

  echo "Test log: $TEST_LOGFILE"
  echo "State log: $STATE_LOGFILE"
  echo "Launching $TEST_SCRIPT with num-processes=$NUM_PROCESSES hold-secs=$HOLD_SECS"

  setsid python -u "$TEST_SCRIPT" \
    --num-processes "$NUM_PROCESSES" \
    --hold-secs "$HOLD_SECS" \
    --use-fabric \
    >"$TEST_LOGFILE" 2>&1 < /dev/null &
  ROOT_PID=$!

  echo "Root PID: $ROOT_PID"

  wait_for_worker_pids
  print_worker_pids

  record_state_snapshot "before toggle"

  if ! run_parallel_toggle "Checkpointing" "${WORKER_PIDS[@]}"; then
    echo "cuda-checkpoint toggle failed." >&2
    exit 1
  fi

  record_state_snapshot "after toggle"

  for (( rank=${#WORKER_PIDS[@]} - 1; rank>=0; rank-- )); do
    reversed_pids+=("${WORKER_PIDS[rank]}")
  done

  if ! run_parallel_toggle "Toggling back" "${reversed_pids[@]}"; then
    echo "cuda-checkpoint toggle-back failed." >&2
    exit 1
  fi

  record_state_snapshot "after toggle back"

  echo "Waiting for test_intranode to finish."
  wait "$ROOT_PID"
  ROOT_PID=""

  echo "Completed successfully."
  echo "Recent test output:"
  tail -n 40 "$TEST_LOGFILE" || true
}

main "$@"
