#!/usr/bin/env bash
set -euo pipefail

TEST_DIR="/home/yy485/cuda-checkpoint/vllm-eep/tools/ep_kernels/ep_kernels_workspace/DeepEP/tests"
CUDA_CHECKPOINT_BIN="/home/yy485/cuda-checkpoint/bin/x86_64_Linux/cuda-checkpoint"
LOG_DIR="./deepep_ll_logs"
TEST_LOGFILE="${LOG_DIR}/test_low_latency.log"
STATE_LOGFILE="${LOG_DIR}/cuda_checkpoint_states.log"

ROOT_PID=""
CLEANUP_DONE=0

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
      printf 'pid=%s state=%s\n' "$pid" "$state"
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
    echo "${label} pid $pid ($(get_process_name "$pid"))"
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

collect_worker_pids() {
  local pid
  local cmdline

  WORKER_PIDS=()
  while IFS= read -r pid; do
    [[ -n "$pid" ]] || continue
    [[ -r "/proc/$pid/cmdline" ]] || continue
    cmdline="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)"
    if [[ "$cmdline" == *"multiprocessing.spawn"* ]] || [[ "$cmdline" == *"spawn_main"* ]]; then
      WORKER_PIDS+=("$pid")
    fi
  done < <(pgrep -P "$ROOT_PID" | sort -n)
}

print_process_snapshot() {
  echo "Current process tree:"
  ps -o pid,ppid,pgid,cmd --forest -g "$ROOT_PID" || true
}

main() {
  local -a reversed_pids=()
  local idx

  mkdir -p "$LOG_DIR"

  if [[ ! -d "$TEST_DIR" ]]; then
    echo "Test directory not found: $TEST_DIR" >&2
    exit 1
  fi

  if [[ ! -x "$CUDA_CHECKPOINT_BIN" ]]; then
    echo "cuda-checkpoint binary not found: $CUDA_CHECKPOINT_BIN" >&2
    exit 1
  fi

  if ! command -v python >/dev/null 2>&1; then
    echo "python is not available in the current shell." >&2
    exit 1
  fi

  sudo -v

  echo "Test log: $TEST_LOGFILE"
  echo "State log: $STATE_LOGFILE"
  echo "Launching: CUDA_VISIBLE_DEVICES=6,7 python test_low_latency.py --num-processes 2 --pressure-test"

  cd "$TEST_DIR"
  CUDA_VISIBLE_DEVICES=6,7 python test_low_latency.py --num-processes 2 --pressure-test
  ROOT_PID=$!

  echo "Root PID: $ROOT_PID"
  echo "Waiting 30 seconds before toggle."
  sleep 20

  if ! kill -0 "$ROOT_PID" >/dev/null 2>&1; then
    echo "test_low_latency.py exited before toggle." >&2
    tail -n 200 "$TEST_LOGFILE" || true
    exit 1
  fi

  collect_worker_pids

  if [[ "${#WORKER_PIDS[@]}" -ne 2 ]]; then
    echo "Expected 2 spawned worker PIDs after 30 seconds, found ${#WORKER_PIDS[@]}." >&2
    print_process_snapshot
    tail -n 200 "$TEST_LOGFILE" || true
    exit 1
  fi

  echo "Worker PIDs:"
  for idx in "${!WORKER_PIDS[@]}"; do
    printf '  worker %s pid %s (%s)\n' "$idx" "${WORKER_PIDS[$idx]}" "$(get_process_name "${WORKER_PIDS[$idx]}")"
  done

  record_state_snapshot "before toggle"

  if ! run_parallel_toggle "Checkpointing" "${WORKER_PIDS[@]}"; then
    echo "cuda-checkpoint toggle failed." >&2
    exit 1
  fi

  record_state_snapshot "after toggle"

  for (( idx=${#WORKER_PIDS[@]} - 1; idx >= 0; idx-- )); do
    reversed_pids+=("${WORKER_PIDS[idx]}")
  done

  if ! run_parallel_toggle "Toggling back" "${reversed_pids[@]}"; then
    echo "cuda-checkpoint toggle-back failed." >&2
    exit 1
  fi

  record_state_snapshot "after toggle back"

  echo "Waiting for test_low_latency.py to finish."
  wait "$ROOT_PID"
  ROOT_PID=""

  echo "Completed successfully."
  tail -n 40 "$TEST_LOGFILE" || true
}

main "$@"
