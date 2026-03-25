#!/usr/bin/env bash
set -euo pipefail

export MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
export UV_USE_IO_URING="${UV_USE_IO_URING:-0}"
export VLLM_DISABLE_NCCL_FOR_DP_SYNCHRONIZATION="${VLLM_DISABLE_NCCL_FOR_DP_SYNCHRONIZATION:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export USE_LIBUV="${USE_LIBUV:-0}"

PORT="${PORT:-8011}"
DP_SIZE="${DP_SIZE:-2}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.30}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
LOGFILE="${LOGFILE:-vllm_baremetal_toggle_parallel.log}"
CUDA_CHECKPOINT_BIN="${CUDA_CHECKPOINT_BIN:-bin/x86_64_Linux/cuda-checkpoint}"
CLEANUP_ON_EXIT="${CLEANUP_ON_EXIT:-1}"
VLLM_ROOT_PID="${VLLM_ROOT_PID:-}"
STATE_LOGFILE="${STATE_LOGFILE:-vllm_baremetal_toggle_parallel.states.log}"

SERVER_PID=""
CLEANUP_DONE=0

CUDA_TOGGLE_CHECKPOINT_MS=0
CUDA_TOGGLE_RESTORE_MS=0
AGGREGATE_MS=0

request_server() {
  curl --max-time 60 --silent "http://127.0.0.1:${PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"$MODEL\",
      \"messages\": [
        {\"role\": \"user\", \"content\": \"Say hello in one short sentence.\"}
      ],
      \"max_tokens\": 32,
      \"temperature\": 0
    }"
}

probe_server() {
  request_server >/dev/null
}

get_local_vllm_version() {
  python -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo "<unknown>"
}

now_ms() {
  date +%s%3N
}

duration_ms() {
  local start_ms="$1"
  local end_ms="$2"
  echo $(( end_ms - start_ms ))
}

print_timing_line() {
  local label="$1"
  local value_ms="$2"

  printf '  %-28s %8d ms (%.3f s)\n' "$label" "$value_ms" "$(awk "BEGIN { printf \"%.3f\", $value_ms / 1000 }")"
}

print_timing_summary() {
  echo
  echo "Timing summary:"
  print_timing_line "cuda checkpoint toggle" "$CUDA_TOGGLE_CHECKPOINT_MS"
  print_timing_line "cuda toggle back" "$CUDA_TOGGLE_RESTORE_MS"
  print_timing_line "aggregate" "$AGGREGATE_MS"
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
    for pid in "${CUDA_PIDS[@]}"; do
      state="$(get_cuda_state "$pid")"
      printf 'pid=%s state=%s\n' "$pid" "$state"
    done
    printf '\n'
  } | tee -a "$STATE_LOGFILE"
}

get_parent_pid() {
  local pid="$1"

  awk '/^PPid:/ { print $2 }' "/proc/$pid/status" 2>/dev/null || true
}

is_descendant_of() {
  local pid="$1"
  local root="$2"
  local parent

  while [[ -n "$pid" && "$pid" != "0" ]]; do
    if [[ "$pid" == "$root" ]]; then
      return 0
    fi
    parent="$(get_parent_pid "$pid")"
    [[ -n "$parent" && "$parent" != "$pid" ]] || break
    pid="$parent"
  done

  return 1
}

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

find_pids_by_name_prefix() {
  local prefix="$1"
  local pid
  local proc_name
  local -a candidate_pids=()

  if [[ -n "$VLLM_ROOT_PID" ]] && [[ -d "/proc/$VLLM_ROOT_PID" ]]; then
    mapfile -t candidate_pids < <(list_descendant_pids "$VLLM_ROOT_PID")
  else
    for proc_dir in /proc/[0-9]*; do
      candidate_pids+=("${proc_dir##*/}")
    done
  fi

  for pid in "${candidate_pids[@]}"; do
    proc_name="$(get_process_name "$pid")"
    if [[ "$proc_name" == "$prefix"* ]]; then
      echo "$pid"
    fi
  done | sort -n
}

list_descendant_pids() {
  local root="$1"
  local parent
  local child
  local -a frontier=("$root")
  local -a next_frontier=()
  local -A seen=()

  while [[ "${#frontier[@]}" -gt 0 ]]; do
    next_frontier=()
    for parent in "${frontier[@]}"; do
      while IFS= read -r child; do
        [[ -n "$child" ]] || continue
        [[ -d "/proc/$child" ]] || continue
        [[ -n "${seen[$child]:-}" ]] && continue

        seen["$child"]=1
        echo "$child"
        next_frontier+=("$child")
      done < <(pgrep -P "$parent" || true)
    done
    frontier=("${next_frontier[@]}")
  done
}

append_cuda_pid_if_running() {
  local pid="$1"
  local state

  [[ -n "$pid" ]] || return
  [[ -d "/proc/$pid" ]] || return
  [[ -n "${SEEN_PIDS[$pid]:-}" ]] && return

  state="$(sudo "$CUDA_CHECKPOINT_BIN" --get-state --pid "$pid" 2>/dev/null || true)"
  if [[ "$state" == "running" ]]; then
    CUDA_PIDS+=("$pid")
    SEEN_PIDS["$pid"]=1
  fi
}

append_cuda_pids_by_name_prefix() {
  local prefix="$1"
  local label="${2:-$1}"
  local pid
  local matched=0
  local added_before

  added_before="${#CUDA_PIDS[@]}"

  while IFS= read -r pid; do
    matched=1
    append_cuda_pid_if_running "$pid"
  done < <(find_pids_by_name_prefix "$prefix")

  if [[ "$matched" -eq 0 ]]; then
    printf 'ERROR: expected process not found by name: %s\n' "$label" >&2
  elif [[ "${#CUDA_PIDS[@]}" -eq "$added_before" ]]; then
    printf 'ERROR: found process by name but none were CUDA-checkpointable: %s\n' "$label" >&2
  fi
}

collect_vllm_processes() {
  CUDA_PIDS=()
  SEEN_PIDS=()

  append_cuda_pid_if_running "$SERVER_PID"

  append_cuda_pids_by_name_prefix "VLLM::Worker" "VLLM::Worker"
  append_cuda_pids_by_name_prefix "VLLM::DPCoordinator" "VLLM::DPCoordinator"

  for (( idx=0; idx<DP_SIZE; idx++ )); do
    append_cuda_pids_by_name_prefix "VLLM::EngineCore_DP${idx}" "VLLM::EngineCore_DP${idx}"
  done

  for (( idx=0; idx<DP_SIZE; idx++ )); do
    append_cuda_pids_by_name_prefix "VLLM::APIServer_${idx}" "VLLM::APIServer_${idx}"
  done
}

run_parallel_toggle() {
  local action="$1"
  shift
  local pid
  local -a pids=("$@")
  local -a bg_pids=()
  local bg_pid
  local rc=0

  [[ "${#pids[@]}" -gt 0 ]] || return 0

  for pid in "${pids[@]}"; do
    echo "${action} PID $pid ($(get_process_name "$pid"))"
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

  [[ "$CLEANUP_ON_EXIT" == "1" ]] || return
  kill_process_group "${SERVER_PID:-}"
}

handle_exit_signal() {
  local sig="$1"
  trap - EXIT INT TERM
  cleanup
  case "$sig" in
    INT) exit 130 ;;
    TERM) exit 143 ;;
  esac
}

trap cleanup EXIT
trap 'handle_exit_signal INT' INT
trap 'handle_exit_signal TERM' TERM

if ! command -v vllm >/dev/null 2>&1; then
  echo "vllm is not in PATH."
  exit 1
fi

if [[ ! -x "${CUDA_CHECKPOINT_BIN}" ]]; then
  echo "cuda-checkpoint binary not found: ${CUDA_CHECKPOINT_BIN}"
  exit 1
fi

sudo -v

VLLM_ARGS=(
  serve "$MODEL"
  --data-parallel-size "$DP_SIZE"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --disable-log-stats
)

if [[ -n "$PORT" ]]; then
  VLLM_ARGS+=(--port "$PORT")
fi

if [[ -n "$MAX_MODEL_LEN" ]]; then
  VLLM_ARGS+=(--max-model-len "$MAX_MODEL_LEN")
fi

echo "Starting vLLM on bare metal..."
echo "Local vLLM version: $(get_local_vllm_version)"
printf 'Launch command: vllm'
printf ' %q' "${VLLM_ARGS[@]}"
printf '\n'

setsid vllm "${VLLM_ARGS[@]}" >"$LOGFILE" 2>&1 < /dev/null &
SERVER_PID=$!
if [[ -z "$VLLM_ROOT_PID" ]]; then
  VLLM_ROOT_PID="$SERVER_PID"
fi

echo "Server PID: $SERVER_PID"
echo "vLLM root PID: $VLLM_ROOT_PID"
echo "Log file: $LOGFILE"
echo "State log: $STATE_LOGFILE"

echo "Waiting for vLLM..."
for _ in $(seq 1 300); do
  if ! kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    echo "vLLM exited early. Recent logs:"
    tail -n 100 "$LOGFILE" || true
    exit 1
  fi
  if probe_server; then
    echo "vLLM is ready."
    break
  fi
  sleep 2
done

if ! probe_server; then
  echo "vLLM never became ready. Recent logs:"
  tail -n 100 "$LOGFILE" || true
  exit 1
fi

echo "Pre-checkpoint request:"
if ! request_server; then
  echo
  echo "Pre-checkpoint request failed."
  tail -n 100 "$LOGFILE" || true
  exit 1
fi
echo

declare -a CUDA_PIDS=()
declare -A SEEN_PIDS=()

collect_vllm_processes

if [[ "${#CUDA_PIDS[@]}" -eq 0 ]]; then
  echo "No CUDA-checkpointable PIDs found."
  tail -n 100 "$LOGFILE" || true
  exit 1
fi

echo "CUDA PIDs:"
for pid in "${CUDA_PIDS[@]}"; do
  printf "  PID %s: %s\n" "$pid" "$(get_process_name "$pid")"
done

record_state_snapshot "before toggle"

aggregate_start_ms="$(now_ms)"

phase_start_ms="$(now_ms)"
if ! run_parallel_toggle "Checkpointing" "${CUDA_PIDS[@]}"; then
  echo "Parallel CUDA checkpoint toggle failed."
  exit 1
fi
CUDA_TOGGLE_CHECKPOINT_MS="$(duration_ms "$phase_start_ms" "$(now_ms)")"

record_state_snapshot "after toggle"

declare -a REVERSED_CUDA_PIDS=()
for (( idx=${#CUDA_PIDS[@]} - 1; idx >= 0; idx-- )); do
  REVERSED_CUDA_PIDS+=("${CUDA_PIDS[idx]}")
done

phase_start_ms="$(now_ms)"
if ! run_parallel_toggle "Uncheckpointing" "${REVERSED_CUDA_PIDS[@]}"; then
  echo "Parallel CUDA toggle back failed."
  exit 1
fi
CUDA_TOGGLE_RESTORE_MS="$(duration_ms "$phase_start_ms" "$(now_ms)")"

record_state_snapshot "after toggle back"

AGGREGATE_MS="$(duration_ms "$aggregate_start_ms" "$(now_ms)")"

echo "Post-uncheckpoint request:"
if request_server; then
  echo
  echo "SUCCESS"
  print_timing_summary
else
  echo
  echo "Post-uncheckpoint request failed."
  print_timing_summary
  tail -n 100 "$LOGFILE" || true
  exit 1
fi
