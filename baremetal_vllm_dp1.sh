#!/usr/bin/env bash
set -euo pipefail

export MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export UV_USE_IO_URING="${UV_USE_IO_URING:-0}"
export VLLM_DISABLE_NCCL_FOR_DP_SYNCHRONIZATION="${VLLM_DISABLE_NCCL_FOR_DP_SYNCHRONIZATION:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export USE_LIBUV="${USE_LIBUV:-0}"

PORT="${PORT:-8000}"
DP_SIZE="${DP_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.20}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
PRIVATE_SHM_SIZE="${PRIVATE_SHM_SIZE:-16G}"
IN_PRIVATE_SHM_NS="${IN_PRIVATE_SHM_NS:-0}"
LOGFILE="${LOGFILE:-vllm_baremetal_dp1.log}"
CUDA_CHECKPOINT_BIN="${CUDA_CHECKPOINT_BIN:-bin/x86_64_Linux/cuda-checkpoint}"
CRIU_BIN="${CRIU_BIN:-criu}"
CKPT_DIR="${CKPT_DIR:-$PWD/checkpoint_vllm_dp1}"
CLEANUP_ON_EXIT="${CLEANUP_ON_EXIT:-1}"
VLLM_ROOT_PID="${VLLM_ROOT_PID:-}"
SERVER_NAME_MATCH="${SERVER_NAME_MATCH:-vllm serve}"
ENGINE_NAME_PREFIX="${ENGINE_NAME_PREFIX:-VLLM::EngineCore}"

SERVER_PID=""
ORIGINAL_SERVER_PID=""
CLEANUP_DONE=0
IMAGE_SIZE_BYTES=0
IMAGE_SIZE_HUMAN="0"

CUDA_TOGGLE_CHECKPOINT_MS=0
CRIU_DUMP_MS=0
CRIU_RESTORE_MS=0
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
  print_timing_line "criu image write" "$CRIU_DUMP_MS"
  print_timing_line "criu restore" "$CRIU_RESTORE_MS"
  print_timing_line "cuda toggle back" "$CUDA_TOGGLE_RESTORE_MS"
  print_timing_line "aggregate" "$AGGREGATE_MS"
  printf '  %-28s %s (%s bytes)\n' "checkpoint image size" "$IMAGE_SIZE_HUMAN" "$IMAGE_SIZE_BYTES"
}

get_parent_pid() {
  local pid="$1"

  awk '/^PPid:/ { print $2 }' "/proc/$pid/status" 2>/dev/null || true
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

list_vllm_tree_pids() {
  local pid

  [[ -n "$VLLM_ROOT_PID" ]] || return 0
  [[ -d "/proc/$VLLM_ROOT_PID" ]] || return 0

  echo "$VLLM_ROOT_PID"
  while IFS= read -r pid; do
    [[ -n "$pid" ]] || continue
    echo "$pid"
  done < <(list_descendant_pids "$VLLM_ROOT_PID")
}

pid_uses_cuda_device() {
  local pid="$1"
  local fd_path
  local target

  [[ -d "/proc/$pid/fd" ]] || return 1

  shopt -s nullglob
  for fd_path in /proc/"$pid"/fd/*; do
    target="$(readlink "$fd_path" 2>/dev/null || true)"
    case "$target" in
      /dev/nvidia*)
        shopt -u nullglob
        return 0
        ;;
    esac
  done
  shopt -u nullglob

  return 1
}

get_cuda_checkpoint_state() {
  local pid="$1"

  sudo "$CUDA_CHECKPOINT_BIN" --get-state --pid "$pid" 2>/dev/null || true
}

append_all_pid() {
  local pid="$1"

  [[ -n "$pid" ]] || return
  [[ -d "/proc/$pid" ]] || return
  [[ -n "${SEEN_ALL_PIDS[$pid]:-}" ]] && return

  ALL_VLLM_PIDS+=("$pid")
  SEEN_ALL_PIDS["$pid"]=1
}

append_cuda_pid_if_running() {
  local pid="$1"
  local state

  [[ -n "$pid" ]] || return
  [[ -d "/proc/$pid" ]] || return
  [[ -n "${SEEN_PIDS[$pid]:-}" ]] && return

  state="$(get_cuda_checkpoint_state "$pid")"
  if [[ "$state" == "running" ]]; then
    CUDA_PIDS+=("$pid")
    SEEN_PIDS["$pid"]=1
  fi
}

pid_matches_checkpoint_target_name() {
  local pid="$1"
  local proc_name="$2"

  if [[ "$proc_name" == *"$SERVER_NAME_MATCH"* ]]; then
    return 0
  fi

  if [[ "$proc_name" == "$ENGINE_NAME_PREFIX"* ]]; then
    return 0
  fi

  return 1
}

collect_vllm_processes() {
  local pid
  local proc_name
  local uses_cuda
  local cuda_state

  ALL_VLLM_PIDS=()
  SEEN_ALL_PIDS=()

  echo "Scanning vLLM process tree rooted at PID $VLLM_ROOT_PID..."
  while IFS= read -r pid; do
    [[ -n "$pid" ]] || continue

    proc_name="$(get_process_name "$pid")"
    append_all_pid "$pid"

    if pid_uses_cuda_device "$pid"; then
      uses_cuda="yes"
    else
      uses_cuda="no"
    fi

    cuda_state="$(get_cuda_checkpoint_state "$pid")"

    printf '  PID %s: uses_cuda=%s checkpointable=%s state=%s name=%s\n' \
      "$pid" \
      "$uses_cuda" \
      "$([[ "$cuda_state" == "running" ]] && echo yes || echo no)" \
      "${cuda_state:-<none>}" \
      "$proc_name"
  done < <(list_vllm_tree_pids | sort -n -u)

  echo "Discovered process names in vLLM tree:"
  for pid in "${ALL_VLLM_PIDS[@]}"; do
    printf "  PID %s: %s\n" "$pid" "$(get_process_name "$pid")"
  done
}

select_checkpoint_targets() {
  local pid
  local proc_name
  local matched=0
  local added_before

  CUDA_PIDS=()
  SEEN_PIDS=()

  echo "Selecting checkpoint targets by process name..."
  for pid in "${ALL_VLLM_PIDS[@]}"; do
    proc_name="$(get_process_name "$pid")"
    if ! pid_matches_checkpoint_target_name "$pid" "$proc_name"; then
      continue
    fi

    matched=1
    added_before="${#CUDA_PIDS[@]}"
    append_cuda_pid_if_running "$pid"

    if [[ "${#CUDA_PIDS[@]}" -gt "$added_before" ]]; then
      printf "  matched checkpoint target PID %s: %s\n" "$pid" "$proc_name"
    else
      printf "  matched but not CUDA-checkpointable PID %s: %s (state=%s)\n" \
        "$pid" \
        "$proc_name" \
        "$(get_cuda_checkpoint_state "$pid")"
    fi
  done

  if [[ "$matched" -eq 0 ]]; then
    echo "No checkpoint target names matched in the vLLM process tree."
  fi
}

print_checkpoint_target_states() {
  local label="$1"
  local pid
  local state

  echo "$label"
  for pid in "${CUDA_PIDS[@]}"; do
    state="$(get_cuda_checkpoint_state "$pid")"
    printf "  PID %s: state=%s name=%s\n" "$pid" "${state:-<none>}" "$(get_process_name "$pid")"
  done
}

compute_criu_root_pids() {
  local pid
  local parent

  CRIU_ROOT_PIDS=()
  SEEN_CRIU_ROOTS=()

  for pid in "${ALL_VLLM_PIDS[@]}"; do
    parent="$(get_parent_pid "$pid")"
    if [[ -z "${SEEN_ALL_PIDS[$parent]:-}" && -z "${SEEN_CRIU_ROOTS[$pid]:-}" ]]; then
      CRIU_ROOT_PIDS+=("$pid")
      SEEN_CRIU_ROOTS["$pid"]=1
    fi
  done
}

dump_criu_roots() {
  local root_pid
  local tree_dir

  mkdir -p "$CKPT_DIR"

  for root_pid in "${CRIU_ROOT_PIDS[@]}"; do
    tree_dir="$CKPT_DIR/tree_${root_pid}"
    rm -rf "$tree_dir"
    mkdir -p "$tree_dir"
    echo "CRIU dump root PID $root_pid ($(get_process_name "$root_pid")) -> $tree_dir"
    sudo "$CRIU_BIN" dump \
      --tree "$root_pid" \
      --images-dir "$tree_dir" \
      --tcp-established \
      --ext-unix-sk \
      --link-remap \
      -o "$tree_dir/dump.log" \
      -v4
  done
}

restore_criu_roots() {
  local root_pid
  local tree_dir
  local restored_pid

  for root_pid in "${CRIU_ROOT_PIDS[@]}"; do
    tree_dir="$CKPT_DIR/tree_${root_pid}"
    echo "CRIU restore root PID $root_pid from $tree_dir"
    sudo "$CRIU_BIN" restore \
      --images-dir "$tree_dir" \
      --tcp-established \
      --ext-unix-sk \
      --link-remap \
      --restore-detached \
      --pidfile "$tree_dir/restored_root.pid" \
      -o "$tree_dir/restore.log" \
      -v4

    restored_pid="$(sudo cat "$tree_dir/restored_root.pid" 2>/dev/null || true)"
    if [[ -z "$restored_pid" ]]; then
      echo "Failed to read restored PID for CRIU root $root_pid"
      exit 1
    fi

    echo "Restored root PID for original $root_pid: $restored_pid"
    if [[ "$root_pid" == "$ORIGINAL_SERVER_PID" ]]; then
      SERVER_PID="$restored_pid"
    fi
  done
}

update_image_size() {
  IMAGE_SIZE_BYTES="$(du -sb "$CKPT_DIR" 2>/dev/null | awk '{print $1}')"
  IMAGE_SIZE_HUMAN="$(du -sh "$CKPT_DIR" 2>/dev/null | awk '{print $1}')"
  IMAGE_SIZE_BYTES="${IMAGE_SIZE_BYTES:-0}"
  IMAGE_SIZE_HUMAN="${IMAGE_SIZE_HUMAN:-0}"
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
  kill -- -"${pid}" >/dev/null 2>&1 || sudo kill -- -"${pid}" >/dev/null 2>&1 || true
  wait "$pid" >/dev/null 2>&1 || true

  if kill -0 "$pid" >/dev/null 2>&1; then
    sleep 1
  fi
  if kill -0 "$pid" >/dev/null 2>&1; then
    kill -KILL -- -"${pid}" >/dev/null 2>&1 || sudo kill -KILL -- -"${pid}" >/dev/null 2>&1 || true
  fi
}

cleanup() {
  [[ "$CLEANUP_DONE" == "0" ]] || return
  CLEANUP_DONE=1

  [[ "$CLEANUP_ON_EXIT" == "1" ]] || return

  kill_process_group "${SERVER_PID:-}"
  if [[ -n "${ORIGINAL_SERVER_PID:-}" && "${ORIGINAL_SERVER_PID}" != "${SERVER_PID:-}" ]]; then
    kill_process_group "$ORIGINAL_SERVER_PID"
  fi
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

enter_private_shm_namespace_if_needed() {
  local script_path
  local workdir

  script_path="$(readlink -f "$0")"
  workdir="$PWD"

  [[ "$IN_PRIVATE_SHM_NS" == "0" ]] || return 0

  export IN_PRIVATE_SHM_NS=1
  exec sudo --preserve-env=PATH,HOME,USER,LOGNAME,SHELL,PWD,CONDA_PREFIX,LD_LIBRARY_PATH,PYTHONPATH,MODEL,CUDA_VISIBLE_DEVICES,UV_USE_IO_URING,VLLM_DISABLE_NCCL_FOR_DP_SYNCHRONIZATION,NCCL_IB_DISABLE,USE_LIBUV,PORT,DP_SIZE,GPU_MEMORY_UTILIZATION,MAX_MODEL_LEN,LOGFILE,CUDA_CHECKPOINT_BIN,CRIU_BIN,CKPT_DIR,CLEANUP_ON_EXIT,PRIVATE_SHM_SIZE,IN_PRIVATE_SHM_NS,VLLM_ROOT_PID \
    unshare --mount --ipc --fork bash -lc "
      set -euo pipefail
      cd \"$workdir\"
      mount --make-rprivate /
      mount -t tmpfs -o mode=1777,nosuid,nodev,size=${PRIVATE_SHM_SIZE} shm /dev/shm
      exec bash \"$script_path\"
    "
}

enter_private_shm_namespace_if_needed

if ! command -v vllm >/dev/null 2>&1; then
  echo "vllm is not in PATH."
  exit 1
fi

if [[ ! -x "${CUDA_CHECKPOINT_BIN}" ]]; then
  echo "cuda-checkpoint binary not found: ${CUDA_CHECKPOINT_BIN}"
  exit 1
fi

if ! command -v "$CRIU_BIN" >/dev/null 2>&1; then
  echo "criu is not in PATH."
  exit 1
fi

# Refresh sudo credentials once before the parallel toggle phases.
sudo -v

VLLM_ARGS=(
  serve "$MODEL"
  --data-parallel-size 1
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --cudagraph-capture-sizes 512
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
ORIGINAL_SERVER_PID="$SERVER_PID"
if [[ -z "$VLLM_ROOT_PID" ]]; then
  VLLM_ROOT_PID="$SERVER_PID"
fi

echo "Server PID: $SERVER_PID"
echo "vLLM root PID: $VLLM_ROOT_PID"
echo "Server mount namespace: $(readlink /proc/$SERVER_PID/ns/mnt)"
echo "Server IPC namespace:   $(readlink /proc/$SERVER_PID/ns/ipc)"
echo "Server /dev/shm mount:"
awk '$5=="/dev/shm"{print}' /proc/$SERVER_PID/mountinfo || true
echo "Log file: $LOGFILE"

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
declare -a ALL_VLLM_PIDS=()
declare -a CRIU_ROOT_PIDS=()
declare -A SEEN_PIDS=()
declare -A SEEN_ALL_PIDS=()
declare -A SEEN_CRIU_ROOTS=()

collect_vllm_processes
select_checkpoint_targets

if [[ "${#ALL_VLLM_PIDS[@]}" -eq 0 ]]; then
  echo "No vLLM processes found under root PID $VLLM_ROOT_PID."
  tail -n 100 "$LOGFILE" || true
  exit 1
fi

if [[ "${#CUDA_PIDS[@]}" -eq 0 ]]; then
  echo "No CUDA-checkpointable PIDs found."
  tail -n 100 "$LOGFILE" || true
  exit 1
fi

compute_criu_root_pids

if [[ "${#CRIU_ROOT_PIDS[@]}" -eq 0 ]]; then
  echo "No CRIU root PIDs found."
  exit 1
fi

echo "CRIU root PIDs:"
for pid in "${CRIU_ROOT_PIDS[@]}"; do
  printf "  PID %s: %s\n" "$pid" "$(get_process_name "$pid")"
done

echo "CUDA PIDs:"
for pid in "${CUDA_PIDS[@]}"; do
  printf "  PID %s: %s\n" "$pid" "$(get_process_name "$pid")"
done

print_checkpoint_target_states "CUDA states before checkpoint toggle:"

aggregate_start_ms="$(now_ms)"

phase_start_ms="$(now_ms)"
if ! run_parallel_toggle "Checkpointing" "${CUDA_PIDS[@]}"; then
  echo "Parallel CUDA checkpoint toggle failed."
  exit 1
fi
CUDA_TOGGLE_CHECKPOINT_MS="$(duration_ms "$phase_start_ms" "$(now_ms)")"

print_checkpoint_target_states "CUDA states after checkpoint toggle:"

phase_start_ms="$(now_ms)"
echo "Running CRIU dump..."
dump_criu_roots
CRIU_DUMP_MS="$(duration_ms "$phase_start_ms" "$(now_ms)")"
update_image_size

phase_start_ms="$(now_ms)"
echo "Running CRIU restore..."
restore_criu_roots
CRIU_RESTORE_MS="$(duration_ms "$phase_start_ms" "$(now_ms)")"

if [[ "${#CUDA_PIDS[@]}" -eq 0 ]]; then
  echo "No CUDA processes available to toggle back."
  exit 1
fi

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

print_checkpoint_target_states "CUDA states after toggle back:"

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
