#!/usr/bin/env bash
set -euo pipefail

export MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export UV_USE_IO_URING="${UV_USE_IO_URING:-0}"
export VLLM_DISABLE_NCCL_FOR_DP_SYNCHRONIZATION="${VLLM_DISABLE_NCCL_FOR_DP_SYNCHRONIZATION:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export USE_LIBUV="${USE_LIBUV:-0}"

PORT="${PORT:-8000}"
DP_SIZE="${DP_SIZE:-4}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.10}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
PRIVATE_SHM_SIZE="${PRIVATE_SHM_SIZE:-16G}"
IN_PRIVATE_SHM_NS="${IN_PRIVATE_SHM_NS:-0}"
LOGFILE="${LOGFILE:-vllm_baremetal.log}"
CUDA_CHECKPOINT_BIN="${CUDA_CHECKPOINT_BIN:-$HOME/cuda-checkpoint/bin/x86_64_Linux/cuda-checkpoint}"
CRIU_BIN="${CRIU_BIN:-criu}"
CKPT_DIR="${CKPT_DIR:-$PWD/checkpoint_vllm_baremetal}"
CLEANUP_ON_EXIT="${CLEANUP_ON_EXIT:-1}"

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

find_pids_by_name_prefix() {
  local prefix="$1"
  local pid
  local proc_name

  for proc_dir in /proc/[0-9]*; do
    pid="${proc_dir##*/}"
    proc_name="$(get_process_name "$pid")"
    if [[ "$proc_name" == "$prefix"* ]]; then
      echo "$pid"
    fi
  done | sort -n
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

  state="$(sudo "$CUDA_CHECKPOINT_BIN" --get-state --pid "$pid" 2>/dev/null || true)"
  if [[ "$state" == "running" ]]; then
    CUDA_PIDS+=("$pid")
    SEEN_PIDS["$pid"]=1
  fi
}

append_all_pids_by_name_prefix() {
  local prefix="$1"
  local label="${2:-$1}"
  local pid
  local matched=0

  while IFS= read -r pid; do
    matched=1
    append_all_pid "$pid"
  done < <(find_pids_by_name_prefix "$prefix")

  if [[ "$matched" -eq 0 ]]; then
    printf 'ERROR: expected process not found by name: %s\n' "$label" >&2
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
  ALL_VLLM_PIDS=()
  SEEN_PIDS=()
  SEEN_ALL_PIDS=()

  append_all_pid "$SERVER_PID"
  append_cuda_pid_if_running "$SERVER_PID"

  append_all_pids_by_name_prefix "VLLM::Worker" "VLLM::Worker"
  append_cuda_pids_by_name_prefix "VLLM::Worker" "VLLM::Worker"

  append_all_pids_by_name_prefix "VLLM::DPCoordinator" "VLLM::DPCoordinator"
  append_cuda_pids_by_name_prefix "VLLM::DPCoordinator" "VLLM::DPCoordinator"

  for (( idx=0; idx<DP_SIZE; idx++ )); do
    append_all_pids_by_name_prefix "VLLM::EngineCore_DP${idx}" "VLLM::EngineCore_DP${idx}"
    append_cuda_pids_by_name_prefix "VLLM::EngineCore_DP${idx}" "VLLM::EngineCore_DP${idx}"
  done

  for (( idx=0; idx<DP_SIZE; idx++ )); do
    append_all_pids_by_name_prefix "VLLM::APIServer_${idx}" "VLLM::APIServer_${idx}"
    append_cuda_pids_by_name_prefix "VLLM::APIServer_${idx}" "VLLM::APIServer_${idx}"
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
  exec sudo --preserve-env=PATH,HOME,USER,LOGNAME,SHELL,PWD,CONDA_PREFIX,LD_LIBRARY_PATH,PYTHONPATH,MODEL,CUDA_VISIBLE_DEVICES,UV_USE_IO_URING,VLLM_DISABLE_NCCL_FOR_DP_SYNCHRONIZATION,NCCL_IB_DISABLE,USE_LIBUV,PORT,DP_SIZE,GPU_MEMORY_UTILIZATION,MAX_MODEL_LEN,LOGFILE,CUDA_CHECKPOINT_BIN,CRIU_BIN,CKPT_DIR,CLEANUP_ON_EXIT,PRIVATE_SHM_SIZE,IN_PRIVATE_SHM_NS \
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

VLLM_ARGS=(
  serve "$MODEL"
  --data-parallel-size "$DP_SIZE"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --disable-log-stats
  --enforce-eager
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

echo "Server PID: $SERVER_PID"
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

aggregate_start_ms="$(now_ms)"

phase_start_ms="$(now_ms)"
for pid in "${CUDA_PIDS[@]}"; do
  echo "Checkpointing PID $pid ($(get_process_name "$pid"))"
  sudo "$CUDA_CHECKPOINT_BIN" --toggle --pid "$pid"
done
CUDA_TOGGLE_CHECKPOINT_MS="$(duration_ms "$phase_start_ms" "$(now_ms)")"

phase_start_ms="$(now_ms)"
echo "Running CRIU dump..."
dump_criu_roots
CRIU_DUMP_MS="$(duration_ms "$phase_start_ms" "$(now_ms)")"
update_image_size

phase_start_ms="$(now_ms)"
echo "Running CRIU restore..."
restore_criu_roots
# sleep 2
CRIU_RESTORE_MS="$(duration_ms "$phase_start_ms" "$(now_ms)")"

if [[ "${#CUDA_PIDS[@]}" -eq 0 ]]; then
  echo "No CUDA processes available to toggle back."
  exit 1
fi

phase_start_ms="$(now_ms)"
echo "Uncheckpointing in reverse order..."
for (( idx=${#CUDA_PIDS[@]} - 1; idx >= 0; idx-- )); do
  pid="${CUDA_PIDS[idx]}"
  echo "Uncheckpointing PID $pid ($(get_process_name "$pid"))"
  sudo "$CUDA_CHECKPOINT_BIN" --toggle --pid "$pid"
done
CUDA_TOGGLE_RESTORE_MS="$(duration_ms "$phase_start_ms" "$(now_ms)")"

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
