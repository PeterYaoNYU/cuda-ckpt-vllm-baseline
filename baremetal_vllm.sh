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
EXPERIMENT="${EXPERIMENT:-all}"         # all | private_shm | private_shm_fork
PRIVATE_SHM_SIZE="${PRIVATE_SHM_SIZE:-16G}"
IN_EXPERIMENT_NS="${IN_EXPERIMENT_NS:-0}"
BASE_LOGFILE="${LOGFILE:-vllm_baremetal.log}"
CUDA_CHECKPOINT_BIN="${CUDA_CHECKPOINT_BIN:-$HOME/cuda-checkpoint/bin/x86_64_Linux/cuda-checkpoint}"
CRIU_BIN="${CRIU_BIN:-criu}"
BASE_CKPT_DIR="${CKPT_DIR:-$PWD/checkpoint_vllm_baremetal}"
CLEANUP_ON_EXIT="${CLEANUP_ON_EXIT:-1}"

case "$EXPERIMENT" in
  private_shm)
    LOGFILE="${LOGFILE:-${BASE_LOGFILE%.log}_private_shm.log}"
    CKPT_DIR="${CKPT_DIR:-${BASE_CKPT_DIR}_private_shm}"
    ;;
  private_shm_fork)
    LOGFILE="${LOGFILE:-${BASE_LOGFILE%.log}_private_shm_fork.log}"
    CKPT_DIR="${CKPT_DIR:-${BASE_CKPT_DIR}_private_shm_fork}"
    ;;
  all)
    LOGFILE="${LOGFILE:-$BASE_LOGFILE}"
    CKPT_DIR="${CKPT_DIR:-$BASE_CKPT_DIR}"
    ;;
  *)
    echo "Unknown EXPERIMENT=$EXPERIMENT"
    exit 1
    ;;
esac

SERVER_PID=""
ORIGINAL_SERVER_PID=""
CLEANUP_DONE=0

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

process_uses_io_uring() {
  local pid="$1"
  local fd
  local target

  [[ -d "/proc/$pid" ]] || return 1

  if grep -q 'anon_inode:\[io_uring\]' "/proc/$pid/maps" 2>/dev/null; then
    return 0
  fi

  for fd in /proc/"$pid"/fd/*; do
    [[ -e "$fd" ]] || continue
    target="$(readlink "$fd" 2>/dev/null || true)"
    if [[ "$target" == "anon_inode:[io_uring]" ]]; then
      return 0
    fi
  done

  return 1
}

print_io_uring_processes() {
  local pid
  local found=0

  echo "vLLM processes using io_uring:"
  for pid in "${ALL_VLLM_PIDS[@]}"; do
    if process_uses_io_uring "$pid"; then
      found=1
      printf "  PID %s: %s\n" "$pid" "$(get_process_name "$pid")"
      grep -n 'anon_inode:\[io_uring\]' "/proc/$pid/maps" 2>/dev/null || true
      ls -l "/proc/$pid/fd" 2>/dev/null | grep 'io_uring' || true
    fi
  done

  if [[ "$found" -eq 0 ]]; then
    echo "  none"
  fi
}

print_shm_conflicts() {
  local pid
  local path
  local key
  local count
  local found=0
  declare -A path_counts=()
  declare -A path_pids=()
  declare -A path_seen=()

  for pid in "${ALL_VLLM_PIDS[@]}"; do
    while IFS= read -r path; do
      [[ -n "$path" ]] || continue
      key="${pid}:${path}"
      [[ -n "${path_seen[$key]:-}" ]] && continue
      path_seen["$key"]=1
      path_counts["$path"]=$(( ${path_counts["$path"]:-0} + 1 ))
      path_pids["$path"]+="${pid} "
    done < <(sed -n 's#.*\(/dev/shm/sem\.[^ ]*\).*#\1#p' "/proc/$pid/maps" 2>/dev/null | sort -u)
  done

  echo "Shared-memory semaphore paths seen in vLLM processes:"
  for path in "${!path_counts[@]}"; do
    found=1
    count="${path_counts[$path]}"
    printf "  %s\n" "$path"
    printf "    seen_in_pids: %s\n" "${path_pids[$path]}"
    printf "    pid_count: %s\n" "$count"
    if [[ -e "$path" ]]; then
      printf "    filesystem_entry_exists: yes\n"
    else
      printf "    filesystem_entry_exists: no\n"
    fi
    if (( count > 1 )); then
      printf "    duplicate_name_across_processes: yes\n"
    else
      printf "    duplicate_name_across_processes: no\n"
    fi
  done

  if [[ "$found" -eq 0 ]]; then
    echo "  none"
  fi
}

print_shm_path_owners() {
  local path="$1"
  local proc_dir
  local pid
  local fd
  local target
  local matched=0

  for proc_dir in /proc/[0-9]*; do
    pid="${proc_dir##*/}"

    if grep -F -q "$path" "/proc/$pid/maps" 2>/dev/null; then
      matched=1
      printf "      owner_pid(map): %s %s\n" "$pid" "$(get_process_name "$pid")"
      continue
    fi

    for fd in /proc/"$pid"/fd/*; do
      [[ -e "$fd" ]] || continue
      target="$(readlink "$fd" 2>/dev/null || true)"
      if [[ "$target" == "$path" || "$target" == "$path (deleted)" ]]; then
        matched=1
        printf "      owner_pid(fd): %s %s [%s -> %s]\n" \
          "$pid" "$(get_process_name "$pid")" "${fd##*/}" "$target"
      fi
    done
  done

  if [[ "$matched" -eq 0 ]]; then
    echo "      owner_pid: none found"
  fi
}

print_shm_owner_report() {
  local pid
  local path
  local seen=0
  declare -A reported_paths=()

  echo "Shared-memory semaphore owner report:"
  for pid in "${ALL_VLLM_PIDS[@]}"; do
    while IFS= read -r path; do
      [[ -n "$path" ]] || continue
      [[ -n "${reported_paths[$path]:-}" ]] && continue
      reported_paths["$path"]=1
      seen=1
      printf "  %s\n" "$path"
      print_shm_path_owners "$path"
    done < <(sed -n 's#.*\(/dev/shm/sem\.[^ ]*\).*#\1#p' "/proc/$pid/maps" 2>/dev/null | sort -u)
  done

  if [[ "$seen" -eq 0 ]]; then
    echo "  none"
  fi
}

print_process_helpers() {
  local proc_dir
  local pid
  local proc_name
  local found=0

  echo "Related multiprocessing helpers:"
  for proc_dir in /proc/[0-9]*; do
    pid="${proc_dir##*/}"
    proc_name="$(get_process_name "$pid")"
    if [[ "$proc_name" == *resource_tracker* || "$proc_name" == *spawn_main* ]]; then
      found=1
      printf "  PID %s: %s\n" "$pid" "$proc_name"
      printf "    parent_pid: %s\n" "$(get_parent_pid "$pid")"
      if is_descendant_of "$pid" "$SERVER_PID"; then
        printf "    under_vllm_root: yes\n"
      else
        printf "    under_vllm_root: no\n"
      fi
    fi
  done

  if [[ "$found" -eq 0 ]]; then
    echo "  none"
  fi
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
  ALL_UNDER_SERVER_ROOT=1

  for pid in "${ALL_VLLM_PIDS[@]}"; do
    if ! is_descendant_of "$pid" "$SERVER_PID"; then
      ALL_UNDER_SERVER_ROOT=0
    fi

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

run_all_experiments() {
  local script_path
  local exp
  local rc=0

  script_path="$(readlink -f "$0")"

  for exp in private_shm private_shm_fork; do
    echo
    echo "=============================="
    echo "Running experiment: $exp"
    echo "=============================="
    echo
    if ! EXPERIMENT="$exp" LOGFILE="" CKPT_DIR="" bash "$script_path"; then
      echo
      echo "Experiment failed: $exp"
      rc=1
    else
      echo
      echo "Experiment succeeded: $exp"
    fi
  done

  return "$rc"
}

enter_private_namespace_if_needed() {
  local script_path
  local workdir

  script_path="$(readlink -f "$0")"
  workdir="$PWD"

  [[ "$IN_EXPERIMENT_NS" == "0" ]] || return 0

  case "$EXPERIMENT" in
    private_shm|private_shm_fork)
      export IN_EXPERIMENT_NS=1
      exec sudo --preserve-env=PATH,HOME,USER,LOGNAME,SHELL,PWD,CONDA_PREFIX,LD_LIBRARY_PATH,PYTHONPATH,MODEL,CUDA_VISIBLE_DEVICES,UV_USE_IO_URING,VLLM_DISABLE_NCCL_FOR_DP_SYNCHRONIZATION,NCCL_IB_DISABLE,USE_LIBUV,PORT,DP_SIZE,GPU_MEMORY_UTILIZATION,MAX_MODEL_LEN,LOGFILE,CUDA_CHECKPOINT_BIN,CRIU_BIN,CKPT_DIR,CLEANUP_ON_EXIT,EXPERIMENT,PRIVATE_SHM_SIZE,IN_EXPERIMENT_NS,BASE_LOGFILE,BASE_CKPT_DIR \
        unshare --mount --ipc --fork bash -lc "
          set -euo pipefail
          cd \"$workdir\"
          mount --make-rprivate /
          mount -t tmpfs -o mode=1777,nosuid,nodev,size=${PRIVATE_SHM_SIZE} shm /dev/shm
          exec bash \"$script_path\"
        "
      ;;
  esac
}

if [[ "$EXPERIMENT" == "all" && "$IN_EXPERIMENT_NS" == "0" ]]; then
  run_all_experiments
  exit $?
fi

enter_private_namespace_if_needed

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

case "$EXPERIMENT" in
  private_shm)
    unset VLLM_WORKER_MULTIPROC_METHOD || true
    ;;
  private_shm_fork)
    export VLLM_WORKER_MULTIPROC_METHOD=fork
    ;;
esac

echo "========================================"
echo "Experiment: $EXPERIMENT"
echo "Log file:    $LOGFILE"
echo "CKPT dir:    $CKPT_DIR"
echo "========================================"
echo "Starting vLLM on bare metal..."
echo "Local vLLM version: $(get_local_vllm_version)"
printf 'Launch command: vllm'
printf ' %q' "${VLLM_ARGS[@]}"
printf '\n'

setsid vllm "${VLLM_ARGS[@]}" >"$LOGFILE" 2>&1 < /dev/null &
SERVER_PID=$!
ORIGINAL_SERVER_PID="$SERVER_PID"

echo "Server PID: $SERVER_PID"
echo "Experiment: $EXPERIMENT"
echo "Server mount namespace: $(readlink /proc/$SERVER_PID/ns/mnt)"
echo "Server IPC namespace:   $(readlink /proc/$SERVER_PID/ns/ipc)"
echo "Server /dev/shm mount:"
awk '$5=="/dev/shm"{print}' /proc/$SERVER_PID/mountinfo || true
echo "VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-<unset>}"
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
ALL_UNDER_SERVER_ROOT=1

collect_vllm_processes

if [[ "${#CUDA_PIDS[@]}" -eq 0 ]]; then
  echo "No CUDA-checkpointable PIDs found."
  tail -n 100 "$LOGFILE" || true
  exit 1
fi

compute_criu_root_pids

echo "All discovered vLLM PIDs:"
for pid in "${ALL_VLLM_PIDS[@]}"; do
  printf "  PID %s: %s\n" "$pid" "$(get_process_name "$pid")"
done

print_io_uring_processes
print_shm_conflicts
print_shm_owner_report
print_process_helpers

if [[ "$ALL_UNDER_SERVER_ROOT" -eq 1 ]]; then
  echo "All discovered vLLM processes are under root PID $SERVER_PID."
else
  echo "Discovered vLLM processes span multiple trees."
fi

echo "CRIU root PIDs:"
for pid in "${CRIU_ROOT_PIDS[@]}"; do
  printf "  PID %s: %s\n" "$pid" "$(get_process_name "$pid")"
done

echo "CUDA PIDs:"
for pid in "${CUDA_PIDS[@]}"; do
  printf "  PID %s: %s\n" "$pid" "$(get_process_name "$pid")"
done

for pid in "${CUDA_PIDS[@]}"; do
  echo "Checkpointing PID $pid ($(get_process_name "$pid"))"
  sudo "$CUDA_CHECKPOINT_BIN" --toggle --pid "$pid"
done

echo "States after checkpoint:"
for pid in "${CUDA_PIDS[@]}"; do
  printf "  PID %s: " "$pid"
  sudo "$CUDA_CHECKPOINT_BIN" --get-state --pid "$pid" || true
done

echo "Running CRIU dump..."
dump_criu_roots

echo "Running CRIU restore..."
restore_criu_roots
sleep 2

echo "criu restore completed"

echo "Re-discovering restored vLLM processes..."
collect_vllm_processes

if [[ "${#CUDA_PIDS[@]}" -eq 0 ]]; then
  echo "No restored CUDA-checkpointable PIDs found."
  exit 1
fi

echo "Restored CUDA PIDs:"
for pid in "${CUDA_PIDS[@]}"; do
  printf "  PID %s: %s\n" "$pid" "$(get_process_name "$pid")"
done

echo "Uncheckpointing in reverse order..."
for (( idx=${#CUDA_PIDS[@]} - 1; idx >= 0; idx-- )); do
  pid="${CUDA_PIDS[idx]}"
  echo "Uncheckpointing PID $pid ($(get_process_name "$pid"))"
  sudo "$CUDA_CHECKPOINT_BIN" --toggle --pid "$pid"
done

echo "States after uncheckpoint:"
for pid in "${CUDA_PIDS[@]}"; do
  printf "  PID %s: " "$pid"
  sudo "$CUDA_CHECKPOINT_BIN" --get-state --pid "$pid" || true
done

echo "Post-uncheckpoint request:"
if request_server; then
  echo
  echo "SUCCESS"
else
  echo
  echo "Post-uncheckpoint request failed."
  tail -n 100 "$LOGFILE" || true
  exit 1
fi
