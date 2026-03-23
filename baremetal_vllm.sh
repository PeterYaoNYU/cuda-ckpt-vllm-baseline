#!/usr/bin/env bash
set -euo pipefail

export MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

PORT="${PORT:-8000}"
DP_SIZE="${DP_SIZE:-4}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
LOGFILE="${LOGFILE:-vllm_baremetal.log}"
CUDA_CHECKPOINT_BIN="${CUDA_CHECKPOINT_BIN:-$HOME/cuda-checkpoint/bin/x86_64_Linux/cuda-checkpoint}"

SERVER_PID=""

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

cleanup() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
    kill -- -"${SERVER_PID}" >/dev/null 2>&1 || true
    wait "${SERVER_PID}" >/dev/null 2>&1 || true
  fi
}

trap cleanup EXIT

if ! command -v vllm >/dev/null 2>&1; then
  echo "vllm is not in PATH."
  exit 1
fi

if [[ ! -x "${CUDA_CHECKPOINT_BIN}" ]]; then
  echo "cuda-checkpoint binary not found: ${CUDA_CHECKPOINT_BIN}"
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

echo "Server PID: $SERVER_PID"
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
declare -A SEEN_PIDS=()

append_cuda_pid_if_running "$SERVER_PID"
append_cuda_pids_by_name_prefix "VLLM::Worker" "VLLM::Worker"
append_cuda_pids_by_name_prefix "VLLM::DPCoordinator" "VLLM::DPCoordinator"
for (( idx=0; idx<DP_SIZE; idx++ )); do
  append_cuda_pids_by_name_prefix "VLLM::EngineCore_DP${idx}" "VLLM::EngineCore_DP${idx}"
done
for (( idx=0; idx<DP_SIZE; idx++ )); do
  append_cuda_pids_by_name_prefix "VLLM::APIServer_${idx}" "VLLM::APIServer_${idx}"
done

if [[ "${#CUDA_PIDS[@]}" -eq 0 ]]; then
  echo "No CUDA-checkpointable PIDs found."
  tail -n 100 "$LOGFILE" || true
  exit 1
fi

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
