#!/usr/bin/env bash
set -euo pipefail

# Host prerequisites:
# - docker works
# - nvidia-container-toolkit works
# - your host cuda-checkpoint repo/binary already exists
#   e.g. /home/yy485/cuda-checkpoint/bin/x86_64_Linux/cuda-checkpoint

export IMAGE_LATEST="vllm/vllm-openai:latest"
export IMAGE_FALLBACK="vllm/vllm-openai:v0.6.6.post1"
export MODEL="Qwen/Qwen3-0.6B"
export CUDA_CHECKPOINT_HOST_DIR="$HOME/cuda-checkpoint"
export HF_CACHE_DIR="$HOME/.cache/huggingface"

probe_container() {
  local cid="$1"
  request_container "$cid" >/dev/null
}

request_container() {
  local cid="$1"
  curl --max-time 60 --silent http://127.0.0.1:8000/v1/chat/completions \
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

run_one_image() {
  local image="$1"

  echo "=== Testing image: $image ==="

  docker pull "$image"
  docker rm -f vllm 2>/dev/null || true

  local cid
  cid="$(docker run --rm --name vllm --detach \
    --gpus all \
    -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
    -v "$HF_CACHE_DIR:/root/.cache/huggingface" \
    -v "$CUDA_CHECKPOINT_HOST_DIR:/cuda-checkpoint" \
    --ipc=host \
    --network=host \
    --shm-size=16g \
    "$image" \
      --model "$MODEL" \
      --data-parallel-size 4 \
      --gpu-memory-utilization 0.06 \
      --disable-log-stats \
      --enforce-eager)"

  echo "Container: $cid"

  echo "Waiting for vLLM..."
  for _ in $(seq 1 300); do
    if ! docker exec "$cid" true >/dev/null 2>&1; then
      echo "Container exited early. Logs:"
      docker logs "$cid" || true
      docker rm -f "$cid" >/dev/null 2>&1 || true
      return 1
    fi
    if probe_container "$cid"; then
      echo "vLLM is ready."
      break
    fi
    sleep 2
  done

  if ! probe_container "$cid"; then
    echo "vLLM never became ready. Logs:"
    docker logs "$cid" || true
    docker rm -f "$cid" >/dev/null 2>&1 || true
    return 1
  fi

  echo "Pre-checkpoint request:"
  if ! request_container "$cid"; then
    echo
    echo "Pre-checkpoint request failed."
    docker logs "$cid" || true
    docker rm -f "$cid" >/dev/null 2>&1 || true
    return 1
  fi
  echo

  mapfile -t CUDA_PIDS < <(
    while IFS= read -r pid; do
      state="$(docker exec "$cid" /cuda-checkpoint/bin/x86_64_Linux/cuda-checkpoint --get-state --pid "$pid" 2>/dev/null || true)"
      if [[ "$state" == "running" ]]; then
        echo "$pid"
      fi
    done < <(docker exec "$cid" ls -1 /proc | grep -E '^[0-9]+$')
  )

  if [[ "${#CUDA_PIDS[@]}" -eq 0 ]]; then
    echo "No CUDA-checkpointable PIDs found."
    docker logs "$cid" || true
    docker rm -f "$cid" >/dev/null 2>&1 || true
    return 1
  fi

  echo "CUDA PIDs: ${CUDA_PIDS[*]}"

  for p in "${CUDA_PIDS[@]}"; do
    echo "Checkpointing PID $p"
    docker exec "$cid" /cuda-checkpoint/bin/x86_64_Linux/cuda-checkpoint --toggle --pid "$p"
  done

  echo "States after checkpoint:"
  for p in "${CUDA_PIDS[@]}"; do
    printf "  PID %s: " "$p"
    docker exec "$cid" /cuda-checkpoint/bin/x86_64_Linux/cuda-checkpoint --get-state --pid "$p" || true
  done

  echo "Uncheckpointing in reverse order..."
  for (( idx=${#CUDA_PIDS[@]}-1 ; idx>=0 ; idx-- )); do
    p="${CUDA_PIDS[idx]}"
    echo "Uncheckpointing PID $p"
    docker exec "$cid" /cuda-checkpoint/bin/x86_64_Linux/cuda-checkpoint --toggle --pid "$p"
  done

  echo "States after uncheckpoint:"
  for p in "${CUDA_PIDS[@]}"; do
    printf "  PID %s: " "$p"
    docker exec "$cid" /cuda-checkpoint/bin/x86_64_Linux/cuda-checkpoint --get-state --pid "$p" || true
  done

  echo "Post-uncheckpoint request:"
  if request_container "$cid"; then
    echo
    echo "SUCCESS on $image"
    docker rm -f "$cid" >/dev/null 2>&1 || true
    return 0
  else
    echo
    echo "Post-uncheckpoint request failed on $image"
    docker logs "$cid" || true
    docker rm -f "$cid" >/dev/null 2>&1 || true
    return 1
  fi
}

# Sanity check: Docker can see GPUs
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi -L

# Try latest first, then fall back to the older repro image.
run_one_image "$IMAGE_LATEST" 
# || run_one_image "$IMAGE_FALLBACK"