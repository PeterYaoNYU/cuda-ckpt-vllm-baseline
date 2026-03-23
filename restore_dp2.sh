export CKPT_DIR=$PWD/checkpoint_vllm_dp2_new
RESTORED_ROOT_PID=1826371

get_descendants() {
  local parent="$1"
  local child
  for child in $(pgrep -P "$parent"); do
    echo "$child"
    get_descendants "$child"
  done
}

mapfile -t ALL_PIDS < <(
  {
    echo "$RESTORED_ROOT_PID"
    get_descendants "$RESTORED_ROOT_PID"
  } | sort -nu
)

echo "All restored descendant process PIDs:"
printf '  %s\n' "${ALL_PIDS[@]}"

mapfile -t GPU_PIDS < <(
  for p in "${ALL_PIDS[@]}"; do
    if sudo find "/proc/$p/fd" -maxdepth 1 -lname '/dev/nvidia*' -print -quit 2>/dev/null | grep -q .; then
      echo "$p"
    fi
  done | sort -nu
)

echo "Restored CUDA/NVIDIA-holding process PIDs:"
printf '  %s\n' "${GPU_PIDS[@]}"

for p in "${GPU_PIDS[@]}"; do
  echo "Resuming CUDA for PID $p"
  sudo /home/yy485/cuda-checkpoint/bin/x86_64_Linux/cuda-checkpoint --toggle --pid "$p"
done