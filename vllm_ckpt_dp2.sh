# ROOT_PID should be the main vllm serve pid
# Example:
ROOT_PID=1885657

get_descendants() {
  local parent="$1"
  local child
  for child in $(pgrep -P "$parent"); do
    echo "$child"
    get_descendants "$child"
  done
}

# Real descendant processes only (no thread IDs)
mapfile -t ALL_PIDS < <(
  {
    echo "$ROOT_PID"
    get_descendants "$ROOT_PID"
  } | sort -nu
)

echo "All descendant process PIDs:"
printf '  %s\n' "${ALL_PIDS[@]}"

# Find which of those processes actually have NVIDIA device files open
mapfile -t GPU_PIDS < <(
  for p in "${ALL_PIDS[@]}"; do
    if sudo find "/proc/$p/fd" -maxdepth 1 -lname '/dev/nvidia*' -print -quit 2>/dev/null | grep -q .; then
      echo "$p"
    fi
  done | sort -nu
)

echo "CUDA/NVIDIA-holding process PIDs:"
printf '  %s\n' "${GPU_PIDS[@]}"

# Optional: show what each one is
for p in "${GPU_PIDS[@]}"; do
  echo "== PID $p : $(ps -p "$p" -o comm= -o args=)"
  sudo find "/proc/$p/fd" -maxdepth 1 -lname '/dev/nvidia*' -ls 2>/dev/null
done

# Suspend CUDA in each real GPU-owning process
for p in "${GPU_PIDS[@]}"; do
  echo "Suspending CUDA for PID $p"
  sudo /home/yy485/cuda-checkpoint/bin/x86_64_Linux/cuda-checkpoint --toggle --pid "$p"
done


# export CKPT_DIR=$PWD/checkpoint_vllm_dp2_new
# mkdir -p "$CKPT_DIR"

# sudo criu dump \
#   --tree "$ROOT_PID" \
#   --images-dir "$CKPT_DIR" \
#   --tcp-established \
#   --ext-unix-sk \
#   --link-remap \
#   -o "$CKPT_DIR/dump.log" \
#   -v4