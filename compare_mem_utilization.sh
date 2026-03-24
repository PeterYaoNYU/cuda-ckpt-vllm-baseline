#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMING_SCRIPT="${TIMING_SCRIPT:-$SCRIPT_DIR/baremetal_vllm_timing_parallel.sh}"
CSV_OUT="${CSV_OUT:-$SCRIPT_DIR/mem_util_compare.csv}"
UTIL_VALUES=(${UTIL_VALUES:-0.1 0.2 0.4 0.8})
BASE_CKPT_DIR="${BASE_CKPT_DIR:-$SCRIPT_DIR/checkpoint_vllm_compare}"
BASE_LOGFILE="${BASE_LOGFILE:-$SCRIPT_DIR/vllm_compare}"
PORT="${PORT:-8003}"

if [[ ! -f "$TIMING_SCRIPT" ]]; then
  echo "Timing script not found: $TIMING_SCRIPT"
  exit 1
fi

sanitize_tag() {
  local value="$1"
  value="${value//./p}"
  printf '%s\n' "$value"
}

parse_summary_csv() {
  local output_file="$1"

  python - "$output_file" <<'PY'
import re
import sys

text = open(sys.argv[1], "r", encoding="utf-8", errors="replace").read()

patterns = {
    "cuda_checkpoint_toggle_ms": r"cuda checkpoint toggle\s+(\d+)\s+ms",
    "criu_image_write_ms": r"criu image write\s+(\d+)\s+ms",
    "criu_restore_ms": r"criu restore\s+(\d+)\s+ms",
    "cuda_toggle_back_ms": r"cuda toggle back\s+(\d+)\s+ms",
    "aggregate_ms": r"aggregate\s+(\d+)\s+ms",
    "checkpoint_image_size_human": r"checkpoint image size\s+([^\s]+)\s+\((\d+)\s+bytes\)",
}

values = {
    "cuda_checkpoint_toggle_ms": "",
    "criu_image_write_ms": "",
    "criu_restore_ms": "",
    "cuda_toggle_back_ms": "",
    "aggregate_ms": "",
    "checkpoint_image_size_human": "",
    "checkpoint_image_size_bytes": "",
}

for key, pattern in patterns.items():
    match = re.search(pattern, text)
    if not match:
        continue
    if key == "checkpoint_image_size_human":
        values["checkpoint_image_size_human"] = match.group(1)
        values["checkpoint_image_size_bytes"] = match.group(2)
    else:
        values[key] = match.group(1)

print(",".join([
    values["cuda_checkpoint_toggle_ms"],
    values["criu_image_write_ms"],
    values["criu_restore_ms"],
    values["cuda_toggle_back_ms"],
    values["aggregate_ms"],
    values["checkpoint_image_size_bytes"],
    values["checkpoint_image_size_human"],
]))
PY
}

printf '%s\n' "memory_utilization,status,cuda_checkpoint_toggle_ms,criu_image_write_ms,criu_restore_ms,cuda_toggle_back_ms,checkpoint_total_ms,aggregate_ms,checkpoint_image_size_bytes,checkpoint_image_size_human" >"$CSV_OUT"

echo "Writing CSV results to $CSV_OUT"

for util in "${UTIL_VALUES[@]}"; do
  tag="$(sanitize_tag "$util")"
  run_ckpt_dir="${BASE_CKPT_DIR}_${tag}"
  run_logfile="${BASE_LOGFILE}_${tag}.log"
  run_output="$(mktemp)"

  echo
  echo "========================================"
  echo "Running GPU memory utilization: $util"
  echo "  log:  $run_logfile"
  echo "  ckpt: $run_ckpt_dir"
  echo "========================================"

  status="success"
  if ! CUDA_VISIBLE_DEVICES="0,1" \
      DP_SIZE="2" \
      GPU_MEMORY_UTILIZATION="$util" \
      PORT="$PORT" \
      CKPT_DIR="$run_ckpt_dir" \
      LOGFILE="$run_logfile" \
      CLEANUP_ON_EXIT="1" \
      bash "$TIMING_SCRIPT" >"$run_output" 2>&1; then
    status="failed"
  fi

  metrics="$(parse_summary_csv "$run_output")"
  IFS=',' read -r cuda_ckpt_ms criu_dump_ms criu_restore_ms cuda_restore_ms aggregate_ms image_size_bytes image_size_human <<<"$metrics"

  checkpoint_total_ms=""
  if [[ -n "$cuda_ckpt_ms" && -n "$criu_dump_ms" ]]; then
    checkpoint_total_ms="$(( cuda_ckpt_ms + criu_dump_ms ))"
  fi

  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$util" \
    "$status" \
    "$cuda_ckpt_ms" \
    "$criu_dump_ms" \
    "$criu_restore_ms" \
    "$cuda_restore_ms" \
    "$checkpoint_total_ms" \
    "$aggregate_ms" \
    "$image_size_bytes" \
    "$image_size_human" >>"$CSV_OUT"

  cat "$run_output"

  sudo rm -rf "$run_ckpt_dir"
  sudo rm -f "$run_output"
done

echo
echo "Done. CSV written to $CSV_OUT"
