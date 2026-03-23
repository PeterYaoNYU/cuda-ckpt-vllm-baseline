ROOT_PID=2765010

export CKPT_DIR=$PWD/checkpoint_vllm_dp4_18
mkdir -p "$CKPT_DIR"

sudo criu dump \
  --tree "$ROOT_PID" \
  --images-dir "$CKPT_DIR" \
  --tcp-established \
  --ext-unix-sk \
  --link-remap \
  -o "$CKPT_DIR/dump.log" \
  -v4