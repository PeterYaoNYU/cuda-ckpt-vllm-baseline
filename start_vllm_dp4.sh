export MODEL=Qwen/Qwen3-0.6B
export CUDA_VISIBLE_DEVICES=0,1,2,3
export UV_USE_IO_URING=0

# vLLM-side: use Gloo for DP synchronization
export VLLM_DISABLE_NCCL_FOR_DP_SYNCHRONIZATION=1

# NCCL-side: forbid InfiniBand/RoCE and fall back to sockets
export NCCL_IB_DISABLE=1

# Optional but useful: pin NCCL to a non-IB interface
# replace ensX/ethX with your actual Ethernet interface
# export NCCL_SOCKET_IFNAME=ens5

setsid -f vllm serve "$MODEL" \
  --data-parallel-size 4 \
  --port 8000 \
  --gpu-memory-utilization 0.06 \
  --max-model-len 8000 \
  --disable-log-stats \
  </dev/null > vllm_dp4.log 2>&1