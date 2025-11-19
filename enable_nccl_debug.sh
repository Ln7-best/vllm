#!/bin/bash
# 启用NCCL详细调试日志

echo "Setting NCCL debug environment variables..."

# 启用NCCL详细日志
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# 禁用一些可能有问题的优化
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SHM_DISABLE=1

# 设置安全的网络接口（如果有多网卡问题）
# export NCCL_SOCKET_IFNAME=eth0  # 替换为您的网络接口

# 设置超时
export NCCL_ASYNC_ERROR_HANDLING=1

echo "NCCL debug environment set. Now run your vLLM command:"
echo "Example: python your_vllm_script.py"
echo ""
echo "Look for NCCL logs like:"
echo "  - NCCL INFO: Bootstrap: Using [xxx]"
echo "  - NCCL INFO: Channel xx/xx created"
echo "  - Any NCCL ERROR or WARNING messages"
