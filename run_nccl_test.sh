#!/bin/bash
# NCCL P2P通信测试启动脚本

set -e

echo "=== NCCL P2P Communication Test ==="

# 默认参数
WORLD_SIZE=${1:-2}  # 默认2个进程
MASTER_ADDR=${2:-localhost}
MASTER_PORT=${3:-29500}

echo "Configuration:"
echo "  World Size: $WORLD_SIZE"
echo "  Master Address: $MASTER_ADDR"
echo "  Master Port: $MASTER_PORT"
echo ""

# 启用NCCL调试（可选）
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# 清理函数
cleanup() {
    echo ""
    echo "Cleaning up background processes..."
    jobs -p | xargs -r kill 2>/dev/null || true
    wait
}
trap cleanup EXIT

echo "Starting $WORLD_SIZE processes..."

# 启动所有进程（除了最后一个在前台运行）
for ((rank=0; rank<WORLD_SIZE-1; rank++)); do
    echo "Starting rank $rank in background..."
    python3 test_nccl_p2p.py \
        --rank $rank \
        --world-size $WORLD_SIZE \
        --master-addr $MASTER_ADDR \
        --master-port $MASTER_PORT &
    
    # 等待一点时间让进程启动
    sleep 1
done

# 最后一个进程在前台运行
final_rank=$((WORLD_SIZE-1))
echo "Starting rank $final_rank in foreground..."
python3 test_nccl_p2p.py \
    --rank $final_rank \
    --world-size $WORLD_SIZE \
    --master-addr $MASTER_ADDR \
    --master-port $MASTER_PORT

echo ""
echo "All processes completed!"
