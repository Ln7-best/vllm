import torch
import torch.distributed as dist
import os
import argparse
from datetime import timedelta

def main():
    # 初始化分布式环境（单机多卡）
    dist.init_process_group(backend='nccl', timeout=timedelta(seconds=60))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)

    print(f"[Rank {rank}] Using GPU {rank}")

    # 创建要发送的张量：每个 rank 发送自己的 rank 编号 * 1.0，形状为 [1000]
    send_tensor = torch.full((1000,), float(rank), dtype=torch.float32, device=device)
    recv_tensor = torch.empty_like(send_tensor)

    # 环形通信：send to next, recv from prev
    next_rank = (rank + 1) % world_size
    prev_rank = (rank - 1 + world_size) % world_size

    # 构建 P2P 操作列表：一个 isend + 一个 irecv
    ops = [
        dist.P2POp(dist.isend, send_tensor, peer=next_rank),
        dist.P2POp(dist.irecv, recv_tensor, peer=prev_rank)
    ]

    # 记录时间（可选）
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()

    # 批量发起异步通信
    work_handles = dist.batch_isend_irecv(ops)

    # 等待所有通信完成
    for work in work_handles:
        work.wait()

    end_event.record()
    torch.cuda.synchronize()
    elapsed_ms = start_event.elapsed_time(end_event)

    # 验证结果：recv_tensor 应该等于 prev_rank 的值
    expected_value = float(prev_rank)
    if torch.allclose(recv_tensor, torch.full_like(recv_tensor, expected_value)):
        print(f"[Rank {rank}] ✅ PASS: received value {expected_value:.1f}, time: {elapsed_ms:.2f} ms")
    else:
        print(f"[Rank {rank}] ❌ FAIL: expected {expected_value}, got {recv_tensor[0].item()}")
        exit(1)

    # 清理
    dist.destroy_process_group()

if __name__ == "__main__":
    # 设置环境变量（如果未设置）
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = 'localhost'
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = '12355'

    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", 0)))
    args = parser.parse_args()

    main()
