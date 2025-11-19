#!/usr/bin/env python3
"""
简单的NCCL P2P通信测试脚本
测试基础的send/recv和batch_isend_irecv功能
"""

import os
import sys
import time
import argparse
import torch
import torch.distributed as dist
from torch.distributed import P2POp

def setup_distributed(rank, world_size, master_addr="localhost", master_port=29500):
    """初始化分布式环境"""
    print(f"[Rank {rank}] Setting up distributed environment...")
    
    # 设置环境变量
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = str(master_port)
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    
    # 初始化进程组
    print(f"[Rank {rank}] Initializing process group with NCCL backend...")
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{master_addr}:{master_port}",
        rank=rank,
        world_size=world_size,
        timeout=30  # 30秒超时
    )
    
    # 设置CUDA设备 - 关键修复点！
    device = torch.cuda.device(rank)
    torch.cuda.set_device(device)
    print(f"[Rank {rank}] Set CUDA device to {device}")
    
    return device

def cleanup_distributed():
    """清理分布式环境"""
    if dist.is_initialized():
        dist.destroy_process_group()

def test_basic_send_recv(rank, world_size, device):
    """测试基本的同步send/recv通信"""
    print(f"\n[Rank {rank}] === Testing Basic Send/Recv ===")
    
    if world_size < 2:
        print("[Error] Need at least 2 processes for P2P communication")
        return False
        
    tensor_size = (1024, 1024)  # 1M float32 elements
    
    if rank == 0:
        # Rank 0 发送数据给 Rank 1
        send_tensor = torch.ones(tensor_size, device=device, dtype=torch.float32) * 42.0
        print(f"[Rank 0] Sending tensor with value 42.0 to Rank 1...")
        dist.send(send_tensor, dst=1)
        print(f"[Rank 0] Send completed")
        
    elif rank == 1:
        # Rank 1 接收来自 Rank 0 的数据
        recv_tensor = torch.zeros(tensor_size, device=device, dtype=torch.float32)
        print(f"[Rank 1] Receiving tensor from Rank 0...")
        dist.recv(recv_tensor, src=0)
        print(f"[Rank 1] Received tensor with value: {recv_tensor[0, 0].item()}")
        
        if recv_tensor[0, 0].item() == 42.0:
            print(f"[Rank 1] ✅ Basic send/recv test PASSED")
            return True
        else:
            print(f"[Rank 1] ❌ Basic send/recv test FAILED")
            return False
    
    return True

def test_batch_isend_irecv(rank, world_size, device):
    """测试batch_isend_irecv - 模拟vLLM的使用方式"""
    print(f"\n[Rank {rank}] === Testing batch_isend_irecv ===")
    
    if world_size < 2:
        print("[Error] Need at least 2 processes for P2P communication")
        return False
    
    tensor_size = (512, 512)
    p2p_ops = []
    
    # 强制CUDA同步 - 关键修复点！
    torch.cuda.synchronize()
    print(f"[Rank {rank}] CUDA synchronized before batch_isend_irecv")
    
    if rank == 0:
        # Rank 0 发送两个tensor给 Rank 1
        send_tensor1 = torch.ones(tensor_size, device=device, dtype=torch.float32) * 100.0
        send_tensor2 = torch.ones(tensor_size, device=device, dtype=torch.float32) * 200.0
        
        p2p_ops.append(P2POp(torch.distributed.isend, send_tensor1, 1))
        p2p_ops.append(P2POp(torch.distributed.isend, send_tensor2, 1))
        
        print(f"[Rank 0] Created {len(p2p_ops)} send operations")
        
    elif rank == 1:
        # Rank 1 接收两个tensor从 Rank 0
        recv_tensor1 = torch.zeros(tensor_size, device=device, dtype=torch.float32)
        recv_tensor2 = torch.zeros(tensor_size, device=device, dtype=torch.float32)
        
        p2p_ops.append(P2POp(torch.distributed.irecv, recv_tensor1, 0))
        p2p_ops.append(P2POp(torch.distributed.irecv, recv_tensor2, 0))
        
        print(f"[Rank 1] Created {len(p2p_ops)} recv operations")
    
    else:
        # 其他ranks - 关键修复点：所有ranks都必须参与！
        print(f"[Rank {rank}] No P2P operations, but participating in collective call...")
        p2p_ops = []  # 空操作列表
    
    # 所有ranks都调用batch_isend_irecv
    print(f"[Rank {rank}] Calling batch_isend_irecv with {len(p2p_ops)} operations...")
    
    try:
        start_time = time.time()
        reqs = dist.batch_isend_irecv(p2p_ops)
        call_time = time.time() - start_time
        
        print(f"[Rank {rank}] batch_isend_irecv returned in {call_time:.3f}s with {len(reqs)} requests")
        
        # 等待所有请求完成
        for i, req in enumerate(reqs):
            req.wait()
            print(f"[Rank {rank}] Request {i+1}/{len(reqs)} completed")
            
        print(f"[Rank {rank}] All P2P operations completed!")
        
        # 验证结果
        if rank == 1:
            val1 = recv_tensor1[0, 0].item()
            val2 = recv_tensor2[0, 0].item()
            print(f"[Rank 1] Received values: {val1}, {val2}")
            
            if val1 == 100.0 and val2 == 200.0:
                print(f"[Rank 1] ✅ batch_isend_irecv test PASSED")
                return True
            else:
                print(f"[Rank 1] ❌ batch_isend_irecv test FAILED")
                return False
        
        return True
        
    except Exception as e:
        print(f"[Rank {rank}] ❌ batch_isend_irecv FAILED with error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Test NCCL P2P communication")
    parser.add_argument("--rank", type=int, required=True, help="Process rank")
    parser.add_argument("--world-size", type=int, required=True, help="Total number of processes")
    parser.add_argument("--master-addr", type=str, default="localhost", help="Master address")
    parser.add_argument("--master-port", type=int, default=29500, help="Master port")
    
    args = parser.parse_args()
    
    print(f"Starting NCCL P2P test: Rank {args.rank}/{args.world_size}")
    
    try:
        # 设置分布式环境
        device = setup_distributed(args.rank, args.world_size, args.master_addr, args.master_port)
        
        # 等待所有进程就绪
        print(f"[Rank {args.rank}] Waiting for all processes to be ready...")
        dist.barrier()
        
        # 测试基本send/recv
        success1 = test_basic_send_recv(args.rank, args.world_size, device)
        
        # 同步后测试batch_isend_irecv
        dist.barrier()
        success2 = test_batch_isend_irecv(args.rank, args.world_size, device)
        
        # 最终同步
        dist.barrier()
        
        if args.rank == 0:
            print(f"\n=== Test Summary ===")
            print(f"Basic send/recv: {'✅ PASSED' if success1 else '❌ FAILED'}")
            print(f"batch_isend_irecv: {'✅ PASSED' if success2 else '❌ FAILED'}")
        
    except Exception as e:
        print(f"[Rank {args.rank}] Test failed with error: {e}")
        sys.exit(1)
        
    finally:
        cleanup_distributed()
        print(f"[Rank {args.rank}] Test completed")

if __name__ == "__main__":
    main()
