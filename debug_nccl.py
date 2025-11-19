#!/usr/bin/env python3
"""
NCCL后端诊断脚本
检查NCCL配置、版本和基础通信能力
"""

import os
import torch
import torch.distributed as dist

def check_nccl_config():
    print("=== NCCL Configuration Check ===")
    
    # 1. Check NCCL version
    try:
        import torch.version
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"NCCL available: {torch.distributed.is_nccl_available()}")
        if hasattr(torch.cuda.nccl, 'version'):
            print(f"NCCL version: {torch.cuda.nccl.version()}")
    except Exception as e:
        print(f"NCCL version check failed: {e}")
    
    # 2. Check NCCL environment variables
    print("\n=== NCCL Environment Variables ===")
    nccl_vars = [
        'NCCL_DEBUG', 'NCCL_DEBUG_SUBSYS', 'NCCL_SOCKET_IFNAME',
        'NCCL_IB_DISABLE', 'NCCL_P2P_DISABLE', 'NCCL_SHM_DISABLE',
        'NCCL_NET_GDR_LEVEL', 'NCCL_BUFFSIZE', 'NCCL_NTHREADS',
        'NCCL_MIN_NCHANNELS', 'NCCL_MAX_NCHANNELS'
    ]
    
    for var in nccl_vars:
        value = os.environ.get(var, "Not set")
        print(f"{var}: {value}")
    
    # 3. Check GPU devices
    print(f"\n=== GPU Information ===")
    if torch.cuda.is_available():
        print(f"CUDA devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"Current device: {torch.cuda.current_device()}")
    else:
        print("CUDA not available")

def test_simple_nccl():
    """简单的NCCL通信测试"""
    print("\n=== Simple NCCL Test ===")
    
    if not dist.is_initialized():
        print("torch.distributed not initialized, cannot test NCCL")
        return
        
    try:
        # Test basic all_reduce
        tensor = torch.ones(2, device='cuda') * dist.get_rank()
        print(f"Rank {dist.get_rank()}: Before all_reduce: {tensor}")
        
        dist.all_reduce(tensor)
        print(f"Rank {dist.get_rank()}: After all_reduce: {tensor}")
        
        print("NCCL all_reduce test PASSED")
        
    except Exception as e:
        print(f"NCCL all_reduce test FAILED: {e}")

if __name__ == "__main__":
    check_nccl_config()
    test_simple_nccl()
