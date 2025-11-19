#!/usr/bin/env python3
"""
P2POp诊断脚本 - 检查batch_isend_irecv可能的问题
"""

import torch
import torch.distributed as dist
from torch.distributed import P2POp

def diagnose_p2p_ops(p2p_ops, ep_group, ep_rank):
    """诊断P2POp列表的潜在问题"""
    print(f"[Rank {ep_rank}] === P2POp Diagnosis ===")
    print(f"[Rank {ep_rank}] Total P2P operations: {len(p2p_ops)}")
    
    if not p2p_ops:
        print(f"[Rank {ep_rank}] No P2P operations - this is normal for some ranks")
        return True
    
    # 统计send/recv数量
    send_count = recv_count = 0
    issues = []
    
    for i, op in enumerate(p2p_ops):
        op_name = op.op.__name__ if hasattr(op.op, '__name__') else 'unknown'
        tensor = op.tensor
        peer = op.peer
        
        print(f"[Rank {ep_rank}] P2P[{i}]: {op_name} -> peer {peer}")
        print(f"[Rank {ep_rank}]   Tensor: shape={tensor.shape}, device={tensor.device}, dtype={tensor.dtype}")
        
        # 1. 检查操作类型
        if op_name == 'isend':
            send_count += 1
        elif op_name == 'irecv':
            recv_count += 1
        else:
            issues.append(f"Unknown operation: {op_name}")
        
        # 2. 检查peer rank有效性
        if peer < 0 or peer >= ep_group.size():
            issues.append(f"Invalid peer rank {peer}, group size is {ep_group.size()}")
        
        # 3. 检查tensor设备
        if not tensor.is_cuda:
            issues.append(f"Tensor on wrong device: {tensor.device} (expected CUDA)")
        
        # 4. 检查tensor连续性
        if not tensor.is_contiguous():
            issues.append(f"Tensor not contiguous at P2P[{i}]")
        
        # 5. 检查tensor是否为空
        if tensor.numel() == 0:
            issues.append(f"Empty tensor at P2P[{i}]")
        
        # 6. 检查数据类型
        if tensor.dtype not in [torch.float16, torch.float32, torch.bfloat16]:
            issues.append(f"Unusual dtype: {tensor.dtype} at P2P[{i}]")
    
    print(f"[Rank {ep_rank}] Send operations: {send_count}")
    print(f"[Rank {ep_rank}] Recv operations: {recv_count}")
    
    # 检查send/recv平衡性（可选）
    if send_count > 0 and recv_count > 0:
        print(f"[Rank {ep_rank}] WARNING: Rank has both send and recv operations")
    
    # 报告问题
    if issues:
        print(f"[Rank {ep_rank}] === ISSUES FOUND ===")
        for issue in issues:
            print(f"[Rank {ep_rank}] ❌ {issue}")
        return False
    else:
        print(f"[Rank {ep_rank}] ✅ All P2P operations look valid")
        return True

def test_simple_p2p_before_batch(p2p_ops, ep_group, ep_rank):
    """在batch_isend_irecv前做简单验证"""
    print(f"[Rank {ep_rank}] Testing individual P2P operations...")
    
    for i, op in enumerate(p2p_ops):
        tensor = op.tensor
        peer = op.peer
        op_name = op.op.__name__ if hasattr(op.op, '__name__') else 'unknown'
        
        # 简单的连接性测试
        try:
            # 创建一个小的测试tensor
            test_tensor = torch.ones(1, dtype=torch.float32, device=tensor.device)
            
            if op_name == 'isend':
                # 测试能否向peer发送
                print(f"[Rank {ep_rank}] Test send to peer {peer}...")
            elif op_name == 'irecv':
                # 测试能否从peer接收
                print(f"[Rank {ep_rank}] Test recv from peer {peer}...")
                
        except Exception as e:
            print(f"[Rank {ep_rank}] ❌ P2P[{i}] test failed: {e}")
            return False
    
    print(f"[Rank {ep_rank}] ✅ Individual P2P tests passed")
    return True

# 在shuffle_layer中使用示例：
def enhanced_shuffle_layer_with_diagnosis(p2p_ops, ep_group, ep_rank):
    """带诊断功能的shuffle_layer P2P部分"""
    
    if not p2p_ops:
        print(f"[Rank {ep_rank}] No P2P operations to execute")
        return
    
    # 执行诊断
    diagnosis_ok = diagnose_p2p_ops(p2p_ops, ep_group, ep_rank)
    if not diagnosis_ok:
        raise RuntimeError(f"P2P operations diagnosis failed on rank {ep_rank}")
    
    # 设置CUDA设备
    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        torch.cuda.set_device(current_device)
        torch.cuda.synchronize()
        print(f"[Rank {ep_rank}] CUDA device set to {current_device}")
    
    # 执行batch_isend_irecv
    try:
        print(f"[Rank {ep_rank}] Executing batch_isend_irecv with {len(p2p_ops)} operations...")
        reqs = dist.batch_isend_irecv(p2p_ops)
        print(f"[Rank {ep_rank}] batch_isend_irecv returned {len(reqs)} requests")
        
        # 等待完成
        for i, req in enumerate(reqs):
            req.wait()
            print(f"[Rank {ep_rank}] Request {i+1}/{len(reqs)} completed")
            
        print(f"[Rank {ep_rank}] All P2P operations completed successfully")
        
    except Exception as e:
        print(f"[Rank {ep_rank}] ❌ batch_isend_irecv failed: {e}")
        raise

if __name__ == "__main__":
    print("P2POp诊断脚本 - 在shuffle_layer中集成使用")
    print("主要检查项目：")
    print("1. Tensor设备位置（必须在CUDA上）")
    print("2. Tensor连续性（必须是contiguous）") 
    print("3. Peer rank有效性（0 <= peer < group_size）")
    print("4. 操作类型有效性（isend/irecv）")
    print("5. Tensor形状和数据类型")
