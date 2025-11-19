# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
The actual execution of the rearrangement.

This involves the exchange of expert weights between GPUs.
"""

from collections.abc import Iterable, MutableSequence, Sequence
from functools import partial

import torch
from torch.distributed import (
    P2POp,
    ProcessGroup,
    all_gather,
    batch_isend_irecv,
    get_global_rank,
)

from vllm.logger import init_logger

logger = init_logger(__name__)


def idx_local_to_global(
    local_idx: int,
    local_cnt: int,
    ep_rank: int,
) -> int:
    """
    Convert a local expert index to a global expert index.
    """
    return ep_rank * local_cnt + local_idx


def idx_global_to_local(
    global_idx: int,
    local_cnt: int,
    ep_rank: int,
) -> int:
    """
    Convert a global expert index to a local expert index.
    """
    return global_idx - ep_rank * local_cnt


def global_idx_to_rank(
    global_idx: int,
    local_cnt: int,
) -> int:
    """
    Convert a global expert index to a rank index.
    """
    return global_idx // local_cnt


def get_ep_ranks_with_expert(
    idx: int,
    num_local_experts: int,
    old_indices: Sequence[int],
    new_indices: Sequence[int],
) -> tuple[MutableSequence[int], MutableSequence[int]]:
    """
    Get the ranks of the experts that need to be exchanged.

    Args:
        idx: The index of the expert.
        num_local_experts: The number of local experts.
        old_indices: The old indices of the experts.
        new_indices: The new indices of the experts.

    Returns:
        A tuple of two lists:
        - The ranks of the experts that need to be sent.
        - The ranks of the experts that need to be received.
    """
    global2rank = partial(
        global_idx_to_rank,
        local_cnt=num_local_experts,
    )

    ranks_to_send: list[int] = []
    ranks_to_recv: list[int] = []

    for i, e in enumerate(old_indices):
        if e == idx:
            rank = global2rank(i)
            if not ranks_to_send or ranks_to_send[-1] != rank:
                ranks_to_send.append(rank)

    for i, e in enumerate(new_indices):
        if e == idx:
            rank = global2rank(i)
            if not ranks_to_recv or ranks_to_recv[-1] != rank:
                ranks_to_recv.append(rank)

    # Remove those ranks that can get this expert locally.
    ranks_to_send_set = set(ranks_to_send)
    ranks_to_recv_actual = [
        rank for rank in ranks_to_recv if rank not in ranks_to_send_set
    ]

    return ranks_to_send, ranks_to_recv_actual


def shuffle_layer(
    num_local_experts: int,
    ep_rank: int,
    old_indices: Sequence[int],
    new_indices: Sequence[int],
    expert_weights: Iterable[torch.Tensor],
    expert_weights_buffer: Sequence[torch.Tensor],
    ep_group: ProcessGroup,
) -> None:
    """
    Perform expert weights rearrangement of one layer.
    """
    local2global = partial(
        idx_local_to_global,
        local_cnt=num_local_experts,
        ep_rank=ep_rank,
    )

    # 0. Do nothing for experts that did not change.
    is_unchanged = [
        old_indices[local2global(i)] == new_indices[local2global(i)]
        for i in range(num_local_experts)
    ]

    # 1. Perform weight copy inside the local rank.
    is_received_locally = is_unchanged[:]
    for src in range(num_local_experts):
        src_global = local2global(src)
        for dst in range(num_local_experts):
            dst_global = local2global(dst)
            if is_received_locally[dst]:
                continue
            if old_indices[src_global] == -1 or new_indices[dst_global] == -1:
                continue
            if old_indices[src_global] == new_indices[dst_global]:
                is_received_locally[dst] = True
                for weight, buffer in zip(expert_weights, expert_weights_buffer):
                    buffer[dst].copy_(weight[src])

    p2p_ops: list[P2POp] = []

    # 2. Initiate sending of weights.
    experts_send_loc: dict[int, int] = {}
    for src in range(num_local_experts):
        expert = old_indices[local2global(src)]
        if expert == -1:
            continue
        if expert in experts_send_loc:
            continue
        experts_send_loc[expert] = src

    # We need to sort here to match send/recv
    for expert, src in sorted(experts_send_loc.items()):
        ranks_to_send, ranks_to_recv = get_ep_ranks_with_expert(
            expert,
            num_local_experts,
            old_indices,
            new_indices,
        )

        # Calculate the ranks to send by this rank
        num_dst_per_sender = len(ranks_to_recv) // len(ranks_to_send)
        sender_pos = ranks_to_send.index(ep_rank)
        recv_begin = sender_pos * num_dst_per_sender
        recv_end = recv_begin + num_dst_per_sender
        recv_ranks = ranks_to_recv[recv_begin:recv_end]

        # Tackle remainders
        remainder_start = len(ranks_to_send) * num_dst_per_sender
        recver_pos = remainder_start + sender_pos
        if recver_pos < len(ranks_to_recv):
            recv_ranks.append(ranks_to_recv[recver_pos])

        for dst in recv_ranks:
            dst_global = get_global_rank(ep_group, dst)
            p2p_ops += [
                P2POp(
                    torch.distributed.isend,
                    weight[src],
                    dst_global,
                )
                for weight in expert_weights
            ]

    # 3. Initiate receiving of weights.
    experts_recv_loc: dict[int, int] = {}
    for dst in range(num_local_experts):
        if is_received_locally[dst]:
            continue
        expert = new_indices[local2global(dst)]
        if expert == -1:
            continue
        if expert in experts_recv_loc:
            continue
        experts_recv_loc[expert] = dst

    # We need to sort here to match send/recv
    for expert, dst in sorted(experts_recv_loc.items()):
        ranks_to_send, ranks_to_recv = get_ep_ranks_with_expert(
            expert,
            num_local_experts,
            old_indices,
            new_indices,
        )

        # Calculate the rank to recv by this rank
        num_dst_per_sender = len(ranks_to_recv) // len(ranks_to_send)
        recver_pos = ranks_to_recv.index(ep_rank)
        remainder_start = len(ranks_to_send) * num_dst_per_sender
        if recver_pos < remainder_start:
            src = ranks_to_send[recver_pos // num_dst_per_sender]
        else:
            src = ranks_to_send[recver_pos - remainder_start]

        src_global = get_global_rank(ep_group, src)
        p2p_ops += [
            P2POp(
                torch.distributed.irecv,
                weight[dst],
                src_global,
            )
            for weight in expert_weights_buffer
        ]

    # 4. Execute the P2P operations. The real communication happens here.
    
    # # === SIMPLE P2P TEST ===
    # dummy_p2p_ops = []
    
    # if ep_group.size() >= 2 and ep_rank == 0 or ep_rank == 4:
    #     device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu')
        
    #     if ep_rank == 0:
    #         # Rank 0 sends to rank 1
    #         tensor = torch.ones(10, dtype=torch.float32, device=device) * 123.0
    #         dummy_p2p_ops.append(P2POp(torch.distributed.isend, tensor, 4))
    #     elif ep_rank == 4:
    #         # Rank 1 receives from rank 0
    #         tensor = torch.zeros(10, dtype=torch.float32, device=device)
    #         dummy_p2p_ops.append(P2POp(torch.distributed.irecv, tensor,  0))
    
    # === DUMMY P2P TEST: rank 0 and 1 only ===
    global_rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    logger.info(f"[Global Rank {global_rank}]")
    
    # Fix: Use ep_rank directly as device ID to handle Ray GPU allocation
    assert torch.cuda.is_available(), "CUDA must be available for P2P operations"
    
    # In Ray distributed environment, each Actor may only see 1 GPU
    # Use ep_rank directly as the device ID since Ray handles GPU allocation
    num_visible_gpus = torch.cuda.device_count()
    
    # Try using ep_rank as device ID first (common in distributed setups)
    import os
    logger.info(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    try:
        torch.cuda.set_device(ep_rank)
        current_device = torch.cuda.current_device()
        device = torch.device(f'cuda:{current_device}')
        device_strategy = "ep_rank_direct" 
    except RuntimeError:
        # Fallback: use modulo if ep_rank exceeds visible GPU count
        local_device_id = ep_rank % max(1, num_visible_gpus)
        torch.cuda.set_device(local_device_id)
        current_device = torch.cuda.current_device()
        device = torch.device(f'cuda:{current_device}')
        device_strategy = "modulo_fallback"
    
    logger.info(f"[Rank {ep_rank}] Device allocation: {device} (global_rank: {global_rank}, ep_rank: {ep_rank}, current_device: {current_device}, visible_gpus: {num_visible_gpus}, strategy: {device_strategy})")

    dummy_p2p_ops = []
    tensors_to_hold = []  # 用于延长 tensor 生命周期

    # Ring communication: 所有设备都参与 (强制执行)
    # 计算Ring通信的邻居节点
    next_rank = (ep_rank + 1) % ep_group.size()  # 下一个节点
    prev_rank = (ep_rank - 1) % ep_group.size()  # 上一个节点
    
    # 获取全局rank
    next_global_rank = get_global_rank(ep_group, next_rank)
    prev_global_rank = get_global_rank(ep_group, prev_rank)
    
    # 每个rank都发送和接收
    # 发送给下一个rank
    send_tensor = torch.ones(10, dtype=torch.float32, device=device) * (ep_rank * 100 + 42)
    dummy_p2p_ops.append(P2POp(torch.distributed.isend, send_tensor, peer=next_global_rank))
    tensors_to_hold.append(send_tensor)
    
    # 从上一个rank接收
    recv_tensor = torch.zeros(10, dtype=torch.float32, device=device)
    dummy_p2p_ops.append(P2POp(torch.distributed.irecv, recv_tensor, peer=prev_global_rank))
    tensors_to_hold.append(recv_tensor)
    
    logger.info(f"[Rank {ep_rank}] Ring communication: sending to rank {next_rank}(global:{next_global_rank}), receiving from rank {prev_rank}(global:{prev_global_rank})")

    # 执行批量通信 (强制执行)
    logger.info(f"[Rank {ep_rank}] Launching Ring P2P with {len(dummy_p2p_ops)} operations...")
    work_handles = batch_isend_irecv(dummy_p2p_ops)
    
    # 等待所有通信完成
    for work in work_handles:
        work.wait()
    
    # 验证Ring通信结果：检查接收到的数据
    if len(tensors_to_hold) >= 2:  # 确保有接收tensor
        recv_tensor = tensors_to_hold[1]  # 第二个tensor是接收的
        expected_value = prev_rank * 100 + 42  # 从上一个rank期望收到的值
        expected_tensor = torch.full_like(recv_tensor, expected_value)
        
        if torch.allclose(recv_tensor, expected_tensor):
            logger.info(f"[Rank {ep_rank}] ✅ Ring P2P success: received {recv_tensor[0].item()} from rank {prev_rank}")
        else:
            logger.error(f"[Rank {ep_rank}] ❌ Ring P2P failed: expected {expected_value}, got {recv_tensor[0].item()}")
            raise RuntimeError("Ring P2P communication test failed!")

    logger.info(f"[Rank {ep_rank}] ✅ Ring P2P test completed successfully for all {ep_group.size()} ranks.")
    
    # Skip real operations
    # if p2p_ops:
    #     reqs = batch_isend_irecv(p2p_ops)
    #     for req in reqs:
    #         req.wait()
    
    # Skip real operations for now
    # if p2p_ops:
    #     reqs = batch_isend_irecv(p2p_ops)
    #     for req in reqs:
    #         req.wait()

    # 5. Copy the weights from the buffer back to the original weights.
    for dst in range(num_local_experts):
        if is_unchanged[dst]:
            continue
        if is_received_locally[dst]:
            for weight, buffer in zip(expert_weights, expert_weights_buffer):
                weight[dst].copy_(buffer[dst])
        else:
            expert = new_indices[local2global(dst)]
            if expert == -1:
                continue
            src = experts_recv_loc[expert]
            for weight, buffer in zip(expert_weights, expert_weights_buffer):
                weight[dst].copy_(buffer[src])



def rearrange_expert_weights_inplace(
    old_global_expert_indices: torch.Tensor,
    new_global_expert_indices: torch.Tensor,
    expert_weights: Sequence[Iterable[torch.Tensor]],
    ep_group: ProcessGroup,
    is_profile: bool = False,
    rank_mapping: dict[int, int] | None = None,
) -> None:
    """
    Rearranges the expert weights in place according to the new expert indices.

    The value of the indices arguments are logical indices of the experts,
    while keys are physical.

    Args:
        old_global_expert_indices: Shape (num_moe_layers, num_physical_experts).
        new_global_expert_indices: Shape (num_moe_layers, num_physical_experts).
        expert_weights: A sequence of shape (num_moe_layers)(weight_count)
            of tensors of shape (num_local_physical_experts, hidden_size_i).
            For example, a linear layer may have up and down projection,
            so weight_count = 2. Each weight's hidden size can be different.
        ep_group: The device process group for expert parallelism.
        is_profile (bool): If `True`, do not perform any actual weight copy.
            This is used during profile run, where we only perform dummy
            communications to reserve enough memory for the buffers.
        rank_mapping: A dictionary mapping old rank to new rank.
    """
    import logging
    logger = logging.getLogger(__name__)
    
    ep_rank = ep_group.rank()
    
    # ===== UPSTREAM SOURCE TRACKING: Record old_global_expert_indices origin =====
    # logger.error("[Expert Weights] (rank %d) === old_global_expert_indices UPSTREAM TRACKING ===", ep_rank)
    # logger.error("[Expert Weights] (rank %d) old_global_expert_indices received as parameter:", ep_rank)
    # logger.error("[Expert Weights] (rank %d) - Type: %s", ep_rank, type(old_global_expert_indices))
    # logger.error("[Expert Weights] (rank %d) - Shape: %s", ep_rank, old_global_expert_indices.shape)
    # logger.error("[Expert Weights] (rank %d) - Device: %s", ep_rank, old_global_expert_indices.device)
    # logger.error("[Expert Weights] (rank %d) - ID: %s", ep_rank, id(old_global_expert_indices))
    # logger.error("[Expert Weights] (rank %d) - Full content: %s", ep_rank, old_global_expert_indices.tolist())
    # logger.error("[Expert Weights] (rank %d) - All unique values: %s", ep_rank, torch.unique(old_global_expert_indices).tolist())
    # logger.error("[Expert Weights] (rank %d) === END UPSTREAM TRACKING ===", ep_rank)
    
    logger.info("[Expert Weights] (rank %d) Starting rearrange_expert_weights_inplace...", ep_rank)
    
    if rank_mapping is not None:
        logger.info("[Expert Weights] (rank %d) Processing rank_mapping: %s", ep_rank, rank_mapping)
        if len(rank_mapping) == ep_group.size():
            # scale down
            logger.info("[Expert Weights] (rank %d) Scale down: mapping new expert indices", ep_rank)
            new_global_expert_indices = _map_new_expert_indices_with_rank_mapping(
                new_global_expert_indices,
                rank_mapping,
            )
        else:
            # scale up
            logger.info("[Expert Weights] (rank %d) Scale up: mapping old expert indices", ep_rank)
            old_global_expert_indices = _map_old_expert_indices_with_rank_mapping(
                old_global_expert_indices,
                rank_mapping,
                ep_group.size(),
            )
        logger.info("[Expert Weights] (rank %d) Rank mapping completed", ep_rank)

    assert old_global_expert_indices.shape[1] == new_global_expert_indices.shape[1]

    num_moe_layers, num_physical_experts = old_global_expert_indices.shape
    assert len(expert_weights) == num_moe_layers

    num_local_physical_experts = next(iter(expert_weights[0])).shape[0]
    assert new_global_expert_indices.shape == (num_moe_layers, num_physical_experts)

    ep_size = ep_group.size()
    assert num_physical_experts == ep_size * num_local_physical_experts
    
    logger.info("[Expert Weights] (rank %d) Parameters: num_moe_layers=%d, num_physical_experts=%d, num_local_physical_experts=%d", 
               ep_rank, num_moe_layers, num_physical_experts, num_local_physical_experts)

    # A buffer to hold the expert weights in one layer during the exchange.
    # NOTE: Currently we assume the same weights across different layers
    # have the same shape.
    logger.info("[Expert Weights] (rank %d) Creating expert weights buffer...", ep_rank)
    expert_weights_buffer = [torch.empty_like(w) for w in expert_weights[0]]

    if is_profile:
        logger.info("[Expert Weights] (rank %d) Profile mode: performing dummy communications", ep_rank)
        # Maximum send size is to send all local experts to all ranks,
        # So we use a dummy `all_gather` to reserve enough communication buffer
        for weight, buffer in zip(expert_weights[0], expert_weights_buffer):
            # A `/dev/null`-like buffer to avoid real memory allocation
            dummy_recv_buffer = [buffer for _ in range(ep_size)]
            # NOTE(bowen): Needed this barrier to avoid OOM during actual
            # execution. I'm not very sure why this is needed
            torch.distributed.barrier()
            all_gather(
                dummy_recv_buffer,
                weight,
                group=ep_group,
            )
        logger.info("[Expert Weights] (rank %d) Profile mode completed", ep_rank)
        return

    logger.info("[Expert Weights] (rank %d) Converting indices to CPU...", ep_rank)
    old_global_expert_indices_cpu = old_global_expert_indices.cpu()
    new_global_expert_indices_cpu = new_global_expert_indices.cpu()

    # NOTE(bowen): We need this synchronize to run, but I don't know why.
    # If you figure out the reason, please let me know -- thank you!
    logger.info("[Expert Weights] (rank %d) Calling torch.cuda.synchronize()...", ep_rank)
    torch.cuda.synchronize()
    logger.info("[Expert Weights] (rank %d) torch.cuda.synchronize() completed", ep_rank)

    logger.info("[Expert Weights] (rank %d) Starting layer-by-layer shuffling (%d layers)...", ep_rank, num_moe_layers)
    for layer in range(num_moe_layers):
        logger.info("[Expert Weights] (rank %d) Shuffling layer %d/%d...", ep_rank, layer + 1, num_moe_layers)
        shuffle_layer(
            num_local_physical_experts,
            ep_rank,
            old_global_expert_indices_cpu[layer].tolist(),
            new_global_expert_indices_cpu[layer].tolist(),
            expert_weights[layer],
            expert_weights_buffer,
            ep_group,
        )
        logger.info("[Expert Weights] (rank %d) Layer %d/%d shuffling completed", ep_rank, layer + 1, num_moe_layers)
    
    logger.info("[Expert Weights] (rank %d) rearrange_expert_weights_inplace completed successfully!", ep_rank)


def _map_old_expert_indices_with_rank_mapping(
    old_global_expert_indices: torch.Tensor,
    rank_mapping: dict[int, int],
    new_ep_size: int,
) -> torch.Tensor:
    """
    Map the old global expert indices to the new global expert indices.

    Args:
        old_global_expert_indices:
            Shape (num_layers, old_ep_size * num_local_physical_experts).
        rank_mapping: Mapping from old rank to new rank.
        new_ep_size: New expert parallelism size.

    Returns:
        Mapped expert indices with shape
        (num_layers, new_ep_size * num_local_physical_experts).
    """
    num_layers, old_num_physical_experts = old_global_expert_indices.shape
    assert rank_mapping, "Rank mapping is required"

    # Get sizes from parameters and rank_mapping
    old_ep_size = len(rank_mapping)
    num_local_physical_experts = old_num_physical_experts // old_ep_size
    new_num_physical_experts = new_ep_size * num_local_physical_experts

    # Create mapped tensor with new shape, initialized to -1
    mapped_expert_indices = torch.full(
        (num_layers, new_num_physical_experts),
        fill_value=-1,
        dtype=old_global_expert_indices.dtype,
        device=old_global_expert_indices.device,
    )
    
    logger.info(
        "[_map_old_expert_indices] Scale transformation: old_ep_size=%d -> new_ep_size=%d", 
        old_ep_size, new_ep_size
    )
    logger.info(
        "[_map_old_expert_indices] Tensor sizes: (%d,%d) -> (%d,%d)", 
        num_layers, old_num_physical_experts, num_layers, new_num_physical_experts
    )
    logger.info("[_map_old_expert_indices] rank_mapping: %s", rank_mapping)
    logger.info(
        "[_map_old_expert_indices] Created mapped_expert_indices filled with -1, shape: %s", 
        mapped_expert_indices.shape
    )

    # Handle rank mapping (scale up/down with rank changes)
    mapped_ranks = []
    for old_rank in range(old_ep_size):
        new_rank = rank_mapping.get(old_rank)
        if new_rank is not None and new_rank >= 0 and new_rank < new_ep_size:
            # This old rank exists in the new configuration
            old_start_idx = old_rank * num_local_physical_experts
            old_end_idx = (old_rank + 1) * num_local_physical_experts
            new_start_idx = new_rank * num_local_physical_experts
            new_end_idx = (new_rank + 1) * num_local_physical_experts

            mapped_expert_indices[:, new_start_idx:new_end_idx] = (
                old_global_expert_indices[:, old_start_idx:old_end_idx]
            )
            mapped_ranks.append(f"old_rank_{old_rank}->new_rank_{new_rank}")
        # If new_rank is None or >= new_ep_size, the experts remain -1
        # (scale down case)
        else:
            logger.info("[_map_old_expert_indices] old_rank %d -> unmapped (stays -1)", old_rank)
    
    logger.info("[_map_old_expert_indices] Mapped ranks: %s", mapped_ranks)
    
    # Log the final result to see -1 distribution
    sample_layer = mapped_expert_indices[0] if num_layers > 0 else []
    neg_ones_count = (sample_layer == -1).sum().item() if len(sample_layer) > 0 else 0
    logger.info(
        "[_map_old_expert_indices] Layer 0 sample: -1 count=%d/%d, first 20 values: %s", 
        neg_ones_count, len(sample_layer), sample_layer[:20].tolist() if len(sample_layer) > 0 else []
    )

    # CRITICAL: Simple confirmation of -1 filling
    sample_layer = mapped_expert_indices[0] if mapped_expert_indices.shape[0] > 0 else []
    neg_ones_total = (sample_layer == -1).sum().item() if len(sample_layer) > 0 else 0
    logger.info("[_map_old_expert_indices] RETURN: shape=%s, -1_count=%d, last_10=%s", 
               mapped_expert_indices.shape, neg_ones_total, 
               sample_layer[-10:].tolist() if len(sample_layer) >= 10 else [])

    return mapped_expert_indices


def _map_new_expert_indices_with_rank_mapping(
    new_global_expert_indices: torch.Tensor,
    rank_mapping: dict[int, int],
) -> torch.Tensor:
    num_layers, new_num_physical_experts = new_global_expert_indices.shape
    assert rank_mapping, "Rank mapping is required"

    # Get sizes from parameters and rank_mapping
    old_ep_size = len(rank_mapping)
    new_ep_size = sum(new_rank != -1 for new_rank in rank_mapping.values())
    num_local_physical_experts = new_num_physical_experts // new_ep_size
    old_num_physical_experts = old_ep_size * num_local_physical_experts

    mapped_expert_indices = torch.full(
        (num_layers, old_num_physical_experts),
        fill_value=-1,
        dtype=new_global_expert_indices.dtype,
        device=new_global_expert_indices.device,
    )

    for old_rank in range(old_ep_size):
        new_rank = rank_mapping[old_rank]
        if new_rank >= 0 and new_rank < new_ep_size:
            old_start_idx = old_rank * num_local_physical_experts
            old_end_idx = (old_rank + 1) * num_local_physical_experts
            new_start_idx = new_rank * num_local_physical_experts
            new_end_idx = (new_rank + 1) * num_local_physical_experts

            mapped_expert_indices[:, old_start_idx:old_end_idx] = (
                new_global_expert_indices[:, new_start_idx:new_end_idx]
            )

    return mapped_expert_indices


__all__ = ["rearrange_expert_weights_inplace"]
