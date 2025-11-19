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
    if p2p_ops:
        # P2P diagnosis logging
        send_count = recv_count = 0
        for i, op in enumerate(p2p_ops):
            op_name = op.op.__name__ if hasattr(op.op, '__name__') else 'unknown'
            if op_name == 'isend':
                send_count += 1
                logger.info("[shuffle_layer] (rank %d) P2P op[%d]: SEND to rank %d", ep_rank, i, op.peer)
            elif op_name == 'irecv':
                recv_count += 1
                logger.info("[shuffle_layer] (rank %d) P2P op[%d]: RECV from rank %d", ep_rank, i, op.peer)
        
        logger.info("[shuffle_layer] (rank %d) Total: %d sends, %d recvs", ep_rank, send_count, recv_count)
        
        # CRITICAL: Deep diagnosis before batch_isend_irecv
        logger.info("[shuffle_layer] (rank %d) === BATCH_ISEND_IRECV DIAGNOSIS ===", ep_rank)
        logger.info("[shuffle_layer] (rank %d) Process group size: %d, rank: %d", ep_rank, ep_group.size(), ep_group.rank())
        
        # Check tensor info for each operation
        for i, op in enumerate(p2p_ops):
            tensor = op.tensor
            logger.info("[shuffle_layer] (rank %d) P2P[%d]: shape=%s, device=%s, dtype=%s, peer=%d", 
                       ep_rank, i, tensor.shape, tensor.device, tensor.dtype, op.peer)
        
        # CRITICAL: Set current GPU device for NCCL PG backend
        # PyTorch docs: "batch_isend_irecv with NCCL PG backend requires torch.cuda.set_device"
        if torch.cuda.is_available():
            current_device = torch.cuda.current_device()
            torch.cuda.set_device(current_device)
            torch.cuda.synchronize()
            logger.info("[shuffle_layer] (rank %d) GPU device set to %d and synchronized", ep_rank, current_device)
        
        # Check if we're waiting for other processes
        logger.info("[shuffle_layer] (rank %d) About to call batch_isend_irecv with %d operations...", ep_rank, len(p2p_ops))
        
        # Execute with timing and timeout detection
        import time
        import threading
        start_time = time.time()
        logger.info("[shuffle_layer] (rank %d) CALLING batch_isend_irecv at timestamp %.6f", ep_rank, start_time)
        
        # More reliable timeout mechanism using threading.Timer
        timeout_occurred = threading.Event()
        
        def timeout_handler():
            timeout_occurred.set()
            logger.error("[shuffle_layer] (rank %d) batch_isend_irecv TIMEOUT after 10 seconds! Switching to fallback...", ep_rank)
        
        timer = threading.Timer(10.0, timeout_handler)  # 10 second timeout (shorter for faster testing)
        timer.start()
        
        try:
            if timeout_occurred.is_set():
                raise TimeoutError("batch_isend_irecv timed out")
            
            reqs = batch_isend_irecv(p2p_ops)
            timer.cancel()  # Cancel timeout if successful
            
            if timeout_occurred.is_set():
                raise TimeoutError("batch_isend_irecv timed out during execution")
                
        except (TimeoutError, Exception) as e:
            timer.cancel()
            if timeout_occurred.is_set():
                logger.error("[shuffle_layer] (rank %d) batch_isend_irecv TIMEOUT! Switching to synchronous P2P fallback...", ep_rank)
            else:
                logger.error("[shuffle_layer] (rank %d) batch_isend_irecv ERROR: %s. Switching to fallback...", ep_rank, str(e))
            
            # FALLBACK: Use synchronous send/recv instead
            for i, op in enumerate(p2p_ops):
                op_name = op.op.__name__ if hasattr(op.op, '__name__') else 'unknown'
                if op_name == 'isend':
                    logger.info("[shuffle_layer] (rank %d) Fallback SEND[%d] to rank %d", ep_rank, i, op.peer)
                    torch.distributed.send(op.tensor, op.peer, group=ep_group)
                elif op_name == 'irecv':
                    logger.info("[shuffle_layer] (rank %d) Fallback RECV[%d] from rank %d", ep_rank, i, op.peer)
                    torch.distributed.recv(op.tensor, op.peer, group=ep_group)
            
            logger.info("[shuffle_layer] (rank %d) Synchronous P2P fallback completed successfully!", ep_rank)
            reqs = []  # No requests to wait for in sync mode
        except Exception as e:
            signal.alarm(0)
            logger.error("[shuffle_layer] (rank %d) batch_isend_irecv FAILED: %s", ep_rank, str(e))
            raise
        call_time = time.time() - start_time
        logger.info("[shuffle_layer] (rank %d) batch_isend_irecv returned in %.3fs, waiting for %d requests...", 
                   ep_rank, call_time, len(reqs))
        
        for i, req in enumerate(reqs):
            req.wait()
            logger.info("[shuffle_layer] (rank %d) Request %d/%d completed", ep_rank, i+1, len(reqs))
        
        logger.info("[shuffle_layer] (rank %d) All P2P operations completed successfully!", ep_rank)
    else:
        logger.info("[shuffle_layer] (rank %d) No P2P operations needed", ep_rank)


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
