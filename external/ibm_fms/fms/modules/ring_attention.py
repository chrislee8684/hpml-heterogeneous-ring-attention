from typing import Tuple, Optional
from typing_extensions import Unpack

import torch
import torch.distributed as dist
from torch import Tensor
from torch.nn import functional as F

from fms.modules.attention import (
    AttentionKwargs,
    MultiHeadAttention,
    register_attention_op,
    _sdpa_store_op,
)


class RingAttentionKwargs(AttentionKwargs):
    """
    Kwargs for ring attention
    """

    ring_size: int


import math

def ring_attention_prefill(
    query: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    nheads: int,
    kvheads: int,
    p_dropout: float,
    scale_factor: Optional[float],
    **attn_kwargs: Unpack[RingAttentionKwargs],
):
    """
    Ring attention implementation for prefill.
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    queries = query.transpose(1, 2)
    keys = key_cache
    values = value_cache

    if nheads != kvheads:
        expansion = nheads // kvheads
        keys = keys.unsqueeze(2).expand(-1, -1, expansion, -1, -1).flatten(1, 2)
        values = values.unsqueeze(2).expand(-1, -1, expansion, -1, -1).flatten(1, 2)

    # initialize max_score and accumulator
    max_score = torch.full((queries.shape[0], queries.shape[1], queries.shape[2], 1), -float('inf'), device=queries.device, dtype=queries.dtype)
    accumulator = torch.zeros_like(queries)
    global_denominator = torch.zeros((queries.shape[0], queries.shape[1], queries.shape[2], 1), device=queries.device, dtype=queries.dtype)

    send_rank = (rank + 1) % world_size
    recv_rank = (rank - 1 + world_size) % world_size

    for i in range(world_size):
        block_idx = (rank - i + world_size) % world_size
        
        q_len = queries.shape[2]
        k_len = keys.shape[2]
        
        if rank == block_idx:
            mask = torch.triu(torch.ones(q_len, k_len, device=queries.device), diagonal=1).bool()
        elif rank > block_idx:
            mask = torch.zeros(q_len, k_len, device=queries.device).bool()
        else:
            mask = torch.ones(q_len, k_len, device=queries.device).bool()

        scale = scale_factor if scale_factor is not None else 1.0 / math.sqrt(queries.shape[-1])
        scores = (queries @ keys.transpose(-2, -1)) * scale
        
        scores.masked_fill_(mask, -float('inf'))

        current_max_score = torch.max(scores, dim=-1, keepdim=True)[0]
        new_max_score = torch.max(max_score, current_max_score)
        
        exp_scores = torch.exp(scores - new_max_score)
        
        accumulator = accumulator * torch.exp(max_score - new_max_score)
        accumulator += exp_scores @ values
        
        global_denominator = global_denominator * torch.exp(max_score - new_max_score) + exp_scores.sum(dim=-1, keepdim=True)
        
        max_score = new_max_score

        if i < world_size - 1:
            send_k = keys.contiguous()
            send_v = values.contiguous()
            recv_k = torch.empty_like(send_k)
            recv_v = torch.empty_like(send_v)
            
            ops = [
                dist.P2POp(dist.isend, send_k, send_rank),
                dist.P2POp(dist.isend, send_v, send_rank),
                dist.P2POp(dist.irecv, recv_k, recv_rank),
                dist.P2POp(dist.irecv, recv_v, recv_rank),
            ]

            reqs = dist.batch_isend_irecv(ops)
            for req in reqs:
                req.wait()
            
            keys = recv_k
            values = recv_v
    
    local_attn_output = accumulator / global_denominator
    local_attn_output = local_attn_output.transpose(1, 2).contiguous()
    return local_attn_output


def ring_attention_prefill_pipelined(
    query: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    nheads: int,
    kvheads: int,
    p_dropout: float,
    scale_factor: Optional[float],
    **attn_kwargs: Unpack[RingAttentionKwargs],
):
    """
    Ring attention implementation for prefill with software pipelining.
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    queries = query.transpose(1, 2)
    keys = key_cache
    values = value_cache

    if nheads != kvheads:
        expansion = nheads // kvheads
        keys = keys.unsqueeze(2).expand(-1, -1, expansion, -1, -1).flatten(1, 2)
        values = values.unsqueeze(2).expand(-1, -1, expansion, -1, -1).flatten(1, 2)

    max_score = torch.full((queries.shape[0], queries.shape[1], queries.shape[2], 1), -float('inf'), device=queries.device, dtype=queries.dtype)
    accumulator = torch.zeros_like(queries)
    global_denominator = torch.zeros((queries.shape[0], queries.shape[1], queries.shape[2], 1), device=queries.device, dtype=queries.dtype)

    send_rank = (rank + 1) % world_size
    recv_rank = (rank - 1 + world_size) % world_size

    reqs = None
    
    for i in range(world_size):
        # In each iteration, we compute on the current keys/values
        # and pre-fetch the next keys/values.
        
        # Start the communication for the *next* iteration (if not the last)
        if i < world_size - 1:
            send_k = keys.contiguous()
            send_v = values.contiguous()
            recv_k_next = torch.empty_like(send_k)
            recv_v_next = torch.empty_like(send_v)
            
            ops = [
                dist.P2POp(dist.isend, send_k, send_rank),
                dist.P2POp(dist.isend, send_v, send_rank),
                dist.P2POp(dist.irecv, recv_k_next, recv_rank),
                dist.P2POp(dist.irecv, recv_v_next, recv_rank),
            ]
            reqs = dist.batch_isend_irecv(ops)

        # Computation for the *current* iteration
        block_idx = (rank - i + world_size) % world_size
        
        q_len = queries.shape[2]
        k_len = keys.shape[2]
        
        if rank == block_idx:
            mask = torch.triu(torch.ones(q_len, k_len, device=queries.device), diagonal=1).bool()
        elif rank > block_idx:
            mask = torch.zeros(q_len, k_len, device=queries.device).bool()
        else:
            mask = torch.ones(q_len, k_len, device=queries.device).bool()

        scale = scale_factor if scale_factor is not None else 1.0 / math.sqrt(queries.shape[-1])
        scores = (queries @ keys.transpose(-2, -1)) * scale
        
        scores.masked_fill_(mask, -float('inf'))

        current_max_score = torch.max(scores, dim=-1, keepdim=True)[0]
        new_max_score = torch.max(max_score, current_max_score)
        
        exp_scores = torch.exp(scores - new_max_score)
        
        accumulator = accumulator * torch.exp(max_score - new_max_score)
        accumulator += exp_scores @ values
        
        global_denominator = global_denominator * torch.exp(max_score - new_max_score) + exp_scores.sum(dim=-1, keepdim=True)
        
        max_score = new_max_score
        
        # Wait for the communication to complete to get the keys/values for the next iteration
        if reqs:
            for req in reqs:
                req.wait()
            keys = recv_k_next
            values = recv_v_next
            reqs = None
            
    local_attn_output = accumulator / global_denominator
    local_attn_output = local_attn_output.transpose(1, 2).contiguous()
    return local_attn_output


def ring_attention_prefill_blocking(
    query: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    nheads: int,
    kvheads: int,
    p_dropout: float,
    scale_factor: Optional[float],
    **attn_kwargs: Unpack[RingAttentionKwargs],
):
    """
    Ring attention implementation for prefill with explicit blocking send/recv.
    This is primarily for debugging distributed hangs.
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    queries = query.transpose(1, 2)
    keys = key_cache
    values = value_cache

    if nheads != kvheads:
        expansion = nheads // kvheads
        keys = keys.unsqueeze(2).expand(-1, -1, expansion, -1, -1).flatten(1, 2)
        values = values.unsqueeze(2).expand(-1, -1, expansion, -1, -1).flatten(1, 2)

    max_score = torch.full((queries.shape[0], queries.shape[1], queries.shape[2], 1), -float('inf'), device=queries.device, dtype=queries.dtype)
    accumulator = torch.zeros_like(queries)
    global_denominator = torch.zeros((queries.shape[0], queries.shape[1], queries.shape[2], 1), device=queries.device, dtype=queries.dtype)

    send_rank = (rank + 1) % world_size
    recv_rank = (rank - 1 + world_size) % world_size

    for i in range(world_size):
        block_idx = (rank - i + world_size) % world_size
        
        q_len = queries.shape[2]
        k_len = keys.shape[2]
        
        if rank == block_idx:
            mask = torch.triu(torch.ones(q_len, k_len, device=queries.device), diagonal=1).bool()
        elif rank > block_idx:
            mask = torch.zeros(q_len, k_len, device=queries.device).bool()
        else:
            mask = torch.ones(q_len, k_len, device=queries.device).bool()

        scale = scale_factor if scale_factor is not None else 1.0 / math.sqrt(queries.shape[-1])
        scores = (queries @ keys.transpose(-2, -1)) * scale
        
        scores.masked_fill_(mask, -float('inf'))

        current_max_score = torch.max(scores, dim=-1, keepdim=True)[0]
        new_max_score = torch.max(max_score, current_max_score)
        
        exp_scores = torch.exp(scores - new_max_score)
        
        accumulator = accumulator * torch.exp(max_score - new_max_score)
        accumulator += exp_scores @ values
        
        global_denominator = global_denominator * torch.exp(max_score - new_max_score) + exp_scores.sum(dim=-1, keepdim=True)
        
        max_score = new_max_score

        if i < world_size - 1:
            send_k = keys.contiguous()
            send_v = values.contiguous()
            recv_k = torch.empty_like(send_k)
            recv_v = torch.empty_like(send_v)
            
            # Even/odd rank ordering to avoid deadlock in simple blocking send/recv
            if rank % 2 == 0:
                dist.send(send_k, send_rank)
                dist.send(send_v, send_rank)
                dist.recv(recv_k, recv_rank)
                dist.recv(recv_v, recv_rank)
            else:
                dist.recv(recv_k, recv_rank)
                dist.recv(recv_v, recv_rank)
                dist.send(send_k, send_rank)
                dist.send(send_v, send_rank)

            keys = recv_k
            values = recv_v
    
    local_attn_output = accumulator / global_denominator
    local_attn_output = local_attn_output.transpose(1, 2).contiguous()
    return local_attn_output

register_attention_op(
    "ring_attention_blocking", store_op=_sdpa_store_op, compute_op=ring_attention_prefill_blocking
)

register_attention_op(
    "ring_attention_pipelined", store_op=_sdpa_store_op, compute_op=ring_attention_prefill_pipelined
)

register_attention_op(
    "ring_attention", store_op=_sdpa_store_op, compute_op=ring_attention_prefill
)
