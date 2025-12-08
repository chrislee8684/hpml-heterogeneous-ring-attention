import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple

from fms.modules.attention import MultiHeadAttention
from fms.distributed.strategy import DistributedStrategy, RingAttentionStrategy

try:
    from triton_offdiag_block import block_softmax_stats_triton
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False

# Global state for profiling
_layer_call_counter = 0
_printed_stream_info = False

# Aggregate timing across all layers
_total_compute_ms = 0.0
_total_comm_ms = 0.0
_total_bytes = 0


def ring_forward(
    self,
    x,
    *,
    mask=None,
    position_ids=None,
    past_key_value_state=None,
    use_cache=False,
    is_causal_mask=False,
    attn_algorithm=None
):

    residual = x
    x_norm = self.ln(x)

    attn_output = ring_attention(
        x_norm=x_norm,
        attn_module=self.attn,
        strategy=self.distributed_strategy,
        valid_len=self.distributed_strategy._local_valid_len,
        mask=mask,
        position_ids=position_ids, # Sharded position_ids
        past_key_value_state=past_key_value_state,
        use_cache=use_cache,
        causal=is_causal_mask,
    )

    # Unpack attention output
    if use_cache:
        x, cache = attn_output
    else:
        x = attn_output
        cache = None

    x = x + residual

    # then we do FF and Add&Norm
    residual = x
    x = self.ff_ln(x)
    x = self.ff_sub_layer(x)
    x = x + residual

    if use_cache:
        return (x, cache)
    else:
        return x


def ring_attention(
    x_norm: Tensor,
    attn_module: MultiHeadAttention,
    strategy: RingAttentionStrategy,
    valid_len: int,
    *,
    mask: Optional[Tensor] = None,
    position_ids: Optional[Tensor] = None,
    past_key_value_state: Optional[Tuple[Tensor, Tensor]] = None,
    use_cache: bool = False,
    causal: bool = False,
):

    batch_size, num_valid_tokens_input_shard, emb_dim = x_norm.shape

    #decode check
    is_decode = (use_cache and past_key_value_state is not None and past_key_value_state[0].numel() > 0)

    if is_decode:
        return _ring_attention_pass_q(
            x_norm=x_norm,
            attn_module=attn_module,
            strategy=strategy,
            valid_len=valid_len,
            mask=mask,
            position_ids=position_ids,
            past_key_value_state=past_key_value_state,
            use_cache=use_cache,
            causal=causal,
        )
    else:
        return _ring_attention_pass_kv(
            x_norm=x_norm,
            attn_module=attn_module,
            strategy=strategy,
            valid_len=valid_len,
            mask=mask,
            position_ids=position_ids,
            past_key_value_state=past_key_value_state,
            use_cache=use_cache,
            causal=causal,
        )

def _ring_attention_pass_q(
    x_norm: Tensor,
    attn_module: MultiHeadAttention,
    strategy: RingAttentionStrategy,
    valid_len: int,
    *,
    mask: Optional[Tensor] = None,
    position_ids: Optional[Tensor] = None,
    past_key_value_state: Optional[Tuple[Tensor, Tensor]] = None,
    use_cache: bool = False,
    causal: bool = False,
):
    return 0

def _ring_attention_pass_kv(
    x_norm: Tensor,
    attn_module: MultiHeadAttention,
    strategy: RingAttentionStrategy,
    valid_len: int,
    *,
    mask: Optional[Tensor] = None,
    position_ids: Optional[Tensor] = None,
    past_key_value_state: Optional[Tuple[Tensor, Tensor]] = None,
    use_cache: bool = False,
    causal: bool = False,
):
      """
      Ring attention for prefill using pass-KV strategy.
      KV tensors rotate around the ring while Q stays local.
      """
      batch_size, num_valid_tokens_input_shard, emb_dim = x_norm.shape
    #   assert num_valid_tokens_input_shard == valid_len
    #   current_rank_token_global_start_idx = strategy.rank * strategy.block_size

      # in hetero:
      assert num_valid_tokens_input_shard == strategy.local_q_len
      current_rank_token_global_start_idx = strategy.local_q_start
      valid_len = strategy.local_q_len

      # slice to valid length to be safe
      current_rank_input_slice = x_norm[:, :valid_len]

      # compute position ids for the current tokens
      if position_ids is not None:
          position_ids_for_rope_computation = position_ids[:, current_rank_token_global_start_idx : current_rank_token_global_start_idx +valid_len]
      elif valid_len > 0:
          position_ids_for_rope_computation = torch.arange(
              current_rank_token_global_start_idx,
              current_rank_token_global_start_idx + valid_len,
              device=x_norm.device
          ).unsqueeze(0).expand(batch_size, -1)
      else:
          position_ids_for_rope_computation = None

      # compute QKV + RoPE for new tokens
      if valid_len:
          q, k, v = _compute_qkv_and_rope(
              attn_module, current_rank_input_slice, position_ids_for_rope_computation
          )
      else:
          nheads, emb_kq_per_head, emb_v_per_head = attn_module.nheads, attn_module.emb_kq_per_head, attn_module.emb_v_per_head
          q = k = torch.empty((batch_size, nheads, 0, emb_kq_per_head), device=x_norm.device, dtype=x_norm.dtype)
          v = torch.empty((batch_size, nheads, 0, emb_v_per_head), device=x_norm.device, dtype=x_norm.dtype)

      scale = attn_module.scale_factor or math.sqrt(attn_module.emb_kq_per_head)
      accum_dtype = torch.float16

      # main ring attention with pass-KV
      out = _compute_attention_ring_pass_kv(
          q, k, v, mask, strategy, current_rank_token_global_start_idx, valid_len, scale, accum_dtype, causal
      )

      if valid_len:
          proj = out.transpose(1, 2).reshape(batch_size, valid_len, -1)
          out = attn_module.dense(proj)
      else:
          out = torch.empty((batch_size, 0, emb_dim), device=x_norm.device, dtype=x_norm.dtype)

      # Return cache if requested
      if use_cache:
          return out, (k, v)
      else:
          return out


def _compute_qkv_and_rope(
    attn: MultiHeadAttention,
    x: Tensor,
    rope_position_ids: Optional[Tensor]
) -> Tuple[Tensor, Tensor, Tensor]:
    batch_size, seq_len, _ = x.shape # x is current_rank_input_slice, so seq_len is valid_len for this rank
    q_proj, k_proj, v_proj = attn.in_proj(x, None, None)
    nheads, kvheads = attn.nheads, attn.kvheads
    emb_kq_per_head, emb_v_per_head = attn.emb_kq_per_head, attn.emb_v_per_head

    # reshape & apply RoPE if needed
    q = q_proj.view(batch_size, seq_len, nheads, emb_kq_per_head)
    k = k_proj.view(batch_size, seq_len, kvheads, emb_kq_per_head)
    v = v_proj.view(batch_size, seq_len, kvheads, emb_v_per_head)
    if attn.position_encoder and seq_len:
        assert rope_position_ids is not None
        valid_rope_pos_mask = rope_position_ids.ne(-1)
        if valid_rope_pos_mask.any():
            rope_internal_max_seq_len = getattr(attn.position_encoder, "max_seq_len", 2048)
            clamped_rope_ids = rope_position_ids.clamp(0, rope_internal_max_seq_len - 1)
            q, k = attn.position_encoder.adjusted_qk(q, k, clamped_rope_ids)

    q, k, v = [x_tensor.permute(0, 2, 1, 3) for x_tensor in (q, k, v)]
    if nheads != kvheads:
        kv_to_q_head_ratio = nheads // kvheads
        k = k.repeat_interleave(kv_to_q_head_ratio, dim=1)
        v = v.repeat_interleave(kv_to_q_head_ratio, dim=1)
    return q, k, v


def _online_softmax_update(
    attn_weights: Tensor,
    v_block: Tensor,
    numerator: Tensor,
    denominator: Tensor,
    prev_max_score: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Online softmax update for a single block of attention.
    """
    # Find max in current block
    block_max = attn_weights.max(dim=-1, keepdim=True).values

    # Update global max
    new_max_score = torch.maximum(prev_max_score, block_max)
    # Correction factor for previous accumulations
    correction = torch.exp(prev_max_score - new_max_score)
    # Exp weights for current block (shifted by new max)
    exp_weights = torch.exp(attn_weights - new_max_score)
    # Update numerator and denominator with correction
    numerator = (numerator * correction) + torch.matmul(exp_weights, v_block)
    denominator = (denominator * correction) + exp_weights.sum(dim=-1, keepdim=True)

    return numerator, denominator, new_max_score


def _print_stream_info(strategy):
    """Print stream info once per forward pass (rank 0 only)."""
    if strategy.rank != 0:
        return

    default_stream = torch.cuda.current_stream()
    streams_different = strategy._comm_stream != default_stream
    print(f"[Ring Attention] Using separate streams: {streams_different}")


def _compute_attention_ring_pass_kv(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Optional[Tensor],
    strategy: RingAttentionStrategy,
    q_start: int,
    num_valid_tokens: int,
    scale: float,
    accum_dtype: torch.dtype,
    causal: bool,
) -> Tensor:
    """
    Ring attention with online softmax and async comm/compute overlap.
    """
    global _layer_call_counter, _printed_stream_info, _total_compute_ms, _total_comm_ms, _total_bytes

    batch_size, nheads, _, emb_v = q.shape[0], q.shape[1], q.shape[2], v.shape[-1]

    # Online softmax accumulators
    numerator = torch.zeros((batch_size, nheads, num_valid_tokens, emb_v), device=q.device, dtype=accum_dtype)
    denominator = torch.zeros((batch_size, nheads, num_valid_tokens, 1), device=q.device, dtype=accum_dtype)
    max_score = torch.full((batch_size, nheads, num_valid_tokens, 1), float("-inf"), device=q.device, dtype=accum_dtype)

    q_cast = q.to(accum_dtype)
    cur_k, cur_v = k.to(accum_dtype), v.to(accum_dtype)
    cur_len = cur_k.shape[2]
    query_indices = torch.arange(q_start, q_start + num_valid_tokens, device=q.device)

    # Timing accumulators
    PROFILE = True
    total_bytes_transferred = 0
    comm_events = []  # List of (start_event, end_event) tuples
    compute_events = []  # List of (start_event, end_event) tuples
    diag_compute_events = []  # (start, end) for diagonal block compute
    offdiag_compute_events = []  # (start, end) for off-diagonal block compute

    # Track layer for printing
    _layer_call_counter += 1
    current_layer = _layer_call_counter
    should_print = (current_layer == 1)  # Only print first layer

    # Print stream info once per forward pass
    if not _printed_stream_info:
        _printed_stream_info = True
        _print_stream_info(strategy)

    for i in range(strategy.world_size):
        # Start async comm for next iteration (overlaps with compute)
        comm_start_event = None

        # NEW: track what we actually compute this iteration
        did_diag_compute = False
        did_offdiag_compute = False

        if i < strategy.world_size - 1:
            reqs, recv_k, recv_v, recv_len, comm_start_event = strategy.ring_shift_kv_async(
                cur_k, cur_v, cur_len, enable_timing=PROFILE
            )

        # Record compute start event on DEFAULT stream
        compute_start = torch.cuda.Event(enable_timing=True) if PROFILE else None
        compute_end = torch.cuda.Event(enable_timing=True) if PROFILE else None
        if compute_start:
            compute_start.record()

        # Compute attention on current block
        source_rank = (strategy.rank - i) % strategy.world_size
        block_offset = source_rank * strategy.block_size
        is_diagonal = (i == 0)  # Diagonal block: Q and K are from same positions

        if num_valid_tokens > 0 and cur_len > 0:
            # For causal attention, check if this block is fully masked
            # Block is fully masked if all K positions are "future" relative to all Q positions
            # i.e., smallest K index > largest Q index
            k_start = block_offset
            k_end = block_offset + cur_len - 1
            q_end = q_start + num_valid_tokens - 1

            if causal and k_start > q_end:
                # Fully masked block - skip computation entirely!
                # (All K positions are "future" relative to all Q positions)
                pass
            elif is_diagonal and causal:
                # Diagonal block with causal: use FlashAttention (is_causal=True works correctly)
                flash_out = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=None,
                    dropout_p=0.0,
                    is_causal=True,
                    scale=1.0/scale
                )
                numerator = flash_out.to(accum_dtype)
                denominator = torch.ones((batch_size, nheads, num_valid_tokens, 1), device=q.device, dtype=accum_dtype)
                max_score = torch.zeros((batch_size, nheads, num_valid_tokens, 1), device=q.device, dtype=accum_dtype)
                did_diag_compute = True
            else:
                # Off-diagonal: Flash-style block softmax (stats-based)
                key_indices = torch.arange(block_offset, block_offset + cur_len, device=q.device)
                mask_slice = (
                    mask[..., q_start:q_start + num_valid_tokens,
                            block_offset:block_offset + cur_len]
                    if mask is not None else None
                )

                # TEMP: naive block stats (later replaced by CUDA kernel)
                z_block, l_block, m_block = _block_softmax_stats(
                    q_cast, cur_k, cur_v,
                    query_indices, key_indices,
                    scale, mask_slice, causal
                )

                numerator, denominator, max_score = _online_softmax_merge_stats(
                    z_block, l_block, m_block,
                    numerator, denominator, max_score
                )

                did_offdiag_compute = True

        # Record compute end event on DEFAULT stream
        if compute_end:
            compute_end.record()
            compute_events.append((compute_start, compute_end))

            if did_diag_compute:
                diag_compute_events.append((compute_start, compute_end))
            elif did_offdiag_compute:
                offdiag_compute_events.append((compute_start, compute_end))

        # Wait for comm and get end event (records AFTER transfers complete)
        if i < strategy.world_size - 1:
            cur_k, cur_v, cur_len, comm_end_event = strategy.ring_shift_kv_wait(
                reqs, recv_k, recv_v, recv_len, enable_timing=PROFILE
            )

            if PROFILE:
                total_bytes_transferred += cur_k.numel() * cur_k.element_size()
                total_bytes_transferred += cur_v.numel() * cur_v.element_size()
                if comm_start_event and comm_end_event:
                    comm_events.append((comm_start_event, comm_end_event))

            cur_k, cur_v = cur_k.to(accum_dtype), cur_v.to(accum_dtype)

    # Synchronize and compute timing from CUDA events
    torch.cuda.synchronize()

    # Calculate actual times from CUDA events
    total_comm_time_ms = 0.0
    total_compute_time_ms = 0.0
    total_offdiag_compute_ms = 0.0
    total_diag_compute_ms = 0.0

    for start_evt, end_evt in comm_events:
        total_comm_time_ms += start_evt.elapsed_time(end_evt)

    for start_evt, end_evt in compute_events:
        total_compute_time_ms += start_evt.elapsed_time(end_evt)
    
    for start_evt, end_evt in diag_compute_events:
        total_diag_compute_ms += start_evt.elapsed_time(end_evt)

    for start_evt, end_evt in offdiag_compute_events:
        total_offdiag_compute_ms += start_evt.elapsed_time(end_evt)

    # Accumulate timing for summary
    global _total_compute_ms, _total_comm_ms, _total_bytes
    _total_compute_ms += total_compute_time_ms
    _total_comm_ms += total_comm_time_ms
    _total_bytes += total_bytes_transferred

    # Print timing (only rank 0, first layer only)
    if PROFILE and strategy.rank == 1 and should_print:
        comm_bandwidth_gbps = (total_bytes_transferred / 1e9) / (total_comm_time_ms / 1000) if total_comm_time_ms > 0 else 0

        print(f"\n[Ring Attention layer={current_layer}] tokens={num_valid_tokens}, world_size={strategy.world_size}")
        print(f"  comm: {total_comm_time_ms:6.2f}ms | compute: {total_compute_time_ms:6.2f}ms")
        print(f"  diag: {total_diag_compute_ms:6.2f} ms | offdiag: {total_offdiag_compute_ms:6.4f} ms")
        print(f"  data: {total_bytes_transferred/1e6:.2f} MB | bandwidth: {comm_bandwidth_gbps:.2f} GB/s")
        if total_comm_time_ms < total_compute_time_ms:
            print(f"  comm hidden behind compute")
        else:
            print(f"  comm is bottleneck")

    if num_valid_tokens == 0:
        return torch.empty((batch_size, nheads, 0, emb_v), device=q.device, dtype=q.dtype)

    return (numerator / (denominator + 1e-8)).to(q.dtype)

def _compute_attention_ring_pass_q():
    return

def _attn_scores(
    Q: Tensor,
    K: Tensor,
    query_indices: Tensor, # global indices for queries in Q
    key_indices: Tensor,   # global indices for keys in K
    scale: float,
    mask: Optional[Tensor],
    causal: bool,
) -> Tensor:
    batch_size, nheads, num_q, _ = Q.shape # num_q is num_queries_in_block for Q
    num_k = K.shape[2]          # num_k is current_block_k_len for K
    if num_q == 0 or num_k == 0:
        return Q.new_empty((batch_size, nheads, num_q, num_k))

    scores = torch.matmul(Q / scale, K.transpose(-2, -1))
    if mask is not None:
        scores = scores + mask.to(scores.dtype)
    if causal:
        # build a [1,1,q_len,k_len] mask where key_pos > query_pos
        future_mask = (key_indices[None, :] > query_indices[:, None])
        future_mask = future_mask.unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(future_mask, float("-inf"))
    return scores


def reset_layer_counter():
    """Call this before each forward pass to reset the layer counter."""
    global _layer_call_counter, _printed_stream_info
    global _total_compute_ms, _total_comm_ms, _total_bytes
    _layer_call_counter = 0
    _printed_stream_info = False
    _total_compute_ms = 0.0
    _total_comm_ms = 0.0
    _total_bytes = 0


def print_timing_summary(rank: int = 0):
    """Print aggregate timing summary across all layers. Call after forward pass."""
    if rank != 0:
        return

    if _total_compute_ms == 0 and _total_comm_ms == 0:
        return

    num_layers = _layer_call_counter
    comm_bandwidth_gbps = (_total_bytes / 1e9) / (_total_comm_ms / 1000) if _total_comm_ms > 0 else 0

    print(f"\n[Ring Attention Summary] {num_layers} layers")
    print(f"  comm (total):    {_total_comm_ms:8.2f}ms")
    print(f"  compute (total): {_total_compute_ms:8.2f}ms")
    print(f"  data: {_total_bytes/1e6:.2f} MB | bandwidth: {comm_bandwidth_gbps:.2f} GB/s")
    if _total_comm_ms < _total_compute_ms:
        print(f"  comm hidden behind compute")
    else:
        print(f"  comm is bottleneck")


def _online_softmax_merge_stats(
    z_block: Tensor,      # [B, H, Q, D_v]
    l_block: Tensor,      # [B, H, Q, 1]
    m_block: Tensor,      # [B, H, Q, 1]
    numerator: Tensor,    # [B, H, Q, D_v]
    denominator: Tensor,  # [B, H, Q, 1]
    prev_max_score: Tensor,  # [B, H, Q, 1]
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Merge a new block's softmax stats (z_block, l_block, m_block)
    into global (numerator, denominator, prev_max_score).
    """
    # new global max per query
    new_max = torch.maximum(prev_max_score, m_block)

    # correction factors
    corr_prev  = torch.exp(prev_max_score - new_max)   # for old accumulators
    corr_block = torch.exp(m_block - new_max)          # for this block

    # merge
    numerator   = numerator * corr_prev  + z_block * corr_block
    denominator = denominator * corr_prev + l_block * corr_block

    return numerator, denominator, new_max

def _block_softmax_stats_naive(
    Q: Tensor,           # [B, H, Q_block, D_k]
    K: Tensor,           # [B, H, K_block, D_k]
    V: Tensor,           # [B, H, K_block, D_v]
    query_indices: Tensor,  # [Q_block] global positions
    key_indices: Tensor,    # [K_block] global positions
    scale: float,
    mask: Optional[Tensor],
    causal: bool,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Compute per-query block stats:
        m_block: max logits in this block
        l_block: sum_j exp(S_ij - m_block_i)
        z_block: sum_j exp(S_ij - m_block_i) * V_j
    using a naive matmul implementation.

    This is slow but matches the math. Later, replace internals with a CUDA Flash-style kernel.
    """
    B, H, Q_len, Dk = Q.shape
    K_len = K.shape[2]
    Dv = V.shape[-1]

    if Q_len == 0 or K_len == 0:
        m_block = Q.new_full((B, H, Q_len, 1), float("-inf"))
        l_block = Q.new_zeros((B, H, Q_len, 1))
        z_block = Q.new_zeros((B, H, Q_len, Dv))
        return z_block, l_block, m_block

    # 1. logits
    scores = torch.matmul(Q / scale, K.transpose(-2, -1))  # [B, H, Q_len, K_len]

    # 2. apply mask (padding + causal)
    if mask is not None:
        scores = scores + mask.to(scores.dtype)

    if causal:
        # future positions: key_idx > query_idx
        future_mask = (key_indices[None, :] > query_indices[:, None])  # [Q_len, K_len]
        future_mask = future_mask.unsqueeze(0).unsqueeze(0)            # [1,1,Q,K]
        scores = scores.masked_fill(future_mask, float("-inf"))

    # 3. m_block: per-query max
    m_block = scores.max(dim=-1, keepdim=True).values  # [B,H,Q,1]

    # 4. l_block: per-query sumexp
    exp_scores = torch.exp(scores - m_block)           # [B,H,Q,K]
    l_block = exp_scores.sum(dim=-1, keepdim=True)     # [B,H,Q,1]

    # 5. z_block: per-query weighted sum of V
    z_block = torch.matmul(exp_scores, V)              # [B,H,Q,Dv]

    return z_block, l_block, m_block



def _block_softmax_stats(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    query_indices: Tensor,
    key_indices: Tensor,
    scale: float,
    mask: Optional[Tensor],
    causal: bool,
) -> Tuple[Tensor, Tensor, Tensor]:
    # Triton path
    if _HAS_TRITON and Q.is_cuda:
        # NOTE: mask is currently ignored in the kernel;
        # if you rely on pad mask, apply it to scores or K/V before calling.

        print("[DEBUG] Using Triton offdiag kernel")
        return block_softmax_stats_triton(
            Q, K, V, query_indices, key_indices, scale, mask, causal
        )
    # Fallback: pure PyTorch, correct but slower
    print("[DEBUG] Using naive block stats")
    return _block_softmax_stats_naive(
        Q, K, V, query_indices, key_indices, scale, mask, causal
    )