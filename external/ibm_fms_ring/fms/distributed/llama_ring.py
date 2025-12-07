import math
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple

from fms.modules.attention import MultiHeadAttention
from fms.distributed.strategy import DistributedStrategy, RingAttentionStrategy



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
      assert num_valid_tokens_input_shard == valid_len
      current_rank_token_global_start_idx = strategy.rank * strategy.block_size

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
    Ring attention computation using online softmax with async communication overlap.

    Pattern: Start Send → Compute (overlaps with send) → Wait for Receive → Swap buffers

    Key insight: ring_shift_kv_start both sends current_k/v AND receives into new buffers.
    We can compute on current_k/v while the send/recv is in flight since we're just reading.
    """
    batch_size, nheads, _, _ = q.shape
    emb_v = v.shape[-1]

    # Initialize online softmax accumulators
    numerator = torch.zeros((batch_size, nheads, num_valid_tokens, emb_v), device=q.device, dtype=accum_dtype)
    denominator = torch.zeros((batch_size, nheads, num_valid_tokens, 1), device=q.device, dtype=accum_dtype)
    max_score = torch.full((batch_size, nheads, num_valid_tokens, 1), float("-inf"), device=q.device, dtype=accum_dtype)

    # Cast tensors to accumulator dtype (on default stream)
    q_cast = q.to(accum_dtype)
    current_k = k.to(accum_dtype)
    current_v = v.to(accum_dtype)

    # Global query indices for causal masking
    query_global_indices = torch.arange(q_start, q_start + num_valid_tokens, device=q.device)

    # Track current KV block length
    current_k_len = current_k.shape[2]

    # Create a separate CUDA stream for compute
    compute_stream = torch.cuda.Stream(device=q.device)
    default_stream = torch.cuda.current_stream()

    # Ensure compute stream can see initial tensors created on default stream
    compute_stream.wait_stream(default_stream)

    # Async handles
    next_k_handle = None
    next_v_handle = None

    # Main ring loop
    # Pattern per iteration: Start Async Send/Recv → Compute → Wait for Recv → Swap
    for i in range(strategy.world_size):
        print(f"[RING DEBUG rank={strategy.rank}] === iteration {i}/{strategy.world_size} ===", flush=True)

        # STEP 1: Start async communication (send current_k/v, receive into new buffers)
        # Do this BEFORE compute so send can overlap with compute
        if i < strategy.world_size - 1:
            print(f"[RING DEBUG rank={strategy.rank}] STEP 1: starting async KV shift", flush=True)
            next_k_handle, next_v_handle = strategy.ring_shift_kv_start(
                current_k, current_v, current_k_len, is_decode_step=False
            )
            print(f"[RING DEBUG rank={strategy.rank}] async comm initiated", flush=True)

        # STEP 2: Compute attention on current KV block (overlaps with send/recv)
        source_rank = (strategy.rank - i + strategy.world_size) % strategy.world_size
        block_offset = source_rank * strategy.block_size
        print(f"[RING DEBUG rank={strategy.rank}] STEP 2: computing, source_rank={source_rank}, block_offset={block_offset}", flush=True)

        with torch.cuda.stream(compute_stream):
            if num_valid_tokens > 0 and current_k_len > 0:
                # Global indices for current KV block
                key_block_global_indices = torch.arange(
                    block_offset, block_offset + current_k_len, device=q.device
                )

                # Slice mask for current block
                current_mask_slice = None
                if mask is not None:
                    current_mask_slice = mask[...,
                        q_start : q_start + num_valid_tokens,
                        block_offset : block_offset + current_k_len
                    ]

                # Compute attention scores
                attn_weights = _attn_scores(
                    q_cast, current_k, query_global_indices, key_block_global_indices,
                    scale, current_mask_slice, causal
                )

                # Update accumulators using online softmax
                numerator, denominator, max_score = _online_softmax_update(
                    attn_weights, current_v, numerator, denominator, max_score
                )

        print(f"[RING DEBUG rank={strategy.rank}] compute kernels launched", flush=True)

        # STEP 3: Wait for async recv to complete and swap buffers for next iteration
        if i < strategy.world_size - 1:
            # First sync compute stream - we need compute to finish before we swap buffers
            print(f"[RING DEBUG rank={strategy.rank}] STEP 3: syncing compute stream", flush=True)
            compute_stream.synchronize()
            print(f"[RING DEBUG rank={strategy.rank}] compute stream synced, waiting for recv", flush=True)

            # Now wait for the P2P recv to complete
            next_k, next_k_len = strategy.ring_shift_wait(next_k_handle)
            next_v, _ = strategy.ring_shift_wait(next_v_handle)
            print(f"[RING DEBUG rank={strategy.rank}] recv complete, next_k_len={next_k_len}", flush=True)

            # Swap buffers: received data becomes current for next iteration
            current_k = next_k.to(accum_dtype)
            current_v = next_v.to(accum_dtype)
            current_k_len = next_k_len

            # Ensure compute stream sees the new buffers
            compute_stream.wait_stream(default_stream)

    # Final sync
    compute_stream.synchronize()
    print(f"[RING DEBUG rank={strategy.rank}] ring loop complete", flush=True)

    # Handle empty case
    if num_valid_tokens == 0:
        return torch.empty((q.shape[0], q.shape[1], 0, v.shape[-1]), device=q.device, dtype=q.dtype)

    # Final normalization
    out = numerator / (denominator + torch.finfo(accum_dtype).eps)
    return out.to(q.dtype)


def _compute_attention_ring_pass_q():
    return


def _attn_scores(
    Q: Tensor,
    K: Tensor,
    query_indices: Tensor,
    key_indices: Tensor,
    scale: float,
    mask: Optional[Tensor],
    causal: bool,
) -> Tensor:
    print(f"[ATTN_SCORES DEBUG] entered, Q.shape={Q.shape}, K.shape={K.shape}", flush=True)
    batch_size, nheads, num_q, _ = Q.shape
    num_k = K.shape[2]
    if num_q == 0 or num_k == 0:
        return Q.new_empty((batch_size, nheads, num_q, num_k))

    print(f"[ATTN_SCORES DEBUG] calling torch.matmul...", flush=True)
    scores = torch.matmul(Q / scale, K.transpose(-2, -1))
    print(f"[ATTN_SCORES DEBUG] matmul done", flush=True)

    if mask is not None:
        scores = scores + mask.to(scores.dtype)

    if causal:
        print(f"[ATTN_SCORES DEBUG] applying causal mask...", flush=True)
        future_mask = (key_indices[None, :] > query_indices[:, None])
        future_mask = future_mask.unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(future_mask, float("-inf"))
        print(f"[ATTN_SCORES DEBUG] causal mask applied", flush=True)

    return scores
