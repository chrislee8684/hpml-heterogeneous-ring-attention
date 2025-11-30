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
      accum_dtype = torch.float32

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

#rewriting to use online softmax and use single approach (use online softmax and calculate max iteratively)
#now with async communication-computation overlap
def _compute_attention_ring_pass_kv(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Optional[Tensor],
    strategy: RingAttentionStrategy,
    q_start: int, # global start index for queries in q
    num_valid_tokens: int,   # number of queries in q for this rank's block (num_queries_in_block)
    scale: float,
    accum_dtype: torch.dtype,
    causal: bool,
) -> Tensor:

    # computing ring attention in one pass using online softmax
    batch_size, nheads, _, _ = q.shape
    emb_v = v.shape[-1]

    #initialize accumulator for num and denom
    numerator = torch.zeros((batch_size, nheads, num_valid_tokens, emb_v), device=q.device, dtype=accum_dtype)
    denominator = torch.zeros((batch_size, nheads, num_valid_tokens, 1), device=q.device, dtype=accum_dtype)

    #max score
    prev_max_score = torch.full((batch_size, nheads, num_valid_tokens, 1), float("-inf"), device=q.device, dtype=accum_dtype)

    #cast to float32
    q_fp32 = q.to(accum_dtype)
    current_k = k.to(accum_dtype)
    current_v = v.to(accum_dtype)

    #getting global query indices
    query_global_indices = torch.arange(q_start, q_start + num_valid_tokens, device=q.device)

    # track current K/V length
    current_k_len = current_k.shape[2]

    # async handles for K and V
    k_handle = None
    v_handle = None

    #main ring loop with async overlap
    for i in range(strategy.world_size):

        # wait for previous async transfer 
        if i > 0 and k_handle is not None and v_handle is not None:
            current_k, current_k_len = strategy.ring_shift_wait(k_handle)
            current_v, _ = strategy.ring_shift_wait(v_handle)
            # cast to accum dtype after receiving
            current_k = current_k.to(accum_dtype)
            current_v = current_v.to(accum_dtype)

        # start async transfer for next iteration
        if i < strategy.world_size - 1:
            k_handle = strategy.ring_shift_start(current_k, current_k_len, is_decode_step=False)
            v_handle = strategy.ring_shift_start(current_v, current_k_len, is_decode_step=False)

        # compute attention for current block (overlapped with transfer)
        # finding original location of kv cache
        source_rank = (strategy.rank - i) % strategy.world_size
        block_offset_for_source_rank = source_rank * strategy.block_size

        if num_valid_tokens > 0 and current_k_len > 0:
            #getting global indices of key
            key_block_global_indices = torch.arange(
                block_offset_for_source_rank,
                block_offset_for_source_rank + current_k_len,
                device=q.device
            )
            #slicing mask
            current_mask_slice = None
            if mask is not None:
                current_mask_slice = mask[...,
                    q_start : q_start + num_valid_tokens,
                    block_offset_for_source_rank : block_offset_for_source_rank + current_k_len
                ]
            #calculate attention score
            attn_weights = _attn_scores(
                q_fp32, current_k, query_global_indices, key_block_global_indices, scale, current_mask_slice, causal
            )

            #online softmax implementation
            #first find max in block
            block_max = attn_weights.max(dim=-1, keepdim=True).values
            #now update max
            new_max_score = torch.maximum(prev_max_score, block_max)

            #now calculate correction factor
            correction = torch.exp(prev_max_score-new_max_score)
            #find exponentials for current block, add to previous which is found through recurrence relation
            exp_weights = torch.exp(attn_weights - new_max_score)
            #update numerator and denominator
            numerator = (numerator*correction) + torch.matmul(exp_weights, current_v)
            denominator = (denominator*correction) + exp_weights.sum(dim=-1, keepdim=True)

            #update state for next iteration
            prev_max_score = new_max_score

    if num_valid_tokens == 0:
        return torch.empty((q.shape[0], q.shape[1], 0, v.shape[-1]),
                            device=q.device, dtype=q.dtype)
    out = numerator / (denominator + torch.finfo(accum_dtype).eps)
    return out.to(q.dtype)

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



