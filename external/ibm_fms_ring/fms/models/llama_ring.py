import math
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple

from fms.modules.attention import MultiHeadAttention
from fms.distributed.strategy import DistributedStrategy, RingAttentionStrategy


# ============================================================
#   RING FORWARD (BLOCK-WISE WITH HETEROGENEOUS GPU SUPPORT)
# ============================================================

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
    """
    Full forward pass of a Transformer block using Ring Attention.
    """

    residual = x
    x_norm = self.ln(x)

    attn_output = RingAttentionKernel.ring_attention(
        x_norm=x_norm,
        attn_module=self.attn,
        strategy=self.distributed_strategy,
        valid_len=self.distributed_strategy._local_valid_len,
        mask=mask,
        position_ids=position_ids,
        past_key_value_state=past_key_value_state,
        use_cache=use_cache,
        causal=is_causal_mask,
    )

    # Unpack attention output (x, cache) if caching enabled
    if use_cache:
        x, cache = attn_output
    else:
        x = attn_output
        cache = None

    x = x + residual

    # Feed-forward block
    residual = x
    x = self.ff_ln(x)
    x = self.ff_sub_layer(x)
    x = x + residual

    if use_cache:
        return (x, cache)
    else:
        return x


# ============================================================
#                    RING ATTENTION KERNEL
# ============================================================

class RingAttentionKernel:

    # --------------------------------------------------------
    #   MAIN RING ATTENTION ENTRY POINT
    # --------------------------------------------------------
    @staticmethod
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

        batch_size, local_seq_len, emb_dim = x_norm.shape
        assert local_seq_len == valid_len

        # Global start index for this rank's Q tokens
        q_start_global = strategy.local_q_start

        # Slice valid portion
        x_slice = x_norm[:, :valid_len]

        # --------------------------------------------
        # Position handling (cache aware)
        # --------------------------------------------
        if use_cache and past_key_value_state is not None and past_key_value_state[0].numel() > 0:
            cached_len = past_key_value_state[0].shape[2]
        else:
            cached_len = 0

        if position_ids is not None:
            pos_ids = position_ids[:, q_start_global : q_start_global + valid_len]
            if cached_len:
                pos_ids = pos_ids + cached_len
        else:
            pos_ids = torch.arange(
                q_start_global + cached_len,
                q_start_global + cached_len + valid_len,
                device=x_norm.device,
            ).unsqueeze(0).expand(batch_size, -1)

        # --------------------------------------------
        # QKV + RoPE
        # --------------------------------------------
        if valid_len > 0:
            q, k_new, v_new = RingAttentionKernel._compute_qkv_and_rope(attn_module, x_slice, pos_ids)
        else:
            # Empty local shard
            nheads = attn_module.nheads
            dk = attn_module.emb_kq_per_head
            dv = attn_module.emb_v_per_head
            q = torch.empty((batch_size, nheads, 0, dk), device=x_norm.device)
            k_new = torch.empty((batch_size, nheads, 0, dk), device=x_norm.device)
            v_new = torch.empty((batch_size, nheads, 0, dv), device=x_norm.device)

        # --------------------------------------------
        # Append cache if decode mode
        # --------------------------------------------
        if cached_len > 0:
            k = torch.cat([past_key_value_state[0], k_new], dim=2)
            v = torch.cat([past_key_value_state[1], v_new], dim=2)
        else:
            k = k_new
            v = v_new

        # --------------------------------------------
        # Scaling factor
        # --------------------------------------------
        if attn_module.scale_factor:
            scale = 1.0 / attn_module.scale_factor
        else:
            scale = math.sqrt(attn_module.emb_kq_per_head)

        # --------------------------------------------
        # Ring Attention
        # --------------------------------------------
        output = RingAttentionKernel._compute_attention_ring(
            q, k, v,
            mask,
            strategy,
            q_start_global,
            valid_len,
            scale,
            torch.float32,
            causal,
        )

        # --------------------------------------------
        # Final projection
        # --------------------------------------------
        if valid_len > 0:
            proj = output.transpose(1, 2).reshape(batch_size, valid_len, -1)
            proj = attn_module.dense(proj)
        else:
            proj = torch.empty((batch_size, 0, emb_dim), device=x_norm.device)

        # --------------------------------------------
        # Return cache if requested
        # --------------------------------------------
        if use_cache:
            return proj, (k, v)
        return proj


    # --------------------------------------------------------
    #    QKV & RoPE
    # --------------------------------------------------------
    @staticmethod
    def _compute_qkv_and_rope(attn, x: Tensor, rope_position_ids: Optional[Tensor]):
        batch, seq, _ = x.shape

        q_proj, k_proj, v_proj = attn.in_proj(x, None, None)

        nheads, kvheads = attn.nheads, attn.kvheads
        dk = attn.emb_kq_per_head
        dv = attn.emb_v_per_head

        q = q_proj.view(batch, seq, nheads, dk)
        k = k_proj.view(batch, seq, kvheads, dk)
        v = v_proj.view(batch, seq, kvheads, dv)

        if attn.position_encoder and seq:
            assert rope_position_ids is not None
            max_pos = getattr(attn.position_encoder, "max_seq_len", 2048)
            rope_clamped = rope_position_ids.clamp(0, max_pos - 1)
            q, k = attn.position_encoder.adjusted_qk(q, k, rope_clamped)

        # Permute to (batch, heads, seq, dim)
        q, k, v = (t.permute(0, 2, 1, 3) for t in (q, k, v))

        # Multi-query attention: repeat smaller KV heads
        if nheads != kvheads:
            repeat = nheads // kvheads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        return q, k, v


    # --------------------------------------------------------
    #    RING ATTENTION: MAIN LOOP
    # --------------------------------------------------------
    @staticmethod
    def _compute_attention_ring(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        mask: Optional[Tensor],
        strategy: RingAttentionStrategy,
        q_start: int,
        num_q: int,
        scale: float,
        accum_dtype: torch.dtype,
        causal: bool,
    ) -> Tensor:

        # Early exit for empty shard
        if num_q == 0:
            return torch.empty((q.shape[0], q.shape[1], 0, v.shape[-1]), device=q.device)

        # First pass: compute global max for softmax
        max_score = RingAttentionKernel._max_pass(
            q, k, mask, q_start, num_q, strategy, scale, causal, accum_dtype
        )

        # Second pass: compute numerator + denominator
        numerator, denominator = RingAttentionKernel._sum_pass(
            q, k, v, mask, q_start, num_q, max_score, strategy, scale, accum_dtype, causal
        )

        return numerator / (denominator + 1e-9)


    # --------------------------------------------------------
    #      MAX PASS (ONLINE SOFTMAX PREP)
    # --------------------------------------------------------
    @staticmethod
    def _max_pass(
        q, k, mask, q_start, num_q, strategy, scale, causal, accum_dtype
    ) -> Tensor:

        batch, heads, _, _ = q.shape

        q_fp32 = q.to(accum_dtype)
        k_fp32 = k.to(accum_dtype)

        max_score = torch.full(
            (batch, heads, num_q, 1),
            torch.finfo(accum_dtype).min,
            device=q.device,
        )

        query_idx = torch.arange(q_start, q_start + num_q, device=q.device)

        # Ring iteration
        for hop in range(strategy.world_size):

            src = (strategy.rank - hop) % strategy.world_size
            k_len = k_fp32.shape[2]
            key_idx = torch.arange(src * strategy.block_size, src * strategy.block_size + k_len, device=q.device)

            if k_len > 0:
                mask_slice = (
                    mask[..., q_start:q_start+num_q, src*strategy.block_size:src*strategy.block_size+k_len]
                    if mask is not None else None
                )
                scores = RingAttentionKernel._attn_scores(q_fp32, k_fp32, query_idx, key_idx, scale, mask_slice, causal)
                max_score = torch.maximum(max_score, scores.amax(dim=-1, keepdim=True))

            # shift to next K block unless last hop
            if hop < strategy.world_size - 1:
                h = strategy.ring_shift_start(k_fp32, k_len)
                k_fp32, k_len = strategy.ring_shift_wait(h)

        return max_score


    # --------------------------------------------------------
    #      SUM PASS (NUMERATOR + DENOMINATOR)
    # --------------------------------------------------------
    @staticmethod
    def _sum_pass(
        q, k, v, mask, q_start, num_q, max_score, strategy, scale, accum_dtype, causal
    ):

        batch, heads, _, dv = v.shape

        q_fp32 = q.to(accum_dtype)
        k_fp32 = k.to(accum_dtype)
        v_fp32 = v.to(accum_dtype)

        numerator = torch.zeros((batch, heads, num_q, dv), device=q.device, dtype=accum_dtype)
        denominator = torch.zeros((batch, heads, num_q, 1), device=q.device, dtype=accum_dtype)

        query_idx = torch.arange(q_start, q_start + num_q, device=q.device)

        log_min = math.log(torch.finfo(accum_dtype).tiny)
        log_max = math.log(torch.finfo(accum_dtype).max)

        # Ring sweep
        for hop in range(strategy.world_size):

            src = (strategy.rank - hop) % strategy.world_size
            k_len = k_fp32.shape[2]
            key_idx = torch.arange(src * strategy.block_size, src * strategy.block_size + k_len, device=q.device)

            if k_len > 0:
                mask_slice = (
                    mask[..., q_start:q_start+num_q, src*strategy.block_size:src*strategy.block_size+k_len]
                    if mask is not None else None
                )
                scores = RingAttentionKernel._attn_scores(q_fp32, k_fp32, query_idx, key_idx, scale, mask_slice, causal)

                delta = scores - max_score
                masked = delta < -10000  # treat as -inf

                delta = delta.clamp(min=log_min, max=log_max)
                exp_scores = torch.exp(delta)

                exp_scores = exp_scores.masked_fill(masked, 0.0)
                exp_scores = exp_scores.masked_fill(torch.isneginf(max_score), 0.0)

                numerator += torch.matmul(exp_scores, v_fp32.narrow(2, 0, k_len))
                denominator += exp_scores.sum(dim=-1, keepdim=True)

            if hop < strategy.world_size - 1:
                h = strategy.ring_shift_start(k_fp32, k_len)
                k_fp32, k_len = strategy.ring_shift_wait(h)

                h = strategy.ring_shift_start(v_fp32, k_len)
                v_fp32, k_len = strategy.ring_shift_wait(h)

        return numerator, denominator


    # --------------------------------------------------------
    #      ATTENTION SCORE COMPUTATION
    # --------------------------------------------------------
    @staticmethod
    def _attn_scores(Q, K, q_idx, k_idx, scale, mask_slice, causal):

        batch, heads, q_len, _ = Q.shape
        k_len = K.shape[2]

        if q_len == 0 or k_len == 0:
            return Q.new_empty((batch, heads, q_len, k_len))

        scores = torch.matmul(Q / scale, K.transpose(-2, -1))

        if mask_slice is not None:
            scores = scores + mask_slice.to(scores.dtype)

        if causal:
            causal_mask = (k_idx[None, :] > q_idx[:, None])
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
            scores = scores.masked_fill(causal_mask, float("-inf"))

        return scores
