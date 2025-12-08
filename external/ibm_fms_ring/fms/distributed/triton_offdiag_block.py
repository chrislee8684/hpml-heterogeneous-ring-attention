import triton
import triton.language as tl
import torch


@triton.jit
def _offdiag_block_stats_kernel(
    Q_ptr, K_ptr, V_ptr,
    query_idx_ptr, key_idx_ptr,
    Z_ptr, M_ptr, L_ptr,
    B: tl.constexpr, H: tl.constexpr,
    Q_LEN: tl.constexpr, K_LEN: tl.constexpr,
    D_K: tl.constexpr, D_V: tl.constexpr,
    stride_qb, stride_qh, stride_qq, stride_qd,
    stride_kb, stride_kh, stride_kk, stride_kd,
    stride_vb, stride_vh, stride_vk, stride_vd,
    stride_zb, stride_zh, stride_zq, stride_zd,
    stride_mb, stride_mh, stride_mq,
    stride_lb, stride_lh, stride_lq,
    scale,
    causal: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Each program handles one (b, h, q) row:
      - Loops over K in tiles of size BLOCK_K
      - Maintains running m, l, z in registers
      - Writes:
          M[b,h,q], L[b,h,q], Z[b,h,q,:]
    """

    # 1. Decode program id into (b, h, q)
    pid = tl.program_id(0)      # 0 .. (B * H * Q_LEN - 1)
    q_idx = pid % Q_LEN
    tmp   = pid // Q_LEN
    h_idx = tmp % H
    b_idx = tmp // H

    # Bounds check: if pid outside, return
    if b_idx >= B:
        return

    # 2. Pointers to this query row Q[b,h,q,:]
    Q_row_ptr = (
        Q_ptr
        + b_idx * stride_qb
        + h_idx * stride_qh
        + q_idx * stride_qq
    )

    # Load query vector [D_K]
    d_offsets = tl.arange(0, D_K)
    q_vec = tl.load(Q_row_ptr + d_offsets * stride_qd)

    # Load this query's global position
    q_pos = tl.load(query_idx_ptr + q_idx)
    
    # Offsets for value dimension (reused when loading and storing V)
    dv_offsets = tl.arange(0, D_V)

    # 3. Initialize running stats m, l, z
    NEG_INF = -1e9
    m = NEG_INF   # scalar
    l = 0.0       # scalar
    z = tl.zeros((D_V,), dtype=tl.float32)  # [D_V]

    # 4. Loop over K in tiles
    for k_start in range(0, K_LEN, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K_LEN

        # 4a. Load K_tile: [BLOCK_K, D_K]
        K_tile_ptr = (
            K_ptr
            + b_idx * stride_kb
            + h_idx * stride_kh
            + k_offsets[:, None] * stride_kk
            + d_offsets[None, :] * stride_kd
        )
        K_tile = tl.where(
            k_mask[:, None],
            tl.load(K_tile_ptr, mask=k_mask[:, None], other=0.0),
            0.0,
        )

        # 4b. Load V_tile: [BLOCK_K, D_V]
        V_tile_ptr = (
            V_ptr
            + b_idx * stride_vb
            + h_idx * stride_vh
            + k_offsets[:, None] * stride_vk
            + dv_offsets[None, :] * stride_vd
        )
        V_tile = tl.where(
            k_mask[:, None],
            tl.load(V_tile_ptr, mask=k_mask[:, None], other=0.0),
            0.0,
        )

        # 4c. Compute scores for this tile: [BLOCK_K]
        # score_j = <q_vec, K_j> / scale
        # tl.dot: (D_K) x (BLOCK_K, D_K)^T → (BLOCK_K)
        scores = tl.sum(q_vec[None, :] * K_tile, axis=1) / scale

        # 4d. Causal mask: if key_pos > query_pos, clamp to -inf
        if causal:
            key_pos = tl.load(key_idx_ptr + k_offsets, mask=k_mask, other=0)
            is_future = key_pos > q_pos
            scores = tl.where(is_future & k_mask, NEG_INF, scores)

        # 4e. Mask out padding keys (k_mask == False)
        scores = tl.where(k_mask, scores, NEG_INF)

        # 4f. Compute tile max m_tile
        m_tile = tl.max(scores, axis=0)

        # If tile is all -inf (no valid keys), skip
        # (m_tile will be NEG_INF; exp(scores - m_tile) will be 0)
        # We can just let math proceed; it won’t change l,z.

        # 4g. Compute exp(scores - m_tile)
        scores_shifted = scores - m_tile
        exp_scores = tl.exp(scores_shifted)

        # 4h. Tile sumexp and tile z
        l_tile = tl.sum(exp_scores, axis=0)                 # scalar
        z_tile = tl.sum(exp_scores[:, None] * V_tile, axis=0)  # [D_V]

        # 4i. Merge with running m, l, z
        new_m = tl.maximum(m, m_tile)

        # If new_m is -inf (no keys so far), we can just overwrite
        # but the general formula also works:
        alpha = tl.exp(m - new_m)
        beta  = tl.exp(m_tile - new_m)

        z = z * alpha + z_tile * beta
        l = l * alpha + l_tile * beta
        m = new_m

    # 5. Write out m, l, z for this (b,h,q)
    # Z[b,h,q,:]
    Z_row_ptr = (
        Z_ptr
        + b_idx * stride_zb
        + h_idx * stride_zh
        + q_idx * stride_zq
    )
    tl.store(Z_row_ptr + dv_offsets * stride_zd, z)

    # M[b,h,q]
    M_ptr_elt = (
        M_ptr
        + b_idx * stride_mb
        + h_idx * stride_mh
        + q_idx * stride_mq
    )
    tl.store(M_ptr_elt, m)

    # L[b,h,q]
    L_ptr_elt = (
        L_ptr
        + b_idx * stride_lb
        + h_idx * stride_lh
        + q_idx * stride_lq
    )
    tl.store(L_ptr_elt, l)



def block_softmax_stats_triton(
    Q: torch.Tensor,           # [B,H,Q_len,D_k], float16/32
    K: torch.Tensor,           # [B,H,K_len,D_k]
    V: torch.Tensor,           # [B,H,K_len,D_v]
    query_indices: torch.Tensor,  # [Q_len], long
    key_indices: torch.Tensor,    # [K_len], long
    scale: float,
    mask: torch.Tensor | None,
    causal: bool,
    block_k: int = 64,
):
    """
    Triton implementation of _block_softmax_stats_naive.

    Returns:
      z_block: [B,H,Q_len,D_v]
      l_block: [B,H,Q_len,1]
      m_block: [B,H,Q_len,1]
    """
    assert Q.is_cuda and K.is_cuda and V.is_cuda
    assert Q.dtype in (torch.float16, torch.bfloat16, torch.float32)

    B, H, Q_len, D_k = Q.shape
    _, _, K_len, D_v = V.shape

    # Allocate outputs in float32 accum dtype
    device = Q.device
    z_block = torch.zeros((B, H, Q_len, D_v), dtype=torch.float32, device=device)
    l_block = torch.zeros((B, H, Q_len, 1),  dtype=torch.float32, device=device)
    m_block = torch.full((B, H, Q_len, 1), -1e9, dtype=torch.float32, device=device)

    # Make sure indices are on device
    query_indices = query_indices.to(device=device, dtype=torch.long)
    key_indices   = key_indices.to(device=device, dtype=torch.long)

    # Strides
    stride_qb, stride_qh, stride_qq, stride_qd = Q.stride()
    stride_kb, stride_kh, stride_kk, stride_kd = K.stride()
    stride_vb, stride_vh, stride_vk, stride_vd = V.stride()
    stride_zb, stride_zh, stride_zq, stride_zd = z_block.stride()
    stride_mb, stride_mh, stride_mq, _        = m_block.stride()
    stride_lb, stride_lh, stride_lq, _        = l_block.stride()

    # Launch grid: one program per (b,h,q)
    grid = (B * H * Q_len,)

    _offdiag_block_stats_kernel[grid](
        Q, K, V,
        query_indices, key_indices,
        z_block, m_block, l_block,
        B, H, Q_len, K_len, D_k, D_v,
        stride_qb, stride_qh, stride_qq, stride_qd,
        stride_kb, stride_kh, stride_kk, stride_kd,
        stride_vb, stride_vh, stride_vk, stride_vd,
        stride_zb, stride_zh, stride_zq, stride_zd,
        stride_mb, stride_mh, stride_mq,
        stride_lb, stride_lh, stride_lq,
        scale,
        causal=causal,
        BLOCK_K=block_k,
        num_warps=4, num_stages=2,
    )

    return z_block, l_block, m_block