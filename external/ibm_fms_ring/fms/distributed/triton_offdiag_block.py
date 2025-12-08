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
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Each program handles BLOCK_Q queries for a fixed (b, h), and loops over K in BLOCK_K tiles.
    """

    # How many query blocks per (b,h)
    Q_BLOCKS = (Q_LEN + BLOCK_Q - 1) // BLOCK_Q

    pid = tl.program_id(0)  # 0 .. B*H*Q_BLOCKS-1

    # Decode pid -> (b_idx, h_idx, q_block_idx)
    bh_blocks = H * Q_BLOCKS
    b_idx = pid // bh_blocks
    rem = pid % bh_blocks
    h_idx = rem // Q_BLOCKS
    q_block_idx = rem % Q_BLOCKS

    if b_idx >= B:
        return

    # Query indices within this block
    q_offsets = q_block_idx * BLOCK_Q + tl.arange(0, BLOCK_Q)
    q_mask = q_offsets < Q_LEN

    # Pointers to Q_tile
    d_offsets = tl.arange(0, D_K)
    Q_tile_ptr = (
        Q_ptr
        + b_idx * stride_qb
        + h_idx * stride_qh
        + q_offsets[:, None] * stride_qq
        + d_offsets[None, :] * stride_qd
    )
    Q_tile = tl.where(
        q_mask[:, None],
        tl.load(Q_tile_ptr, mask=q_mask[:, None], other=0.0),
        0.0,
    )   # [BLOCK_Q, D_K]

    dv_offsets = tl.arange(0, D_V)

    # Load this block's query global positions
    q_pos = tl.load(query_idx_ptr + q_offsets, mask=q_mask, other=0)

    NEG_INF = -1e9
    # Running stats per query in this block
    m = tl.full((BLOCK_Q,), NEG_INF, tl.float32)
    l = tl.zeros((BLOCK_Q,), tl.float32)
    z = tl.zeros((BLOCK_Q, D_V), tl.float32)

    # Loop over K in BLOCK_K tiles
    for k_start in range(0, K_LEN, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K_LEN

        # K_tile: [BLOCK_K, D_K]
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

        # V_tile: [BLOCK_K, D_V]
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

        # scores_tile: [BLOCK_Q, BLOCK_K] = Q_tile @ K_tile^T / scale
        scores = tl.dot(Q_tile, tl.trans(K_tile)) / scale

        # Causal mask: key_pos > query_pos → -inf
        if causal:
            key_pos = tl.load(key_idx_ptr + k_offsets, mask=k_mask, other=0)
            # shapes: q_pos: [BLOCK_Q], key_pos: [BLOCK_K]
            # broadcast to [BLOCK_Q, BLOCK_K]
            is_future = (key_pos[None, :] > q_pos[:, None])
            scores = tl.where(is_future & q_mask[:, None] & k_mask[None, :], NEG_INF, scores)

        # Mask out invalid (q,k) pairs
        scores = tl.where(q_mask[:, None] & k_mask[None, :], scores, NEG_INF)

        # Tile-wise max per query
        m_tile = tl.max(scores, axis=1)  # [BLOCK_Q]

        # Shifted scores, exp, sumexp
        scores_shifted = scores - m_tile[:, None]
        exp_scores = tl.exp(scores_shifted)
        l_tile = tl.sum(exp_scores, axis=1)  # [BLOCK_Q]

        # z_tile: [BLOCK_Q, D_V] = exp_scores @ V_tile
        z_tile = tl.dot(exp_scores, V_tile)  # [BLOCK_Q, D_V]

        # Online merge with running m,l,z
        new_m = tl.maximum(m, m_tile)
        alpha = tl.exp(m - new_m)
        beta = tl.exp(m_tile - new_m)

        z = z * alpha[:, None] + z_tile * beta[:, None]
        l = l * alpha + l_tile * beta
        m = new_m

    # Write back: Z[b,h,q,:], M[b,h,q], L[b,h,q]
    Z_base_ptr = (
        Z_ptr
        + b_idx * stride_zb
        + h_idx * stride_zh
        + q_offsets[:, None] * stride_zq
        + dv_offsets[None, :] * stride_zd
    )
    tl.store(Z_base_ptr, z, mask=q_mask[:, None])

    M_base_ptr = (
        M_ptr
        + b_idx * stride_mb
        + h_idx * stride_mh
        + q_offsets * stride_mq
    )
    tl.store(M_base_ptr, m, mask=q_mask)

    L_base_ptr = (
        L_ptr
        + b_idx * stride_lb
        + h_idx * stride_lh
        + q_offsets * stride_lq
    )
    tl.store(L_base_ptr, l, mask=q_mask)



def block_softmax_stats_triton(
    Q: torch.Tensor,           # [B,H,Q_len,D_k]
    K: torch.Tensor,           # [B,H,K_len,D_k]
    V: torch.Tensor,           # [B,H,K_len,D_v]
    query_indices: torch.Tensor,
    key_indices: torch.Tensor,
    scale: float,
    mask: torch.Tensor | None,
    causal: bool,
    block_q: int = 32,
    block_k: int = 64,
):
    assert Q.is_cuda and K.is_cuda and V.is_cuda
    B, H, Q_len, D_k = Q.shape
    _, _, K_len, D_v = V.shape

    device = Q.device

    # ensure contiguous for nicer strides
    Q = Q.contiguous()
    K = K.contiguous()
    V = V.contiguous()

    # outputs in fp32 accum dtype
    z_block = torch.zeros((B, H, Q_len, D_v), dtype=torch.float32, device=device)
    l_block = torch.zeros((B, H, Q_len, 1),  dtype=torch.float32, device=device)
    m_block = torch.full((B, H, Q_len, 1), -1e9, dtype=torch.float32, device=device)

    query_indices = query_indices.to(device=device, dtype=torch.long)
    key_indices   = key_indices.to(device=device, dtype=torch.long)

    stride_qb, stride_qh, stride_qq, stride_qd = Q.stride()
    stride_kb, stride_kh, stride_kk, stride_kd = K.stride()
    stride_vb, stride_vh, stride_vk, stride_vd = V.stride()
    stride_zb, stride_zh, stride_zq, stride_zd = z_block.stride()
    stride_mb, stride_mh, stride_mq, _        = m_block.stride()
    stride_lb, stride_lh, stride_lq, _        = l_block.stride()

    # grid: number of (b,h,q_block) tiles
    q_blocks = (Q_len + block_q - 1) // block_q
    grid = (B * H * q_blocks,)

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
        BLOCK_Q=block_q,
        BLOCK_K=block_k,
        num_warps=4,
        num_stages=2,
    )

    # add last dim for l/m to match [B,H,Q,1]
    l_block = l_block.unsqueeze(-1).squeeze(-1)  # already [B,H,Q,1]
    m_block = m_block.unsqueeze(-1).squeeze(-1)  # already [B,H,Q,1]
    return z_block, l_block, m_block