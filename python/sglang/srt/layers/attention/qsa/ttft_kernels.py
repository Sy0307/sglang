"""Triton kernels for QSA prefill.

The indexer scoring kernel keeps the same dot order as the TileLang path,
so the logits match bit-for-bit for bf16 with four heads.
"""
import math

import torch
import triton
import triton.language as tl


@triton.jit
def _qsa_mqa_prefill_kernel(q_ptr, k_ptr, out_ptr, starts_ptr, ends_ptr, rows, keys, inv_scale,
                            HEADS: tl.constexpr, DIM: tl.constexpr, BQ: tl.constexpr, BN: tl.constexpr,
                            KEYS_PER_CTA: tl.constexpr, KQ: tl.constexpr = False, SUM: tl.constexpr = 0):
    pid_q = tl.program_id(0)
    pid_k = tl.program_id(1)
    r = pid_q * BQ + tl.arange(0, BQ)
    r_ok = r < rows
    starts = tl.load(starts_ptr + r, mask=r_ok, other=0)
    ends = tl.load(ends_ptr + r, mask=r_ok, other=0)
    # Q tile [BQ*HEADS, DIM]; row index = row * HEADS + head.
    qr = pid_q * BQ * HEADS + tl.arange(0, BQ * HEADS)
    d = tl.arange(0, DIM)
    q = tl.load(q_ptr + qr[:, None] * DIM + d[None, :], mask=(qr < rows * HEADS)[:, None], other=0.0)
    lo = tl.min(tl.where(r_ok, starts, 2147483647), axis=0)
    hi = tl.max(tl.where(r_ok, ends, -1), axis=0)
    k0 = pid_k * KEYS_PER_CTA
    for kb in range(0, KEYS_PER_CTA, BN):
        c = k0 + kb + tl.arange(0, BN)
        c_ok = c < keys
        valid = (c[None, :] >= starts[:, None]) & (c[None, :] < ends[:, None])
        if (k0 + kb + BN > lo) & (k0 + kb < hi):
            if KQ:
                kn = tl.load(k_ptr + c[:, None] * DIM + d[None, :], mask=c_ok[:, None], other=0.0)
                s = tl.trans(tl.dot(kn, tl.trans(q)))  # K @ Q^T, as the TileLang kernel
            else:
                kt = tl.load(k_ptr + c[None, :] * DIM + d[:, None], mask=c_ok[None, :], other=0.0)
                s = tl.dot(q, kt)  # [BQ*HEADS, BN] fp32
            s = tl.maximum(s, 0.0)
            if SUM == 0:
                s = tl.sum(tl.reshape(s, (BQ, HEADS, BN)), axis=1)
            else:
                s4 = tl.reshape(tl.permute(tl.reshape(s, (BQ, HEADS, BN)), (0, 2, 1)), (BQ, BN, 2, 2))
                a, b = tl.split(s4)  # a = [h0, h2], b = [h1, h3]
                h0, h2 = tl.split(a)
                h1, h3 = tl.split(b)
                if SUM == 1:
                    s = (h0 + h1) + (h2 + h3)
                else:
                    s = ((h0 + h1) + h2) + h3
            val = tl.where(valid, s * inv_scale, -float('inf'))
        else:
            val = tl.full((BQ, BN), -float('inf'), tl.float32)
        tl.store(out_ptr + r[:, None].to(tl.int64) * keys + c[None, :], val, mask=r_ok[:, None] & c_ok[None, :])


def triton_qsa_mqa_prefill(q, k, row_starts, row_ends, score_scale=None, *, BQ=32, BN=64, KEYS_PER_CTA=256, num_warps=4, num_stages=2, KQ=False, SUM=0):
    rows, heads, dim = q.shape
    keys = k.shape[0]
    out = torch.empty((rows, keys), dtype=torch.float32, device=q.device)
    if rows == 0 or keys == 0:
        return out.fill_(-float('inf'))
    scale = score_scale or math.sqrt(dim)
    inv_scale = float(torch.tensor(1.0, dtype=torch.float32) / torch.tensor(scale, dtype=torch.float32))
    qc = q.contiguous()
    kc = k[:, 0].contiguous()
    starts = row_starts.to(device=q.device, dtype=torch.int32).contiguous()
    ends = row_ends.to(device=q.device, dtype=torch.int32).contiguous()
    grid = (triton.cdiv(rows, BQ), triton.cdiv(keys, KEYS_PER_CTA))
    _qsa_mqa_prefill_kernel[grid](qc, kc, out, starts, ends, rows, keys, inv_scale, HEADS=heads, DIM=dim, BQ=BQ, BN=BN,
                                  KEYS_PER_CTA=KEYS_PER_CTA, KQ=KQ, SUM=SUM, num_warps=num_warps, num_stages=num_stages)
    return out


@triton.jit
def _sparse_gqa_chunk_prefill_paged(
    q, k_pool, v_pool, out, indices, cu_q, kv_lens, req_to_token, req_indices, scale, topk,
    sq_m: tl.constexpr, sq_h: tl.constexpr, sq_d: tl.constexpr,
    sk_n: tl.constexpr, sk_h: tl.constexpr, sk_d: tl.constexpr,
    sv_n: tl.constexpr, sv_h: tl.constexpr, sv_d: tl.constexpr,
    so_m: tl.constexpr, so_h: tl.constexpr, so_d: tl.constexpr,
    si_m: tl.constexpr, si_g: tl.constexpr, si_n: tl.constexpr,
    sr_0: tl.constexpr, sr_1: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr, GROUP_SIZE: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
):
    """_sparse_gqa_chunk_prefill reading K/V straight from the paged pool via
    req_to_token (same loads and math as the packed path, bit-identical output)."""
    query_relative = tl.program_id(0).to(tl.int64)
    batch_group = tl.program_id(1)
    group = batch_group % NUM_KV_HEADS
    batch = batch_group // NUM_KV_HEADS
    q_start = tl.load(cu_q + batch)
    q_end = tl.load(cu_q + batch + 1)
    query = (q_start + query_relative).to(tl.int64)
    if query >= q_end:
        return
    kv_len = tl.load(kv_lens + batch).to(tl.int64)
    req = tl.load(req_indices + batch).to(tl.int64)
    visible = query_relative + kv_len - (q_end - q_start) + 1
    row_topk = tl.minimum(topk, visible)
    row_limit = tl.minimum(topk, ((row_topk + BLOCK_N - 1) // BLOCK_N) * BLOCK_N)
    offs_h = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    q_values = tl.load(q + query * sq_m + (group * GROUP_SIZE + offs_h[:, None]) * sq_h + offs_d[None, :] * sq_d,
                       mask=(offs_h < GROUP_SIZE)[:, None], other=0.0)
    q_values = (q_values * scale * 1.4426950408).to(q_values.dtype)
    k_base = k_pool + group * sk_h
    v_base = v_pool + group * sv_h
    r2t = req_to_token + req * sr_0
    idx_row = indices + query * si_m + group * si_g
    max_value = tl.full([BLOCK_M], -float("inf"), tl.float32)
    normalizer = tl.zeros([BLOCK_M], tl.float32)
    accumulator = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
    offs_n = tl.arange(0, BLOCK_N)
    for start in range(0, row_limit, BLOCK_N):
        current = start + offs_n
        token = tl.load(idx_row + current * si_n, mask=current < topk, other=-1)
        valid = token >= 0
        slot = tl.load(r2t + token.to(tl.int64) * sr_1, mask=valid, other=0).to(tl.int64)
        keys = tl.load(k_base + slot[None, :] * sk_n + offs_d[:, None] * sk_d, mask=valid[None, :], other=0.0)
        values = tl.load(v_base + slot[:, None] * sv_n + offs_d[None, :] * sv_d, mask=valid[:, None], other=0.0)
        keys = keys.to(q_values.dtype)
        values = values.to(q_values.dtype)
        scores = tl.where(valid[None, :], tl.dot(q_values, keys), -float("inf"))
        next_max = tl.maximum(max_value, tl.max(scores, 1))
        alpha = tl.math.exp2(max_value - next_max)
        probabilities = tl.math.exp2(scores - next_max[:, None])
        accumulator = tl.dot(probabilities.to(values.dtype), values, accumulator * alpha[:, None])
        normalizer = normalizer * alpha + tl.sum(probabilities, 1)
        max_value = next_max
    output = accumulator / normalizer[:, None]
    tl.store(out + query * so_m + (group * GROUP_SIZE + offs_h[:, None]) * so_h + offs_d[None, :] * so_d,
             output, mask=(offs_h < GROUP_SIZE)[:, None])


class PagedKV:
    """Stand-in for a packed prefill K or V: the pool buffer plus its token map."""

    __slots__ = ("pool", "req_to_token", "req_indices", "output_dtype")

    def __init__(self, pool, req_to_token, req_indices, output_dtype):
        self.pool, self.req_to_token, self.req_indices, self.output_dtype = pool, req_to_token, req_indices, output_dtype


def sparse_gqa_paged(q, k: PagedKV, v: PagedKV, indices, cu_q, kv_lens, scale, config):
    total_q, num_q_heads, head_dim = q.shape
    kp, vp = k.pool, v.pool
    num_kv_heads = kp.shape[1]
    group_size = num_q_heads // num_kv_heads
    block_m = max(16, triton.next_power_of_2(group_size))
    block_n, warps, stages = config
    out = torch.empty_like(q)
    r2t = k.req_to_token
    _sparse_gqa_chunk_prefill_paged[(total_q, (cu_q.shape[0] - 1) * num_kv_heads)](
        q, kp, vp, out, indices, cu_q, kv_lens, r2t, k.req_indices, scale, indices.shape[-1],
        q.stride(0), q.stride(1), q.stride(2), kp.stride(0), kp.stride(1), kp.stride(2), vp.stride(0), vp.stride(1), vp.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        indices.stride(0), indices.stride(1) if indices.ndim == 3 else 0, indices.stride(2) if indices.ndim == 3 else indices.stride(1),
        r2t.stride(0), r2t.stride(1),
        NUM_KV_HEADS=num_kv_heads, GROUP_SIZE=group_size, BLOCK_M=block_m, BLOCK_N=block_n, HEAD_DIM=head_dim,
        num_warps=warps, num_stages=stages)
    return out
