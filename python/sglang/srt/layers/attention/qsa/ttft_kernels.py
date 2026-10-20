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
