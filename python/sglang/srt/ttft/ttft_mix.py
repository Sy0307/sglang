"""Row-split replica of GatedResidual._mix_compute (P12b), bit-identical per row.

_mix_compute is torch.compile'd: mm (cuBLAS) -> div+silu (Triton) -> mm (cuBLAS) ->
sigmoid*x mean over streams (Triton). For rows [a, b) of an M-row call this module runs
  * the two GEMMs with the cuBLASLt algorithm that reproduces torch's result at M rows
    (most of these shapes use serial split-K, so a heuristic pick at b - a rows would
    accumulate in a different order), and
  * the two pointwise kernels as written by inductor (same expressions, same Triton
    defaults), with a runtime element count.
Algorithms are matched on random data outside graph capture (prebuild) and cached by
(M, K, N); a call whose algorithm is missing returns None so the caller keeps the
unsplit path.
"""
import ctypes

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16BF = 68, 0, 14
_DESC_TRANSA, _DESC_TRANSB, _PREF_MAX_WS = 3, 4, 1
_WS_BYTES = 32 << 20


class _Algo(ctypes.Structure):
    _fields_ = [("data", ctypes.c_uint64 * 8)]


class _Heur(ctypes.Structure):
    _fields_ = [("algo", _Algo), ("workspaceSize", ctypes.c_size_t), ("state", ctypes.c_int),
                ("wavesCount", ctypes.c_float), ("reserved", ctypes.c_int * 4)]


_lt = None
_handles = {}
_ws = {}
_algos = {}
_layouts = {}


def _chk(r, what):
    if r != 0:
        raise RuntimeError(f"cuBLASLt {what} failed: status {r}")


def _lib():
    global _lt
    if _lt is None:
        torch.zeros(1, device="cuda") @ torch.zeros(1, 1, device="cuda")
        path = next((l.split()[-1] for l in open("/proc/self/maps") if "libcublasLt.so" in l), "libcublasLt.so.13")
        _lt = ctypes.CDLL(path)
    return _lt


def _handle():
    dev = torch.cuda.current_device()
    h = _handles.get(dev)
    if h is None:
        h = ctypes.c_void_p()
        _chk(_lib().cublasLtCreate(ctypes.byref(h)), "create")
        _handles[dev] = h
    return h


def _workspace(stream):
    key = (stream.device_index, stream.cuda_stream)
    w = _ws.get(key)
    if w is None:
        w = _ws[key] = torch.empty(_WS_BYTES, dtype=torch.uint8, device=f"cuda:{stream.device_index}")
    return w


def _problem(M, K, N):
    """Descriptors of D[N, M] = op_T(W[K, N]) x X[K, M] (column-major view of x @ w.T)."""
    key = (torch.cuda.current_device(), M, K, N)
    p = _layouts.get(key)
    if p is None:
        lt = _lib()
        desc = ctypes.c_void_p()
        _chk(lt.cublasLtMatmulDescCreate(ctypes.byref(desc), CUBLAS_COMPUTE_32F, CUDA_R_32F), "desc")
        t, n = ctypes.c_int(1), ctypes.c_int(0)
        _chk(lt.cublasLtMatmulDescSetAttribute(desc, _DESC_TRANSA, ctypes.byref(t), 4), "transa")
        _chk(lt.cublasLtMatmulDescSetAttribute(desc, _DESC_TRANSB, ctypes.byref(n), 4), "transb")
        a, b, c = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
        _chk(lt.cublasLtMatrixLayoutCreate(ctypes.byref(a), CUDA_R_16BF, ctypes.c_uint64(K), ctypes.c_uint64(N), ctypes.c_int64(K)), "A")
        _chk(lt.cublasLtMatrixLayoutCreate(ctypes.byref(b), CUDA_R_16BF, ctypes.c_uint64(K), ctypes.c_uint64(M), ctypes.c_int64(K)), "B")
        _chk(lt.cublasLtMatrixLayoutCreate(ctypes.byref(c), CUDA_R_16BF, ctypes.c_uint64(N), ctypes.c_uint64(M), ctypes.c_int64(N)), "C")
        p = _layouts[key] = (desc, a, b, c)
    return p


_ONE, _ZERO = ctypes.c_float(1.0), ctypes.c_float(0.0)


def _run(x, w, out, algo):
    M, K = x.shape
    N = w.shape[0]
    desc, a, b, c = _problem(M, K, N)
    stream = torch.cuda.current_stream()
    ws = _workspace(stream)
    _chk(_lib().cublasLtMatmul(_handle(), desc, ctypes.byref(_ONE), ctypes.c_void_p(w.data_ptr()), a,
                               ctypes.c_void_p(x.data_ptr()), b, ctypes.byref(_ZERO),
                               ctypes.c_void_p(out.data_ptr()), c, ctypes.c_void_p(out.data_ptr()), c,
                               ctypes.byref(algo), ctypes.c_void_p(ws.data_ptr()), ctypes.c_size_t(_WS_BYTES),
                               ctypes.c_void_p(stream.cuda_stream)), "matmul")


def algo_for(M, K, N):
    """The algorithm reproducing torch.nn.functional.linear at M rows; built only
    outside graph capture, None when unknown or unmatched."""
    key = (torch.cuda.current_device(), M, K, N)
    if key in _algos:
        return _algos[key]
    if torch.cuda.is_current_stream_capturing():
        return None
    g = torch.Generator(device="cuda").manual_seed(M * 7919 + K * 31 + N)
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda", generator=g)
    w = (torch.randn(N, K, dtype=torch.bfloat16, device="cuda", generator=g) / K ** 0.5).to(torch.bfloat16)
    ref = torch.nn.functional.linear(x, w)
    desc, a, b, c = _problem(M, K, N)
    lt = _lib()
    pref = ctypes.c_void_p()
    _chk(lt.cublasLtMatmulPreferenceCreate(ctypes.byref(pref)), "pref")
    wsb = ctypes.c_uint64(_WS_BYTES)
    _chk(lt.cublasLtMatmulPreferenceSetAttribute(pref, _PREF_MAX_WS, ctypes.byref(wsb), 8), "pref ws")
    res, got = (_Heur * 32)(), ctypes.c_int()
    _chk(lt.cublasLtMatmulAlgoGetHeuristic(_handle(), desc, a, b, c, c, pref, 32, res, ctypes.byref(got)), "heuristic")
    lt.cublasLtMatmulPreferenceDestroy(pref)
    found = None
    for i in range(got.value):
        if res[i].state != 0:
            continue
        out = torch.empty_like(ref)
        try:
            _run(x, w, out, res[i].algo)
        except RuntimeError:
            continue
        if torch.equal(out, ref):
            found = _Algo()
            ctypes.memmove(ctypes.byref(found), ctypes.byref(res[i].algo), ctypes.sizeof(_Algo))
            break
    torch.cuda.synchronize()
    _algos[key] = found
    return found


def linear_rows(x, w, out, full_rows):
    """out[r] = (x @ w.T)[r] for the rows of x, as torch computes them in a full_rows-row call."""
    algo = algo_for(full_rows, x.shape[1], w.shape[0])
    if algo is None:
        return False
    _run(x, w, out, algo)
    return True


@triton.jit
def _silu_quarter_kernel(in_out_ptr0, xnumel, XBLOCK: tl.constexpr):
    # inductor's triton_poi_fused_div_silu_0 with a runtime element count
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = tl.full([1], 0.25, tl.float32)
    tmp2 = tmp0 * tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp4 = -tmp3
    tmp5 = libdevice.exp(tmp4)
    tmp6 = tl.full([1], 1.0, tl.float32)
    tmp7 = tmp5 + tmp6
    tmp8 = (tmp3 / tmp7)
    tmp9 = tmp8.to(tl.float32)
    tl.store(in_out_ptr0 + (x0), tmp9, xmask)


@triton.jit
def _gate_mean_kernel(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    # inductor's triton_poi_fused_mean_mul_sigmoid_view_1 (hc_count 4, hidden 2560)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 2560)
    x1 = xindex // 2560
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 10240 * x1), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0 + 10240 * x1), xmask).to(tl.float32)
    tmp5 = tl.load(in_ptr0 + (2560 + x0 + 10240 * x1), xmask).to(tl.float32)
    tmp7 = tl.load(in_ptr1 + (2560 + x0 + 10240 * x1), xmask).to(tl.float32)
    tmp11 = tl.load(in_ptr0 + (5120 + x0 + 10240 * x1), xmask).to(tl.float32)
    tmp13 = tl.load(in_ptr1 + (5120 + x0 + 10240 * x1), xmask).to(tl.float32)
    tmp17 = tl.load(in_ptr0 + (7680 + x0 + 10240 * x1), xmask).to(tl.float32)
    tmp19 = tl.load(in_ptr1 + (7680 + x0 + 10240 * x1), xmask).to(tl.float32)
    tmp1 = tl.sigmoid(tmp0)
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp6 = tl.sigmoid(tmp5)
    tmp8 = tmp6 * tmp7
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp4 + tmp9
    tmp12 = tl.sigmoid(tmp11)
    tmp14 = tmp12 * tmp13
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp10 + tmp15
    tmp18 = tl.sigmoid(tmp17)
    tmp20 = tmp18 * tmp19
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp16 + tmp21
    tmp23 = tl.full([1], 4.0, tl.float32)
    tmp24 = (tmp22 / tmp23)
    tmp25 = tmp24.to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp25, xmask)


def supported(hc, normed):
    return (hc.hc_count == 4 and hc.hidden_size == 2560 and normed.dtype == torch.bfloat16
            and hc.input_mix_weight_down.weight.dtype == torch.bfloat16
            and tuple(hc.input_mix_weight_down.weight.shape) == (hc.config.hc_lowrank, 10240)
            and tuple(hc.input_mix_weight_up.weight.shape) == (10240, hc.config.hc_lowrank))


def mix_rows(hc, normed_rows, out_rows, low_buf, up_buf, full_rows):
    """GatedResidual mix of normed_rows (rows of a full_rows-row call) into out_rows.
    low_buf / up_buf: [rows, lowrank] / [rows, 10240] bf16 scratch. False if an algorithm is missing."""
    rows = normed_rows.shape[0]
    if not linear_rows(normed_rows, hc.input_mix_weight_down.weight, low_buf, full_rows):
        return False
    n = low_buf.numel()
    _silu_quarter_kernel[(triton.cdiv(n, 1024),)](low_buf, n, XBLOCK=1024)
    if not linear_rows(low_buf, hc.input_mix_weight_up.weight, up_buf, full_rows):
        return False
    n = rows * 2560
    _gate_mean_kernel[(triton.cdiv(n, 1024),)](up_buf, normed_rows, out_rows, n, XBLOCK=1024)
    return True


def prebuild(full_rows, lowrank=320):
    """Match the mix GEMM algorithms for a graph bucket before it is captured."""
    if full_rows >= 2 and not torch.cuda.is_current_stream_capturing():
        algo_for(full_rows, 10240, lowrank)
        algo_for(full_rows, lowrank, 10240)
