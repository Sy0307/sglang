"""TTFT patches for Qwen3.8-Flash-Next (enabled by SGLANG_TTFT_PATCH).

Vendored in this branch: importing this module installs the patch finder.
P1: QSA sparse prefill sizes its grid from the host-known q.shape[0] instead of a
    device max() + .item(); the kernel returns early for rows past each request's
    q_end, so the result is unchanged and the per-layer GPU->CPU sync disappears.
P2: the /generate contract check validates input_ids in C (all exact ints) before
    falling back to the per-element Python check (7.8 ms -> <1 ms for 33K ids).
P3: Starlette parses request bodies with orjson, falling back to json.loads.
P10: prefill BF16 GEMMs of shapes where cuBLASLt beats cuBLAS on SM120 (1024x3072->2560
     out/o_proj, 2560->6656 qkv+gate) run with cuBLASLt preferred; other shapes unchanged.
P9: QSA sparse prefill with a cached prefix reads K/V straight from the paged pool
    (ttft_kernels.sparse_gqa_paged) instead of packing them first; bit-identical.
P8: QSA indexer prefill scoring runs one Triton kernel (ttft_kernels.py) that splits
    keys across CTAs and fuses scale + mask; bit-identical to the TileLang path for
    bf16 with 4 heads (same dot order, (h0+h1)+(h2+h3) head sum), else falls back.
P4: the overlap event loop processes a pending prefill result before forming the
    next batch (the normal loop's order), so the first token is not held behind
    the next decode launch; decode-after-decode iterations still overlap.
"""
import __future__
import importlib.abc
import importlib.util
import inspect
import os
import sys

# Patched functions are recompiled from source; annotations stay unevaluated,
# as under the modules' own `from __future__ import annotations`.
_LAZY_ANN = __future__.annotations.compiler_flag

_TRACE_FILE = None


def _tr(name):
    """Diagnostics only (patch "trace"): append `name,perf_counter` to a per-process file."""
    global _TRACE_FILE
    if _TRACE_FILE is None:
        import time as _time

        _TRACE_FILE = (open(f"{os.environ['TTFT_TRACE_DIR']}/ev-{os.getpid()}.csv", "a", buffering=1), _time.perf_counter)
    _TRACE_FILE[0].write(f"{name},{_TRACE_FILE[1]():.6f}\n")


_PATCHES = set(filter(None, os.environ.get("SGLANG_TTFT_PATCH", "").split(",")))


def _patch_sparse_attn(mod):
    src = inspect.getsource(mod.sparse_gqa_fwd_interface_triton_ck)
    old = "max_q = int((cu_q[1:] - cu_q[:-1]).max().item())"
    assert src.count(old) == 1, "sparse_attn source changed; refusing to patch"
    if "p7" in _PATCHES:
        # P7: the stock table matches only H20 and otherwise uses the L20 table, whose
        # large-batch entry is (BLOCK_N=16, warps=1, stages=2); on SM120 (16, 2, 3) is
        # ~18% faster for 1K-query prefill (same math, different MMA accumulation split).
        def _ttft_best_config(total_q, _orig=mod._get_best_config):
            import torch

            if total_q > 512 and torch.cuda.get_device_capability()[0] == 12:
                return (16, 2, 3)
            return _orig(total_q)

        mod._ttft_best_config = _ttft_best_config
        cfg_old = "block_n, warps, stages = _get_best_config(total_q)"
        assert src.count(cfg_old) == 1, "sparse_attn config call changed; refusing to patch"
        src = src.replace(cfg_old, "block_n, warps, stages = _ttft_best_config(total_q)")
    exec(compile(src.replace(old, "max_q = total_q"), mod.__file__, "exec", _LAZY_ANN, True), mod.__dict__)
    print(f"[ttft-patch] P1 sparse_attn max_q host-side (pid {os.getpid()})", file=sys.stderr, flush=True)


def _patch_api_types(mod):
    slow = mod.decode_TokenIds
    expect_seq = mod._expect_seq

    def decode_TokenIds(value):
        # JSON decoding yields exact ints for integer literals; any other element
        # type (bool, float, str, ...) takes the original per-element path, so
        # accepted values and error texts are unchanged.
        seq = expect_seq(value, expected="a sequence")
        if set(map(type, seq)) <= {int}:
            return list(seq)
        return slow(seq)

    mod.decode_TokenIds = decode_TokenIds
    print(f"[ttft-patch] P2 api_types decode_TokenIds fast path (pid {os.getpid()})", file=sys.stderr, flush=True)


def _patch_starlette_requests(mod):
    import json

    import orjson

    # orjson turns integers past 64 bits into floats, so any 19+ digit run keeps
    # the stdlib parser; up to 18 digits both parsers agree. translate + find
    # stay in C (a regex scan of a 200 KB body costs ~4 ms).
    digit_mask = bytes(0x64 if 0x30 <= b <= 0x39 else 0x20 for b in range(256))
    long_run = b"d" * 19

    trace = "trace" in _PATCHES

    async def json_(self):
        if not hasattr(self, "_json"):
            body = await self.body()
            if trace:
                _tr("http_body")
            if long_run in body.translate(digit_mask):
                self._json = json.loads(body)
            else:
                try:
                    self._json = orjson.loads(body)
                except orjson.JSONDecodeError:
                    # NaN/Infinity, lone surrogates, non-UTF-8: json.loads
                    # keeps its own behaviour (result or error) for those.
                    self._json = json.loads(body)
            if trace:
                _tr("http_json_done")
        return self._json

    mod.Request.json = json_
    print(f"[ttft-patch] P3 starlette Request.json via orjson (pid {os.getpid()})", file=sys.stderr, flush=True)


def _patch_scheduler(mod):
    import textwrap

    src = inspect.getsource(mod.Scheduler.event_loop_overlap)
    edits = [
        (
            "            # Get the next batch to run\n            plan = self.get_next_batch_to_run(",
            "            # [ttft-patch P4] Hand a prefill result out before forming the\n"
            "            # next batch, in the normal loop's order; decode steps keep overlap.\n"
            "            early = False\n"
            "            if (\n"
            "                self.last_batch is not None\n"
            "                and self.result_queue\n"
            "                and self.last_batch.forward_mode.is_extend()\n"
            "                and not self.require_mlp_sync\n"
            "                and self.last_batch.spec_algorithm.is_none()\n"
            "            ):\n"
            "                pop_and_process()\n"
            "                early = True\n\n"
            "            # Get the next batch to run\n            plan = self.get_next_batch_to_run(",
        ),
        ("            if disable_overlap_for_batch:\n                pop_and_process()",
         "            if disable_overlap_for_batch and not early:\n                pop_and_process()"),
        ("                if not disable_overlap_for_batch:\n                    pop_and_process()",
         "                if not disable_overlap_for_batch and not early:\n                    pop_and_process()"),
    ]
    for old, new in edits:
        assert src.count(old) == 1, f"event_loop_overlap source changed; refusing to patch: {old[:50]!r}"
        src = src.replace(old, new)
    src = textwrap.dedent(src)
    ns = {}
    exec(compile(src, mod.__file__, "exec", _LAZY_ANN, True), mod.__dict__, ns)
    mod.Scheduler.event_loop_overlap = ns["event_loop_overlap"]
    print(f"[ttft-patch] P4 overlap loop hands prefill results out first (pid {os.getpid()})", file=sys.stderr, flush=True)


def _patch_batch_result_processor(mod):
    import textwrap

    src = inspect.getsource(mod.SchedulerBatchResultProcessor.process_batch_result_prefill)
    edits = [
        (
            "        skip_stream_req = None\n",
            "        skip_stream_req = None\n"
            "        # [ttft-patch P5] radix-tree checkpoints of still-running requests\n"
            "        # run after their first token is streamed.\n"
            "        deferred_checkpoints = []\n",
        ),
        (
            "                    elif not batch.decoding_reqs or req not in batch.decoding_reqs:\n"
            "                        checkpoint_kv_cache(req, self.tree_cache)\n"
            "                        if get_memory().enable_hisparse:\n"
            "                            self.hisparse_coordinator.admit_request_into_staging(req)\n",
            "                    elif not batch.decoding_reqs or req not in batch.decoding_reqs:\n"
            "                        deferred_checkpoints.append(req)\n",
        ),
        (
            "        self.output_streamer.stream_output(\n"
            "            batch.reqs, batch.return_logprob, skip_stream_req\n"
            "        )\n",
            "        self.output_streamer.stream_output(\n"
            "            batch.reqs, batch.return_logprob, skip_stream_req\n"
            "        )\n"
            "        if deferred_checkpoints:\n"
            "            self.token_to_kv_pool_allocator.free_group_begin()\n"
            "            for req in deferred_checkpoints:\n"
            "                checkpoint_kv_cache(req, self.tree_cache)\n"
            "                if get_memory().enable_hisparse:\n"
            "                    self.hisparse_coordinator.admit_request_into_staging(req)\n"
            "            self.token_to_kv_pool_allocator.free_group_end()\n",
        ),
    ]
    for old, new in edits:
        assert src.count(old) == 1, f"process_batch_result_prefill source changed; refusing to patch: {old[:60]!r}"
        src = src.replace(old, new)
    ns = {}
    exec(compile(textwrap.dedent(src), mod.__file__, "exec", _LAZY_ANN, True), mod.__dict__, ns)
    mod.SchedulerBatchResultProcessor.process_batch_result_prefill = ns["process_batch_result_prefill"]
    print(f"[ttft-patch] P5 prefill checkpoint after first-token stream (pid {os.getpid()})", file=sys.stderr, flush=True)


def _patch_common(mod):
    import numpy as np
    import torch

    def flatten_arrays_to_pinned_cpu(parts, pin):
        combined = np.concatenate([np.frombuffer(p, dtype=np.int64) for p in parts])
        if not pin:
            return torch.from_numpy(combined)
        # A pinned block from the caching host allocator, filled in place: the
        # same tensor as from_numpy(...).pin_memory() without its pointer query.
        out = torch.empty(combined.shape[0], dtype=torch.int64, pin_memory=True)
        out.numpy()[:] = combined
        return out

    mod.flatten_arrays_to_pinned_cpu = flatten_arrays_to_pinned_cpu
    print(f"[ttft-patch] P6 pinned flatten via caching host allocator (pid {os.getpid()})", file=sys.stderr, flush=True)


def _patch_mqa(mod):
    import torch

    from sglang.srt.ttft.ttft_kernels import triton_qsa_mqa_prefill

    orig = mod.tilelang_qsa_mqa_prefill

    def tilelang_qsa_mqa_prefill(q, k, row_starts, row_ends, score_scale=None):
        if (
            q.is_cuda
            and q.dtype == torch.bfloat16
            and k.dtype == torch.bfloat16
            and q.ndim == 3
            and q.shape[1] == 4
            and q.shape[0] > 0
            and k.shape[0] > 0
        ):
            mod._validate_q(q)
            mod._validate_k(k)
            return triton_qsa_mqa_prefill(q, k, row_starts, row_ends, score_scale, KQ=True, SUM=1)
        return orig(q, k, row_starts, row_ends, score_scale)

    mod.tilelang_qsa_mqa_prefill = tilelang_qsa_mqa_prefill
    print(f"[ttft-patch] P8 QSA indexer prefill scoring via Triton (pid {os.getpid()})", file=sys.stderr, flush=True)


def _patch_sparse_attn_dump(mod):
    """Diagnostics only: save the first 12 large prefill selections of this process."""
    import torch

    inner = mod.sparse_gqa_fwd_interface_triton_ck
    out_dir = os.environ["TTFT_DUMP_DIR"]
    state = {"n": 0}

    def sparse_gqa_fwd_interface_triton_ck(q, k, v, indices, cu_q, cu_k, kv_lens, scale):
        if state["n"] < 12 and q.shape[0] >= 1024 and k.shape[0] >= 16384:
            torch.save(
                {"indices": indices.cpu(), "cu_q": cu_q.cpu(), "cu_k": cu_k.cpu(), "kv_lens": kv_lens.cpu(),
                 "q_shape": tuple(q.shape), "k_shape": tuple(k.shape)},
                f"{out_dir}/sel-{os.getpid()}-{state['n']}.pt",
            )
            state["n"] += 1
        return inner(q, k, v, indices, cu_q, cu_k, kv_lens, scale)

    mod.sparse_gqa_fwd_interface_triton_ck = sparse_gqa_fwd_interface_triton_ck


def _wrap_trace(cls_or_mod, attr, before=None, after=None, cond=None, is_async=False):
    fn = getattr(cls_or_mod, attr)
    if is_async:
        async def wrapped(*a, **kw):
            if before and (cond is None or cond(a, kw, None)):
                _tr(before)
            r = await fn(*a, **kw)
            if after and (cond is None or cond(a, kw, r)):
                _tr(after)
            return r
    else:
        def wrapped(*a, **kw):
            if before and (cond is None or cond(a, kw, None)):
                _tr(before)
            r = fn(*a, **kw)
            if after and (cond is None or cond(a, kw, r)):
                _tr(after)
            return r
    setattr(cls_or_mod, attr, wrapped)


def _trace_http_server(mod):
    _wrap_trace(mod, "serve_generate_request", before="serve_start", is_async=True)
    _wrap_trace(mod.ORJSONRequest, "json", before="http_json_start", after="http_json_end", is_async=True)
    _wrap_trace(mod, "generate_contract_error", before="contract_start", after="contract_end")
    _wrap_trace(mod, "dumps_json", after="sse_dumped")


def _trace_tokenizer_manager(mod):
    _wrap_trace(mod.TokenizerManager, "_dispatch_to_scheduler", before="tm_dispatch_start", after="tm_dispatch_end")
    _wrap_trace(mod.TokenizerManager, "_tokenize_one_request", before="tm_tokenize_start", after="tm_tokenize_end", is_async=True)
    _wrap_trace(mod.TokenizerManager, "_create_tokenized_object", before="tm_create_start", after="tm_create_end")
    _wrap_trace(mod.TokenizerManager, "_send_one_request", before="tm_send_start", is_async=True)
    _wrap_trace(mod.TokenizerManager, "_handle_batch_output", before="tm_out", is_async=True)


def _trace_request_receiver(mod):
    _wrap_trace(mod.SchedulerRequestReceiver, "recv_requests", after="sched_recv", cond=lambda a, kw, r: bool(r))


def _trace_scheduler(mod):
    def is_extend_plan(a, kw, r):
        b = getattr(r, "batch_to_run", None)
        return b is not None and b.forward_mode.is_extend()

    def is_extend_arg(a, kw, r):
        return a[1].forward_mode.is_extend()

    _wrap_trace(mod.Scheduler, "get_next_batch_to_run", after="gnb_done", cond=is_extend_plan)
    _wrap_trace(mod.Scheduler, "run_batch", before="run_start", after="run_end", cond=is_extend_arg)


def _trace_batch_result_processor(mod):
    _wrap_trace(mod.SchedulerBatchResultProcessor, "process_batch_result_prefill", before="pbr_start", after="pbr_end")


def _trace_output_streamer(mod):
    _wrap_trace(mod.SchedulerOutputStreamer, "stream_output", before="stream_out")


def _trace_detokenizer(mod):
    _wrap_trace(mod.DetokenizerManager, "handle_batch_token_id_out", before="detok_in", after="detok_out")


def _patch_qsa_backend(mod):
    import torch

    from sglang.srt.layers.attention.qsa import sparse_attn as sa
    from sglang.srt.ttft.ttft_kernels import PagedKV, sparse_gqa_paged

    orig_pack = mod.pack_qsa_prefill_kv
    orig_sparse = mod.sparse_gqa_fwd_interface_triton_ck

    def pack_qsa_prefill_kv(*, k, v, req_to_token, req_indices, cu_k, total_k, max_k, output_dtype=None):
        if k.is_cuda and k.dtype == v.dtype and (output_dtype is None or output_dtype == k.dtype):
            return (PagedKV(k, req_to_token, req_indices, output_dtype), PagedKV(v, req_to_token, req_indices, output_dtype))
        return orig_pack(k=k, v=v, req_to_token=req_to_token, req_indices=req_indices, cu_k=cu_k,
                         total_k=total_k, max_k=max_k, output_dtype=output_dtype)

    def sparse_gqa_fwd_interface_triton_ck(q, k, v, indices, cu_q, cu_k, kv_lens, scale):
        if isinstance(k, PagedKV):
            pick = getattr(sa, "_ttft_best_config", sa._get_best_config)
            return sparse_gqa_paged(q, k, v, indices, cu_q, kv_lens, scale, pick(q.shape[0]))
        return orig_sparse(q, k, v, indices, cu_q, cu_k, kv_lens, scale)

    mod.pack_qsa_prefill_kv = pack_qsa_prefill_kv
    mod.sparse_gqa_fwd_interface_triton_ck = sparse_gqa_fwd_interface_triton_ck
    print(f"[ttft-patch] P9 QSA prefill reads paged K/V directly (pid {os.getpid()})", file=sys.stderr, flush=True)


def _patch_unquant(mod):
    import torch
    import torch.nn.functional as F

    lt_shapes = {(2560, 3072), (6656, 2560)}
    orig_apply = mod.UnquantizedLinearMethod.apply
    set_lib = torch.backends.cuda.preferred_blas_library

    def apply(self, layer, x, bias=None):
        w = layer.weight
        if (
            x.is_cuda
            and x.ndim == 2
            and x.shape[0] >= 256
            and x.dtype == torch.bfloat16
            and w.dtype == torch.bfloat16
            and tuple(w.shape) in lt_shapes
            and type(w.data) is torch.Tensor
        ):
            prev = set_lib()
            set_lib("cublaslt")
            try:
                return F.linear(x, w, bias)
            finally:
                set_lib(prev)
        return orig_apply(self, layer, x, bias)

    mod.UnquantizedLinearMethod.apply = apply
    print(f"[ttft-patch] P10 cuBLASLt for selected prefill BF16 GEMMs (pid {os.getpid()})", file=sys.stderr, flush=True)


def _rewrite_method(mod, cls, name, edits, tag):
    """Recompile cls.name from its source with exact one-shot text edits."""
    import textwrap

    src = inspect.getsource(getattr(cls, name))
    for old, new in edits:
        assert src.count(old) == 1, f"{tag}: {cls.__name__}.{name} source changed; refusing to patch: {old[:60]!r}"
        src = src.replace(old, new)
    ns = {}
    exec(compile(textwrap.dedent(src), mod.__file__, "exec", _LAZY_ANN, True), mod.__dict__, ns)
    setattr(cls, name, ns[name])


def _patch_qsa_backend_syncs(mod):
    """P11 (QSA): the two host syncs per QSA prefill forward that P1/P9 leave.

    kv_lens of the sparse kernel came from a pageable torch.tensor(..., device=)
    each layer (a blocking H2D); it is now staged once per distinct length tuple
    from pinned memory. The extend row map's repeat_interleave gets its known
    output size, so it no longer reads the total back from the GPU.
    """
    import collections

    import torch

    cache = collections.OrderedDict()

    def _ttft_seq_lens_tensor(sequence_lens, device):
        key = (tuple(sequence_lens), device)
        t = cache.get(key)
        if t is None:
            t = torch.tensor(sequence_lens, dtype=torch.int32, pin_memory=True).to(device, non_blocking=True)
            cache[key] = t
            if len(cache) > 64:
                cache.popitem(last=False)
        return t

    def _ttft_extend_total(forward_batch):
        lens = forward_batch.extend_seq_lens_cpu
        return None if lens is None else int(sum(lens))

    mod._ttft_seq_lens_tensor = _ttft_seq_lens_tensor
    mod._ttft_extend_total = _ttft_extend_total
    cls = mod.QwenSparseAttnBackend
    _rewrite_method(mod, cls, "forward_extend", [(
        "        sequence_lens_tensor = torch.tensor(\n"
        "            sequence_lens, dtype=torch.int32, device=q.device\n"
        "        )\n",
        "        sequence_lens_tensor = _ttft_seq_lens_tensor(sequence_lens, q.device)\n",
    )], "P11")
    if "p14" in _PATCHES:
        # P14 builds the row map on the host (see _patch_qsa_host_plan); both edits go in
        # one rewrite because a recompiled method's source cannot be read back.
        row_map = [(
            "                token_to_batch_idx = torch.repeat_interleave(\n"
            "                    torch.arange(\n"
            "                        batch_size,\n"
            "                        device=sequence_lengths.device,\n"
            "                        dtype=torch.int32,\n"
            "                    ),\n"
            "                    extend_seq_lens.to(torch.long),\n"
            "                )\n",
            "                token_to_batch_idx = _ttft_token_to_batch_idx(\n"
            "                    forward_batch, batch_size, sequence_lengths.device, extend_seq_lens\n"
            "                )\n",
        )]
    else:
        row_map = [(
            "                    extend_seq_lens.to(torch.long),\n"
            "                )\n",
            "                    extend_seq_lens.to(torch.long),\n"
            "                    output_size=_ttft_extend_total(forward_batch),\n"
            "                )\n",
        )]
    _rewrite_method(mod, cls, "_metadata_from_forward_batch", row_map, "P11")
    print(f"[ttft-patch] P11 QSA kv_lens staged, row map sized on host (pid {os.getpid()})", file=sys.stderr, flush=True)


def _patch_qsa_host_plan(mod):
    """P14: the QSA extend write plan and token->request map from host lengths.

    For an extend batch whose lengths are on the host, the ~25 small device ops of
    _qsa_write_plan (cumsum, searchsorted, where, gathers) and the repeat_interleave
    become the same integer arithmetic in numpy, one pinned copy, and the single
    token_slot_table gather on the device. The first calls are checked element by
    element against the original and a mismatch raises.
    """
    import numpy as np
    import torch

    cls = mod.QwenSparseAttnBackend
    orig_plan = cls._qsa_build_write_plan
    checks = [3]

    def _host_lists(forward_batch, n):
        lens, ext = forward_batch.seq_lens_cpu, forward_batch.extend_seq_lens_cpu
        if lens is None or ext is None or len(ext) < n:
            return None
        lens = np.asarray(lens[:n].tolist() if hasattr(lens, "tolist") else lens[:n], dtype=np.int64)
        ext = np.asarray(list(ext)[:n], dtype=np.int64)
        return lens, ext

    def _to_dev(arrs, device):
        host = torch.from_numpy(np.stack(arrs)).pin_memory()
        return host.to(device, non_blocking=True)

    def _qsa_build_write_plan(self, *, forward_batch, speculative_paged, token_slot_table, sequence_lengths):
        n = sequence_lengths.numel()
        hl = None
        if not (forward_batch.forward_mode.is_decode() or speculative_paged) and forward_batch.extend_seq_lens is not None:
            hl = _host_lists(forward_batch, n)
        if hl is None or n == 0:
            return orig_plan(self, forward_batch=forward_batch, speculative_paged=speculative_paged,
                             token_slot_table=token_slot_table, sequence_lengths=sequence_lengths)
        lens, ext = hl
        ratio = self.token_to_kv_pool.qsa_compress_ratio
        end_blocks = lens // ratio
        prefix = np.maximum(lens - ext, 0)
        if (prefix % ratio).any() or (end_blocks * ratio > token_slot_table.shape[1]).any():
            return orig_plan(self, forward_batch=forward_batch, speculative_paged=speculative_paged,
                             token_slot_table=token_slot_table, sequence_lengths=sequence_lengths)
        start_blocks = prefix // ratio
        capacity = int(forward_batch.input_ids.numel()) // ratio + n
        counts = np.maximum(end_blocks - start_blocks, 0)
        ends = np.cumsum(counts)
        starts = ends - counts
        entries = np.arange(capacity, dtype=np.int64)
        rows = np.searchsorted(ends, entries, side="right")
        valid = entries < ends[-1]
        rows = np.where(valid, np.minimum(rows, max(n - 1, 0)), 0)
        blocks = np.where(valid, start_blocks[rows] + entries - starts[rows], 0)
        row_token_starts = np.cumsum(ext) - ext
        member_rows = np.where(valid, row_token_starts[rows] + blocks * ratio - prefix[rows], 0)
        dev = _to_dev([rows, blocks * ratio, valid.astype(np.int64), blocks * ratio + (ratio - 1), member_rows],
                      token_slot_table.device)
        rows_d, cols_d, valid_d, group_end_positions, member_rows_d = dev.unbind(0)
        write_locs = torch.where(
            valid_d.bool(),
            token_slot_table[rows_d, cols_d].long() // ratio,
            torch.zeros_like(cols_d),
        ).to(torch.int32)
        out = (write_locs, group_end_positions, rows_d, member_rows_d)
        if checks[0] > 0 and not torch.cuda.is_current_stream_capturing():
            checks[0] -= 1
            ref = orig_plan(self, forward_batch=forward_batch, speculative_paged=speculative_paged,
                            token_slot_table=token_slot_table, sequence_lengths=sequence_lengths)
            for name, a, b in zip(("write_locs", "group_positions", "rows", "member_rows"), out, ref):
                if a.dtype != b.dtype or not torch.equal(a, b):
                    raise RuntimeError(f"P14 host write plan mismatch in {name}")
            print(f"[ttft-patch] P14 host write plan verified against the original (pid {os.getpid()})", file=sys.stderr, flush=True)
        return out

    def _ttft_token_to_batch_idx(forward_batch, batch_size, device, extend_seq_lens):
        hl = _host_lists(forward_batch, batch_size)
        if hl is None:
            return torch.repeat_interleave(
                torch.arange(batch_size, device=device, dtype=torch.int32),
                extend_seq_lens.to(torch.long),
            )
        idx = np.repeat(np.arange(batch_size, dtype=np.int32), hl[1])
        return torch.from_numpy(idx).pin_memory().to(device, non_blocking=True)

    # The call site is rewritten by P11's single edit of _metadata_from_forward_batch.
    mod._ttft_token_to_batch_idx = _ttft_token_to_batch_idx
    cls._qsa_build_write_plan = _qsa_build_write_plan
    print(f"[ttft-patch] P14 QSA extend write plan and row map from host lengths (pid {os.getpid()})", file=sys.stderr, flush=True)


def _patch_scheduler_idle(mod):
    """P15: full idle housekeeping on the first idle iteration and then every Nth.

    A fully idle scheduler spins through on_idle, which runs pool/tree invariant checks
    and a forced load publish on every iteration; a request that lands mid-iteration
    waits for it (and the peer rank for the next broadcast). Consecutive idle
    iterations now run only the cheap part. The count is per loop iteration, which
    the TP ranks share (each iteration broadcasts), so they stay in step.
    """
    orig = mod.Scheduler.on_idle
    every = max(1, int(os.environ.get("SGLANG_TTFT_IDLE_EVERY", "64")))

    def on_idle(self):
        idle = self.is_fully_idle()
        n = getattr(self, "_ttft_idle_n", 0) if idle else 0
        self._ttft_idle_n = n + 1 if idle else 0
        if not idle or n % every == 0:
            return orig(self)
        self.dp_attn_adapter.drop_sync_wait_carry()
        self.maybe_send_health_check_signal()
        self.metrics_reporter.record_scheduler_idle()
        self.kv_events_publisher.publish_kv_events()
        self.maybe_sleep_on_idle()
        self.metrics_reporter.record_scheduler_idle()

    mod.Scheduler.on_idle = on_idle
    print(f"[ttft-patch] P15 full idle housekeeping every {every} idle iterations (pid {os.getpid()})", file=sys.stderr, flush=True)


def _patch_hybrid_backend_syncs(mod):
    """P11 (GDN): tag the extend query_start_loc with its host lengths so FLA's
    chunk index table (first GDN layer of every forward) is built without a
    .tolist() of the device tensor."""
    _rewrite_method(mod, mod.MambaAttnBackendBase, "_forward_metadata", [(
        "                    logical_num_tokens = sum(forward_batch.extend_seq_lens_cpu)\n",
        "                    logical_num_tokens = sum(forward_batch.extend_seq_lens_cpu)\n"
        "                    query_start_loc._ttft_cpu_lens = tuple(\n"
        "                        int(x) for x in forward_batch.extend_seq_lens_cpu\n"
        "                    )\n",
    )], "P11")
    print(f"[ttft-patch] P11 GDN query_start_loc carries host lengths (pid {os.getpid()})", file=sys.stderr, flush=True)


def _patch_fla_index(mod):
    """P11 (FLA): prepare_chunk_indices from host lengths when the caller has them.

    Same values as the original (int64 arange per sequence, then the cast to
    cu_seqlens' dtype); only the source of the lengths changes.
    """
    import torch

    orig = mod.prepare_chunk_indices
    last = []  # [(cu_seqlens, chunk_size, result)], one entry: the same tensor is passed by every GDN layer

    def prepare_chunk_indices(cu_seqlens, chunk_size):
        lens = getattr(cu_seqlens, "_ttft_cpu_lens", None)
        if lens is None or not cu_seqlens.is_cuda:
            return orig(cu_seqlens, chunk_size)
        for t, c, r in last:
            if t is cu_seqlens and c == chunk_size:
                return r
        if not last:
            print(f"[ttft-patch] P11 FLA host-length path active (pid {os.getpid()})", file=sys.stderr, flush=True)
        indices = torch.cat([torch.arange((n + chunk_size - 1) // chunk_size) for n in lens])
        host = torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens.dtype).pin_memory()
        r = host.to(cu_seqlens.device, non_blocking=True)
        last[:] = [e for e in last if e[0] is cu_seqlens][-3:] + [(cu_seqlens, chunk_size, r)]
        return r

    mod.prepare_chunk_indices = prepare_chunk_indices
    print(f"[ttft-patch] P11 FLA chunk indices from host lengths (pid {os.getpid()})", file=sys.stderr, flush=True)


def _patch_qsa_metadata(mod):
    """P13: QSA prefill scoring inputs, layer-independent part built once per forward.

    get_prefill_mqa_inputs recomputed the compressed-slot index lists and the
    per-row key ranges (~20 small ops) in every QSA layer although they depend
    only on the batch; only the gather from the layer's own compressed-key
    buffer differs per layer. The cached part is keyed on the identity and
    version of every input tensor, and the ops are the original ones.
    """
    import torch

    cls = mod.QSAIndexerMetadata
    orig = cls.get_prefill_mqa_inputs
    last = []  # [(key, inputs kept alive, value)]
    hits = []

    def _k(t):
        return (id(t), t.data_ptr(), t._version)

    def get_prefill_mqa_inputs(self, layer_id, positions, query_sequence_ids=None):
        qids = self.token_to_batch_idx if query_sequence_ids is None else query_sequence_ids
        lens_list = self.prefill_sequence_lengths_cpu
        if (lens_list is None or qids is None or not positions.is_cuda
                or positions.numel() < qids.numel()):
            return orig(self, layer_id, positions, query_sequence_ids)
        # positions arrives as a fresh view per layer: key it by storage, not identity.
        pkey = (positions.data_ptr(), tuple(positions.shape), tuple(positions.stride()), positions._version)
        key = (_k(self.sequence_lengths), _k(self.token_slot_table), _k(qids), pkey,
               tuple(int(x) for x in lens_list), self.compress_ratio)
        if last and last[0][0] == key:
            locs, row_starts, row_ends, sequence_lengths = last[0][2]
            if not hits:
                print(f"[ttft-patch] P13 cache hit at layer {layer_id} (pid {os.getpid()})", file=sys.stderr, flush=True)
            hits.append(1) if not hits else None
        else:
            ratio = self.compress_ratio
            sequence_lengths = self.sequence_lengths.to(torch.int32)
            locs = []
            for sequence_id in range(len(lens_list)):
                complete_blocks = int(lens_list[sequence_id]) // ratio
                if complete_blocks == 0:
                    continue
                locs.append(self.token_slot_table[sequence_id, : complete_blocks * ratio : ratio].long() // ratio)
            row_starts, row_ends, _ = mod.build_qsa_row_ranges(
                sequence_lengths,
                positions[: qids.numel()].to(sequence_lengths.device),
                qids.to(sequence_lengths.device),
                ratio,
            )
            last[:] = [(key, (self.sequence_lengths, self.token_slot_table, qids, positions),
                        (locs, row_starts, row_ends, sequence_lengths))]
        pool = self.token_to_kv_pool
        compressed_buffer = pool.get_qsa_compressed_k_buffer(layer_id)
        parts = [compressed_buffer.index_select(0, l) for l in locs]
        compressed_keys = (
            torch.cat(parts, dim=0)
            if parts
            else compressed_buffer.new_empty((0, pool.qsa_index_kv_heads, pool.qsa_index_head_dim))
        )
        return compressed_keys, row_starts, row_ends, sequence_lengths

    cls.get_prefill_mqa_inputs = get_prefill_mqa_inputs
    print(f"[ttft-patch] P13 QSA prefill scoring inputs shared across layers (pid {os.getpid()})", file=sys.stderr, flush=True)


# ---- P12: token-split boundary pipeline --------------------------------------
# After an attention (H1) and after the MoE (H2) the layer completes a TP sum,
# writes it into the hyper-connection streams (hc_combine) and, after an
# attention, normalizes the streams for the FFN's mix. All three are row-wise.
# P12 sums the first half of the rows, starts that half's combine (+ norm) on a
# side stream, and sums the second half meanwhile; the GEMMs that follow still
# run once over all rows, so every kernel sees the same values per row.
_P12_STREAMS = {}
_P12_SEEN = set()
_P12_LAYERS = {}  # layer_id -> decoder layer, filled by the attention read (P12c)
_P12_PRE = []  # [streams, layer, normed, mixed or None]: P12c's read of the next layer
_P12_MIN_ROWS = int(os.environ.get("SGLANG_TTFT_SPLIT_MIN_ROWS", "512"))
_P12_ORDER = os.environ.get("SGLANG_TTFT_P12_ORDER", "ar_first")
# Side-stream row work runs in this many sequential row chunks: one hc_combine CTA per
# row would otherwise fill every SM and hold the concurrent all-reduce off them (~17 us).
_P12_CHUNKS = int(os.environ.get("SGLANG_TTFT_P12_CHUNKS", "4"))


def _p12_note(tag, m, h):
    if tag not in _P12_SEEN:
        _P12_SEEN.add(tag)
        print(f"[ttft-patch] P12 {tag} split path active: rows {m} -> {h} + {m - h} (pid {os.getpid()})", file=sys.stderr, flush=True)


def _p12_two_sums(reduce, hidden_states, h, cur, side):
    """Sum the first half and fork the side stream after it. By default the second
    half's sum is issued right behind the first, before the fork, so in the captured
    graph it continues on the first sum's stream instead of becoming a branch the
    graph starts late (~15 us); the side stream waits on an event for the first
    half. SGLANG_TTFT_P12_ORDER=fork_first restores fork-then-second-sum (b=None)."""
    import torch

    a = reduce(hidden_states[:h])
    if _P12_ORDER == "fork_first":
        side.wait_stream(cur)
        return a, None
    ev = torch.cuda.Event()
    ev.record(cur)
    b = reduce(hidden_states[h:])
    side.wait_event(ev)
    return a, b


def _p12_side_stream():
    import torch

    dev = torch.cuda.current_device()
    s = _P12_STREAMS.get(dev)
    if s is None:
        s = _P12_STREAMS[dev] = torch.cuda.Stream(device=dev)
    return s


def _p12_split_at(x):
    import torch

    if not (isinstance(x, torch.Tensor) and x.is_cuda and x.dim() == 2 and x.is_contiguous()
            and x.dtype == torch.bfloat16 and x.shape[0] >= _P12_MIN_ROWS):
        return None
    m = x.shape[0]
    h = (m // 2 + 15) // 16 * 16
    return h if 32 < h < m - 32 else None


def _p12_row_chunks(lo, hi):
    n = max(1, _P12_CHUNKS)
    step = max(16, (hi - lo + n - 1) // n)
    return [slice(a, min(hi, a + step)) for a in range(lo, hi, step)]


def _p12_combine_ok(hc, block_output, hyper_input, normed, partials):
    """The rows take GatedResidual.combine's JIT hc_combine path (and only it)."""
    import torch

    w = hc.block_inject_weight.weight
    return (partials is None and normed is not None and hc._jit_combine_ok
            and block_output.dtype == hyper_input.dtype == normed.dtype == w.dtype == torch.bfloat16
            and hyper_input.shape == normed.shape and hyper_input.is_contiguous() and normed.is_contiguous()
            and hyper_input.shape[0] == block_output.shape[0]
            and hyper_input.shape[-1] == hc.hc_count * hc.hidden_size)


def _p12_combine_into(hc, block_output, hyper_input, normed, out):
    from sglang.kernels.ops.elementwise.hc_combine import hc_combine

    hc_combine(block_output, hyper_input, normed, hc.block_inject_weight.weight, hc.hc_count, hc.hidden_size, out=out)


def _p12_norm_ok(norm_mod, x):
    import torch

    return norm_mod._jit_group_size is not None and x.dtype in (torch.bfloat16, torch.float16)


def _p12_norm_into(norm_mod, x, out):
    from sglang.kernels.ops.layernorm.grouped_gemma_rmsnorm import grouped_gemma_rmsnorm

    grouped_gemma_rmsnorm(x, norm_mod.weight, norm_mod._jit_group_size, norm_mod.variance_epsilon, out=out)


def _ttft_mix():
    from sglang.srt.ttft import ttft_mix

    return ttft_mix


def _p12b_mix_ok(hc, residual):
    """The FFN mix takes GatedResidual._mix_compute (torch.compile) and both of its
    GEMM algorithms for this row count are known (matched before graph capture)."""
    import sglang.srt.layers.hyperconnection as hcmod

    tm = _ttft_mix()
    if hc._jit_mix_ok or not tm.supported(hc, residual):
        return False
    if hcmod.fused_hc_mix_supported(residual, hc.input_mix_weight_down.weight, hc.input_mix_weight_up.weight):
        return False
    m, lowrank = residual.shape[0], hc.input_mix_weight_down.weight.shape[0]
    return tm.algo_for(m, residual.shape[1], lowrank) is not None and tm.algo_for(m, lowrank, residual.shape[1]) is not None


def _patch_prefill_graph_runner(mod):
    """P12b: match the mix GEMM algorithms of each prefill bucket before it is captured."""
    orig = mod.PrefillCudaGraphRunner.capture_one_shape

    def capture_one_shape(self, size, *args, **kw):
        if size >= _P12_MIN_ROWS:
            _ttft_mix().prebuild(size)
        return orig(self, size, *args, **kw)

    mod.PrefillCudaGraphRunner.capture_one_shape = capture_one_shape
    print(f"[ttft-patch] P12b prebuilds pinned mix GEMMs per prefill bucket (pid {os.getpid()})", file=sys.stderr, flush=True)


def _patch_boundary_prepare(mod):
    """P12 H1: attention output sum -> attn combine -> FFN norm, split in two."""
    import torch

    from sglang.srt.distributed import tensor_model_parallel_all_reduce
    from sglang.srt.layers.layer_boundary.layout import SumGroup
    from sglang.srt.layers.layer_boundary.residual.gated import _AttnUpdate, _FfnReadout

    orig = mod._reduce_update_read

    def _reduce_update_read(hidden_states, residual, forward_batch, norm, **kw):
        read, update, group = kw.get("read"), kw.get("update"), kw.get("group", SumGroup.ATTN_TP)
        h = _p12_split_at(hidden_states)
        if (h is None or not isinstance(read, _FfnReadout) or not isinstance(update, _AttnUpdate)
                or update.state is not read.state or norm is not None or kw.get("gathers_residual")
                or kw.get("fusions") or kw.get("read_fusions") or kw.get("cache") is not None
                or kw.get("quant_format") or kw.get("post_residual_addition") is not None
                or group not in (SumGroup.ATTN_TP, SumGroup.TP) or update.quantized_sum):
            return orig(hidden_states, residual, forward_batch, norm, **kw)
        state = read.state
        layer = getattr(state.attn_combine, "__self__", None)
        if layer is None or getattr(layer, "_hc_norms", None) is not None:
            return orig(hidden_states, residual, forward_batch, norm, **kw)
        hc_a, hc_m = layer.attn_hyper_connection, layer.mlp_hyper_connection
        normed = state.normed
        if (not _p12_combine_ok(hc_a, hidden_states, residual, normed, state.gate_partials)
                or not hc_m.config.hc_per_branch_norm or not _p12_norm_ok(hc_m.hc_norm, residual)):
            return orig(hidden_states, residual, forward_batch, norm, **kw)
        if group is SumGroup.ATTN_TP:
            reduce = lambda x: mod.attn_tp_all_reduce(x, forward_batch, may_quantize=False)
        else:
            reduce = tensor_model_parallel_all_reduce
        m = hidden_states.shape[0]
        split_mix = "p12b" in _PATCHES and _p12b_mix_ok(hc_m, residual)
        _p12_note("H1b" if split_mix else "H1", m, h)
        cur, side = torch.cuda.current_stream(), _p12_side_stream()
        new_res = torch.empty_like(residual)
        new_normed = torch.empty_like(residual)
        if split_mix:
            lowrank = hc_m.input_mix_weight_down.weight.shape[0]
            low = residual.new_empty((m, lowrank))
            up = torch.empty_like(residual)
            mixed = residual.new_empty((m, hc_m.hidden_size))
        a, b = _p12_two_sums(reduce, hidden_states, h, cur, side)
        with torch.cuda.stream(side):
            for c in _p12_row_chunks(0, h):
                _p12_combine_into(hc_a, a[c], residual[c], normed[c], new_res[c])
                _p12_norm_into(hc_m.hc_norm, new_res[c], new_normed[c])
            if split_mix:
                _ttft_mix().mix_rows(hc_m, new_normed[:h], mixed[:h], low[:h], up[:h], m)
        if b is None:
            b = reduce(hidden_states[h:])
        _p12_combine_into(hc_a, b, residual[h:], normed[h:], new_res[h:])
        _p12_norm_into(hc_m.hc_norm, new_res[h:], new_normed[h:])
        if split_mix:
            _ttft_mix().mix_rows(hc_m, new_normed[h:], mixed[h:], low[h:], up[h:], m)
        cur.wait_stream(side)
        if split_mix:
            return state._read(lambda r: (mixed, (r, new_normed, None)), new_res, None)
        return state._read(lambda r: hc_m.mix(r, normalized_input=new_normed), new_res, None)

    mod._reduce_update_read = _reduce_update_read
    print(f"[ttft-patch] P12 split attention-output sum/combine/norm (pid {os.getpid()})", file=sys.stderr, flush=True)


def _patch_boundary_exit(mod):
    """P12 H2: MoE output sum -> FFN combine, split in two."""
    import torch

    from sglang.srt.layers.layer_boundary.ops import keep_output
    from sglang.srt.layers.layer_boundary.residual.gated import _FfnUpdate

    orig = mod.ExitPolicy._complete_now

    def _complete_now(self, hidden_states, residual, *, forward_batch, dp_step, steps, output_move, owes_sum):
        update = steps.output.update
        h = _p12_split_at(hidden_states)
        layer = getattr(getattr(getattr(update, "state", None), "ffn_combine", None), "__self__", None)
        if (h is None or not owes_sum or residual is None or output_move is not keep_output
                or steps.output.transform is not None or steps.returns_over_dp
                or not isinstance(update, _FfnUpdate) or update.quantized_sum or layer is None
                or getattr(layer, "_hc_norms", None) is not None):
            return orig(self, hidden_states, residual, forward_batch=forward_batch, dp_step=dp_step,
                        steps=steps, output_move=output_move, owes_sum=owes_sum)
        state, hc_m = update.state, layer.mlp_hyper_connection
        normed = state.normed
        if not _p12_combine_ok(hc_m, hidden_states, residual, normed, state.gate_partials):
            return orig(self, hidden_states, residual, forward_batch=forward_batch, dp_step=dp_step,
                        steps=steps, output_move=output_move, owes_sum=owes_sum)
        group = steps.output.group
        reduce = lambda x: mod.sum_output(x, group, forward_batch, may_quantize=False)
        m = hidden_states.shape[0]
        # P12c: also read the next layer's attention input (its norm, and with p12b its
        # mix) from the new streams, half by half, while the second half is summed.
        nxt = _P12_LAYERS.get(getattr(layer, "layer_id", -2) + 1) if "p12c" in _PATCHES else None
        hc_n = nxt.attn_hyper_connection if nxt is not None else None
        pre = (hc_n is not None and nxt.ple is None and getattr(nxt, "_hc_norms", None) is None
               and hc_n.config.hc_per_branch_norm and _p12_norm_ok(hc_n.hc_norm, residual))
        pre_mix = pre and "p12b" in _PATCHES and _p12b_mix_ok(hc_n, residual)
        _p12_note("H2c" if pre else "H2", m, h)
        cur, side = torch.cuda.current_stream(), _p12_side_stream()
        new_res = torch.empty_like(residual)
        if pre:
            new_normed = torch.empty_like(residual)
        if pre_mix:
            low = residual.new_empty((m, hc_n.input_mix_weight_down.weight.shape[0]))
            up = torch.empty_like(residual)
            mixed = residual.new_empty((m, hc_n.hidden_size))
        a, b = _p12_two_sums(reduce, hidden_states, h, cur, side)
        with torch.cuda.stream(side):
            for c in _p12_row_chunks(0, h):
                _p12_combine_into(hc_m, a[c], residual[c], normed[c], new_res[c])
                if pre:
                    _p12_norm_into(hc_n.hc_norm, new_res[c], new_normed[c])
            if pre_mix:
                _ttft_mix().mix_rows(hc_n, new_normed[:h], mixed[:h], low[:h], up[:h], m)
        if b is None:
            b = reduce(hidden_states[h:])
        _p12_combine_into(hc_m, b, residual[h:], normed[h:], new_res[h:])
        if pre:
            _p12_norm_into(hc_n.hc_norm, new_res[h:], new_normed[h:])
        if pre_mix:
            _ttft_mix().mix_rows(hc_n, new_normed[h:], mixed[h:], low[h:], up[h:], m)
        cur.wait_stream(side)
        state.clear_coefficients()
        _P12_PRE[:] = [new_res, nxt, new_normed, mixed if pre_mix else None] if pre else []
        return new_res, None

    mod.ExitPolicy._complete_now = _complete_now
    print(f"[ttft-patch] P12 split MoE-output sum/combine (pid {os.getpid()})", file=sys.stderr, flush=True)


def _patch_qwen4_attn_read(mod):
    """P12c: the attention stage's read takes the norm (and mix) P12's MoE-exit split
    already computed for this layer from these exact streams; it also registers the
    layers so the exit can find the next one."""
    cls = mod.Qwen4ExpLayerExtensionMixin
    orig = cls._attn_mix

    def _attn_mix(self, residual):
        _P12_LAYERS.setdefault(getattr(self, "layer_id", None), self)
        pre = _P12_PRE[:]
        _P12_PRE.clear()
        if pre and pre[0] is residual and pre[1] is self and self.ple is None:
            _, _, normed, mixed = pre
            if mixed is None:
                return self.attn_hyper_connection.mix(residual, normalized_input=normed)
            return mixed, (residual, normed, None)
        return orig(self, residual)

    cls._attn_mix = _attn_mix
    print(f"[ttft-patch] P12c attention read reuses the MoE-exit split read (pid {os.getpid()})", file=sys.stderr, flush=True)


def _patch_model_config_text_only(mod):
    """textonly: serve the checkpoint as a text model (enable_multimodal=False); the
    CLI flag can only turn multimodal on. Needed by the embedded Rust frontend, which
    refuses multimodal model types it has no pipeline for. mrope stays as configured."""
    orig = mod.ModelConfig.__init__

    def __init__(self, *a, **kw):
        kw["enable_multimodal"] = False
        return orig(self, *a, **kw)

    mod.ModelConfig.__init__ = __init__
    print(f"[ttft-patch] textonly: enable_multimodal forced off (pid {os.getpid()})", file=sys.stderr, flush=True)


# ---- rustside: /flush_cache and cached_tokens for the embedded Rust frontend ----
# The Rust frontend (SGLANG_RUST_SERVER=1) serves neither /flush_cache nor cached_tokens
# in the /generate stream, and its routes/egress schema are compiled in. A small HTTP
# side channel in the rank-0 scheduler (127.0.0.1:$SGLANG_TTFT_SIDE_PORT, default
# listen port + 1000) fills the gap for measurement and ops:
#   /flush_cache          injects FlushCacheReqInput into the scheduler's own request
#                         pull (broadcast to the other TP ranks like any request); the
#                         reply is routed back here instead of into the Rust egress ring
#   /last_cached_tokens   cached_tokens / input length of the most recent prefill
#   /get_server_info, /get_model_info, /health   proxied to the Rust listener
_SIDE = {"queue": None, "waiters": {}, "last": None, "n": 0, "rust_port": None}


def _side_start(rust_port):
    import http.server
    import json
    import queue
    import threading
    import urllib.request

    from sglang.srt.managers.io_struct import FlushCacheReqInput

    _SIDE["queue"] = queue.Queue()
    _SIDE["rust_port"] = rust_port
    port = int(os.environ.get("SGLANG_TTFT_SIDE_PORT", rust_port + 1000))

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="application/json"):
            data = body if isinstance(body, bytes) else body.encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _route(self):
            path = self.path.split("?")[0]
            if path == "/flush_cache":
                _SIDE["n"] += 1
                rid = f"ttft-side-flush-{_SIDE['n']}"
                waiter = {"event": threading.Event(), "output": None}
                _SIDE["waiters"][rid] = waiter
                _SIDE["queue"].put(FlushCacheReqInput(rid=rid))
                if not waiter["event"].wait(120):
                    _SIDE["waiters"].pop(rid, None)
                    return self._send(504, "flush_cache timed out\n", "text/plain")
                out = waiter["output"]
                if getattr(out, "success", False):
                    return self._send(200, "Cache flushed.\nPlease check backend logs for more details. (When there are running or waiting requests, the operation will not be performed.)\n", "text/plain")
                return self._send(400, f"Cache not flushed. {getattr(out, 'message', '')}\n", "text/plain")
            if path == "/last_cached_tokens":
                return self._send(200, json.dumps(_SIDE["last"]))
            target = {"/get_server_info": "/server_info", "/get_model_info": "/get_model_info", "/health": "/health"}.get(path)
            if target is not None:
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{_SIDE['rust_port']}{target}", timeout=30) as r:
                        return self._send(r.status, r.read(), r.headers.get("Content-Type", "application/json"))
                except Exception as e:
                    return self._send(502, json.dumps({"error": repr(e)}))
            return self._send(404, json.dumps({"error": f"no route {path}"}))

        do_GET = _route
        do_POST = _route

    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, name="ttft-side-http", daemon=True).start()
    print(f"[ttft-patch] rustside: /flush_cache and /last_cached_tokens on 127.0.0.1:{port} (pid {os.getpid()})", file=sys.stderr, flush=True)


def _patch_rust_side_server(mod):
    cls = mod.RustServer
    orig_launch = cls.launch.__func__
    orig_push = cls.push_control_output

    def launch(klass, scheduler):
        inst = orig_launch(klass, scheduler)
        _side_start(inst.http_port)
        return inst

    def push_control_output(self, recv_req, output):
        waiter = _SIDE["waiters"].pop(getattr(recv_req, "rid", None), None)
        if waiter is not None:
            waiter["output"] = output
            waiter["event"].set()
            return
        return orig_push(self, recv_req, output)

    cls.launch = classmethod(launch)
    cls.push_control_output = push_control_output


def _patch_rust_side_receiver(mod):
    import queue

    orig = mod.SchedulerRequestReceiver._pull_raw_reqs

    def _pull_raw_reqs(self):
        reqs = orig(self)
        q = _SIDE["queue"]
        if q is not None and reqs is not None:
            while True:
                try:
                    reqs.append(q.get_nowait())
                except queue.Empty:
                    break
        return reqs

    mod.SchedulerRequestReceiver._pull_raw_reqs = _pull_raw_reqs


def _patch_rust_side_record(mod):
    orig = mod.SchedulerBatchResultProcessor.process_batch_result_prefill

    def process_batch_result_prefill(self, batch, *a, **kw):
        out = orig(self, batch, *a, **kw)
        for req in batch.reqs:
            _SIDE["last"] = {"rid": req.rid, "cached_tokens": int(getattr(req, "cached_tokens", 0) or 0),
                             "input_tokens": len(req.origin_input_ids)}
        return out

    mod.SchedulerBatchResultProcessor.process_batch_result_prefill = process_batch_result_prefill


# --- reqrec: scheduler-side record of what each frontend hands the scheduler (diagnostics) ---
_REC = {"f": None, "done": set()}


def _rec_write(obj):
    import json as _json
    if _REC["f"] is None:
        _REC["f"] = open(os.environ["SGLANG_TTFT_REQREC"], "a")
    _REC["f"].write(_json.dumps(obj, ensure_ascii=False, default=lambda o: sorted(o) if isinstance(o, (set, frozenset)) else str(o)) + "\n")
    _REC["f"].flush()


def _patch_reqrec_scheduler(mod):
    import hashlib
    import msgspec

    from sglang.srt.runtime_context import get_parallel

    orig = mod.Scheduler.handle_generate_request

    def handle_generate_request(self, recv_req, *a, **kw):
        if get_parallel().tp_rank == 0:
            ids = list(recv_req.input_ids or [])
            sp = recv_req.sampling_params
            sp = msgspec.structs.asdict(sp) if isinstance(sp, msgspec.Struct) else dict(vars(sp))
            _rec_write({"t": "in", "rid": recv_req.rid, "n": len(ids),
                        "sha": hashlib.sha256(repr(ids).encode()).hexdigest()[:16], "ids": ids if len(ids) <= 20000 else None,
                        "sp": sp, "input_text": recv_req.input_text, "return_logprob": recv_req.return_logprob,
                        "logprob_start_len": recv_req.logprob_start_len, "top_logprobs_num": recv_req.top_logprobs_num,
                        "stream": recv_req.stream, "return_hidden_states": recv_req.return_hidden_states})
        return orig(self, recv_req, *a, **kw)

    mod.Scheduler.handle_generate_request = handle_generate_request


def _patch_reqrec_streamer(mod):
    from sglang.srt.runtime_context import get_parallel

    orig = mod.SchedulerOutputStreamer._stream_output_generation

    def _stream_output_generation(self, reqs, *a, **kw):
        out = orig(self, reqs, *a, **kw)
        if get_parallel().tp_rank == 0:
            for req in reqs:
                if req.finished() and req.rid not in _REC["done"]:
                    _REC["done"].add(req.rid)
                    _rec_write({"t": "out", "rid": req.rid, "output_ids": list(req.output_ids),
                                "finish": req.finished_reason.to_json() if req.finished_reason else None,
                                "cached_tokens": int(req.cached_tokens or 0), "reasoning_tokens": int(req.reasoning_tokens or 0)})
        return out

    mod.SchedulerOutputStreamer._stream_output_generation = _stream_output_generation


# --- rustso: serve the embedded Rust frontend from a rebuilt `_server` extension (path in
# SGLANG_TTFT_RUST_SO) instead of the wheel's; the venv is untouched ---
class _RustSoFinder(importlib.abc.MetaPathFinder):
    NAME = "sglang.srt.rust_extensions._server"

    def find_spec(self, name, path, target=None):
        if name != self.NAME:
            return None
        return importlib.util.spec_from_file_location(name, os.environ["SGLANG_TTFT_RUST_SO"])


# --- rustmeta: per-request meta columns for the rebuilt Rust frontend. The wheel's push_generation
# sends rids/finish/prompt_tokens/tok_lens (+12 extras columns when logprobs are on); the rebuilt
# BatchHeader takes three more trailing columns (cached_tokens, reasoning_tokens, retractions) ---
def _patch_rust_meta(mod):
    old = """        header = msgspec.msgpack.encode(header_cols)
"""
    new = """        if len(header_cols) == 4:
            header_cols += [[]] * 12
        header_cols += [payload.cached_tokens, payload.reasoning_tokens, payload.retraction_counts or []]
        header = msgspec.msgpack.encode(header_cols)
"""
    _rewrite_method(mod, mod.RustServer, "push_generation", [(old, new)], "rustmeta")


def _patch_rust_meta_accumulator(mod):
    """In Rust mode the stream accumulator skips the per-request counts Python's
    TokenizerManager reports; collect the two the rebuilt frontend reads."""
    old = """        if not self.rust_server_mode:
            # Everything below feeds"""
    new = """        if self.rust_server_mode:
            self.reasoning_tokens.append(req.reasoning_tokens)
            self.cached_tokens.append(req.cached_tokens)
        if not self.rust_server_mode:
            # Everything below feeds"""
    _rewrite_method(mod, mod._GenerationStreamAccumulator, "accept", [(old, new)], "rustmeta")


def _add_target(name, fn):
    prev = _TARGETS.get(name)
    if prev is None:
        _TARGETS[name] = fn
        return

    def both(mod):
        prev(mod)
        fn(mod)

    _TARGETS[name] = both


_TARGETS = {}
if "p10" in _PATCHES:
    _TARGETS["sglang.srt.layers.quantization.unquant"] = _patch_unquant
if "p9" in _PATCHES:
    _TARGETS["sglang.srt.layers.attention.qwen_sparse_attn_backend"] = _patch_qsa_backend
if "p8" in _PATCHES:
    _TARGETS["sglang.srt.layers.attention.qsa.mqa"] = _patch_mqa
if "p5" in _PATCHES:
    _TARGETS["sglang.srt.managers.scheduler_components.batch_result_processor"] = _patch_batch_result_processor
if "p6" in _PATCHES:
    _TARGETS["sglang.srt.utils.common"] = _patch_common
if "p4" in _PATCHES:
    _TARGETS["sglang.srt.managers.scheduler"] = _patch_scheduler
if "p1" in _PATCHES or "dump" in _PATCHES:
    def _patch_sparse_attn_all(mod):
        if "p1" in _PATCHES:
            _patch_sparse_attn(mod)
        if "dump" in _PATCHES:
            _patch_sparse_attn_dump(mod)

    _TARGETS["sglang.srt.layers.attention.qsa.sparse_attn"] = _patch_sparse_attn_all
if "p2" in _PATCHES:
    _TARGETS["sglang.api.v1.api_types"] = _patch_api_types
if "p3" in _PATCHES:
    _TARGETS["starlette.requests"] = _patch_starlette_requests


if "p11" in _PATCHES:
    _add_target("sglang.srt.layers.attention.qwen_sparse_attn_backend", _patch_qsa_backend_syncs)
    _add_target("sglang.srt.layers.attention.hybrid_linear_attn_backend", _patch_hybrid_backend_syncs)
    _add_target("sglang.kernels.ops.attention.fla.index", _patch_fla_index)
if "p14" in _PATCHES:
    if "p11" not in _PATCHES:
        raise RuntimeError("p14 rewrites the p11 form of the QSA row map; enable p11 too")
    _add_target("sglang.srt.layers.attention.qwen_sparse_attn_backend", _patch_qsa_host_plan)
if "rustside" in _PATCHES:
    _add_target("sglang.srt.rust_server.server", _patch_rust_side_server)
    _add_target("sglang.srt.managers.scheduler_components.request_receiver", _patch_rust_side_receiver)
    _add_target("sglang.srt.managers.scheduler_components.batch_result_processor", _patch_rust_side_record)
if "reqrec" in _PATCHES:
    _add_target("sglang.srt.managers.scheduler", _patch_reqrec_scheduler)
    _add_target("sglang.srt.managers.scheduler_components.output_streamer", _patch_reqrec_streamer)
if "rustmeta" in _PATCHES:
    _add_target("sglang.srt.rust_server.server", _patch_rust_meta)
    _add_target("sglang.srt.managers.scheduler_components.output_streamer", _patch_rust_meta_accumulator)
if "textonly" in _PATCHES:
    _add_target("sglang.srt.configs.model_config", _patch_model_config_text_only)
if "p15" in _PATCHES:
    _add_target("sglang.srt.managers.scheduler", _patch_scheduler_idle)
if "p13" in _PATCHES:
    _add_target("sglang.srt.layers.attention.qsa.metadata", _patch_qsa_metadata)
if "p12" in _PATCHES:
    _add_target("sglang.srt.layers.layer_boundary.prepare", _patch_boundary_prepare)
    if "p12b" in _PATCHES:
        _add_target("sglang.srt.model_executor.runner.prefill_cuda_graph_runner", _patch_prefill_graph_runner)
    if "p12c" in _PATCHES:
        _add_target("sglang.srt.models.qwen4_exp", _patch_qwen4_attn_read)
    _add_target("sglang.srt.layers.layer_boundary.exit", _patch_boundary_exit)


def _trace2_scheduler(mod):
    _wrap_trace(mod.Scheduler, "process_input_requests", before="pir_start", after="pir_end", cond=lambda a, kw, r: bool(a[1]))
    _wrap_trace(mod.Scheduler, "get_new_batch_prefill", before="gnbp_start", after="gnbp_end")


def _trace2_schedule_batch(mod):
    _wrap_trace(mod.ScheduleBatch, "prepare_for_extend", before="pfe_start", after="pfe_end")


def _trace2_tp_worker(mod):
    _wrap_trace(mod.TpModelWorker, "forward_batch_generation", before="fbg_start", after="fbg_end",
                cond=lambda a, kw, r: a[1].forward_mode.is_extend())


def _trace2_model_runner(mod):
    _wrap_trace(mod.ModelRunner, "forward", before="mrf_start", after="mrf_end",
                cond=lambda a, kw, r: a[1].forward_mode.is_extend())


def _trace2_prefill_runner(mod):
    _wrap_trace(mod.PrefillCudaGraphRunner, "execute", before="pcg_start", after="pcg_end")
    _wrap_trace(mod.PrefillCudaGraphRunner, "load_batch", after="pcg_loaded")


def _trace2_hybrid_backend(mod):
    _wrap_trace(mod.HybridLinearAttnBackend, "init_forward_metadata", before="ifm_start", after="ifm_end",
                cond=lambda a, kw, r: a[1].forward_mode.is_extend())


def _trace2_radix(mod):
    _wrap_trace(mod.UnifiedRadixCache, "match_prefix", before="mp_start", after="mp_end")


_CPROF = {}


def _cprof_wrap(cls, attr, tag, cond=None):
    """Diagnostics only (patch "cprof"): cProfile every call of cls.attr that passes cond,
    accumulate, and dump pstats text to $TTFT_TRACE_DIR/cprof-<tag>-<pid>.txt every 20 calls."""
    import cProfile
    import io
    import pstats

    fn = getattr(cls, attr)
    state = {"prof": cProfile.Profile(), "n": 0}

    def wrapped(*a, **kw):
        if cond is not None and not cond(a, kw):
            return fn(*a, **kw)
        state["prof"].enable()
        try:
            return fn(*a, **kw)
        finally:
            state["prof"].disable()
            state["n"] += 1
            if state["n"] % 20 == 0:
                buf = io.StringIO()
                st = pstats.Stats(state["prof"], stream=buf).sort_stats("cumulative")
                buf.write(f"calls {state['n']}\n")
                st.print_stats(45)
                with open(f"{os.environ['TTFT_TRACE_DIR']}/cprof-{tag}-{os.getpid()}.txt", "w") as f:
                    f.write(buf.getvalue())

    setattr(cls, attr, wrapped)


def _is_extend_arg1(a, kw):
    fb = a[1] if len(a) > 1 else kw.get("forward_batch")
    return fb is not None and fb.forward_mode.is_extend()


if "trace" in _PATCHES:
    def _chain(name, extra):
        prev = _TARGETS.get(name)

        def both(mod):
            if prev is not None:
                prev(mod)
            extra(mod)

        _TARGETS[name] = both

    _chain("sglang.srt.entrypoints.http_server", _trace_http_server)
    _chain("sglang.srt.managers.tokenizer_manager", _trace_tokenizer_manager)
    _chain("sglang.srt.managers.scheduler_components.request_receiver", _trace_request_receiver)
    _chain("sglang.srt.managers.scheduler", _trace_scheduler)
    _chain("sglang.srt.managers.scheduler_components.batch_result_processor", _trace_batch_result_processor)
    _chain("sglang.srt.managers.scheduler_components.output_streamer", _trace_output_streamer)
    _chain("sglang.srt.managers.detokenizer_manager", _trace_detokenizer)
    if "cprof" in _PATCHES:
        _chain("sglang.srt.layers.attention.hybrid_linear_attn_backend",
               lambda m: _cprof_wrap(m.HybridLinearAttnBackend, "init_forward_metadata", "ifm", _is_extend_arg1))
        _chain("sglang.srt.managers.schedule_batch",
               lambda m: _cprof_wrap(m.ScheduleBatch, "prepare_for_extend", "pfe"))
        _chain("sglang.srt.managers.tokenizer_manager",
               lambda m: _cprof_wrap(m.TokenizerManager, "_create_tokenized_object", "tmc"))
    if "trace2" in _PATCHES:
        _chain("sglang.srt.managers.scheduler", _trace2_scheduler)
        _chain("sglang.srt.managers.schedule_batch", _trace2_schedule_batch)
        _chain("sglang.srt.managers.tp_worker", _trace2_tp_worker)
        _chain("sglang.srt.model_executor.model_runner", _trace2_model_runner)
        _chain("sglang.srt.model_executor.runner.prefill_cuda_graph_runner", _trace2_prefill_runner)
        _chain("sglang.srt.layers.attention.hybrid_linear_attn_backend", _trace2_hybrid_backend)
        _chain("sglang.srt.mem_cache.unified_radix_cache", _trace2_radix)


class _PatchFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        fn = _TARGETS.get(name)
        if fn is None:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return spec
        orig_exec = spec.loader.exec_module

        def exec_module(module):
            orig_exec(module)
            fn(module)

        spec.loader.exec_module = exec_module
        return spec


if _TARGETS:
    sys.meta_path.insert(0, _PatchFinder())
if "rustso" in _PATCHES:
    sys.meta_path.insert(0, _RustSoFinder())
