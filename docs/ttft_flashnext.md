# TTFT patches for Qwen3.8-Flash-Next (SGLang v0.5.22 branch)

This branch is `sgl-project/sglang` v0.5.22 plus the lossless TTFT work for
Qwen3.8-Flash-Next-NVFP4. It is the source-build counterpart of the prebuilt
`handoff-ttft-tob-20261020` package.

## What is included

- `rust/`: the rebuilt Rust frontend with Python/MOSS parity for
  `/v1/chat/completions`, tool calls, sampling defaults, reasoning split,
  response shapes, and `/flush_cache`, `/get_server_info`, `cached_tokens` on
  the main port.
- `python/sglang/srt/ttft/`: P1-P15, `textonly`, `rustmeta` and the supporting
  Triton kernels. The package is imported from `sglang/__init__.py` only when
  `SGLANG_TTFT_PATCH` is set; without it the branch behaves like v0.5.22.
- P7 is intentionally not included (it was not bit-identical on cached-prefix
  extends).

## Build and install

```bash
git clone -b ttft-flashnext-v0.5.22 https://github.com/Sy0307/sglang.git
cd sglang
python3 -m venv .venv
. .venv/bin/activate
pip install -U pip
pip install -e "python[all]"          # builds the Rust extension from rust/
```

Rust 1.92 is pinned by `rust/rust-toolchain.toml`; `pip install -e` builds the
extension as part of the normal SGLang build.

## Launch (Rust frontend)

```bash
export SGLANG_TTFT_PATCH=p1,p2,p4,p5,p6,p8,p9,p11,p12,p12b,p12c,p13,p14,p15,textonly,rustmeta
export SGLANG_RUST_SERVER=1
python -m sglang.launch_server \
  --model-path /path/to/RadixArk--Qwen3.8-Flash-Next-NVFP4 \
  --served-model-name qwen38-nvfp4 --host 127.0.0.1 --port 30002 \
  --tp-size 2 --ep-size 2 --moe-a2a-backend none --dtype bfloat16 \
  --kv-cache-dtype bf16 --qsa-indexer-dtype bfloat16 --mamba-ssm-dtype float32 \
  --no-ple-offload-embedding --mem-fraction-static 0.85 --context-length 65536 \
  --max-total-tokens 131072 --max-running-requests 16 --max-mamba-cache-size 96 \
  --chunked-prefill-size 4096 --cuda-graph-max-bs-decode 16 --stream-interval 1 \
  --reasoning-parser auto --tool-call-parser qwen3_coder \
  --moe-runner-backend flashinfer_cutlass --disable-custom-all-reduce \
  --boundary-reduction rs
```

`textonly` is required with the Rust frontend (the model has no Rust multimodal
pipeline; image requests return 400). `rustmeta` lets the scheduler report
`cached_tokens`, `reasoning_tokens` and `num_retractions` to the Rust frontend.
`rustso` is not needed here: the extension is built into the package.

For the Python frontend, drop `SGLANG_RUST_SERVER` and `textonly`:

```bash
export SGLANG_TTFT_PATCH=p1,p2,p4,p5,p6,p8,p9,p11,p12,p12b,p12c,p13,p14,p15
python -m sglang.launch_server ...   # same arguments
```

## Expected numbers

Same-machine comparison on 8x RTX PRO 6000 (tob-04), 32K cached prefix + 1K new
tokens:

| config | unique suffix | historical |
|---|---|---|
| v0.5.22 original | 106.5 ms | 105.6 ms |
| this branch, Python frontend | 87.1 ms | 86.4 ms |
| this branch, Rust frontend | 85.4 ms | 84.6 ms |

The environment settings (disable ACPI C2, NUMA1 governor=performance) are not
part of the branch; see the handoff README. Without them Sinnet TTFT is much
worse.

## Caveats

- Rust frontend serves text only; images return 400.
- A few request shapes still take the original Dynamo path (tool schema with
  `pattern`/`if-then`, system messages carrying tools, `defer_loading`,
  `response_format` plus tool constraints, `return_*` fields). Tool parsing is
  aligned, but prompt/sampling details are not.
- `/v1/completions` is unchanged.
- The MOSS Python-side listener patch is not part of SGLang; when deploying
  behind the MOSS stack, keep the existing listener integration.
