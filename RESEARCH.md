# Speed-frontier research: usable LLMs per size on RTX PRO 6000 (96 GB)

Goal (revised): map **how fast a *working* LLM of a given size runs** on this molab
instance, and turn that into a notebook with a model table + run instructions +
(eventually) an automated loader widget. The "largest model" findings
(`notebook.py`) are kept for reference.

**Hardware:** NVIDIA RTX PRO 6000 Blackwell, **96 GB VRAM** (sm_120), 160 GB RAM,
20 cores, CUDA 13.0 driver. Base env: torch 2.12+cu130, transformers 5.10,
triton 3.7, flash-attn-4 (sm120 patch). **No `nvcc`/CUDA toolchain** → no JIT CUDA
kernel builds (rules out marlin, source builds of llama.cpp).

## Scout: `llmfit` (our model radar)

`llmfit recommend --json` detects the box correctly (95.59 GB VRAM, CUDA/Blackwell)
and **recommends vLLM** as the runtime for the fast models. Its 2026 catalog skews
to frontier MoE models; standout heroes per bucket (estimated tok/s):

| Bucket | Hero model (llmfit) | Quant | VRAM | est tok/s | runtime |
|---|---|---|---|---|---|
| ~30 B | `Qwen3-Coder-30B-A3B-Instruct-AWQ` | AWQ-4bit | 15.6 GB | ~576 | vLLM |
| ~35 B | `Qwen3.x-35B-A3B …-GPTQ/AWQ-int4` | 4-bit | 18.4 GB | ~101 | vLLM |
| ~40 B | `Intel/DeepSeek-V4-Flash-W4A16` | W4A16 | 20.4 GB | ~101 | vLLM |
| ~80 B | `Qwen3-Next-80B-A3B …AWQ-4bit` | AWQ-4bit | 40.5 GB | ~53 | vLLM |

llmfit's catalog omits 4–15 B (too small to be "frontier"); pick those directly.

## Engine strategy (the real "way faster" unlock)

The previous agent ran **unquantized** Qwen/Gemma topping at 32 B via plain
`transformers` — memory-heavy and slow. The speed unlock is a real serving engine:

- **vLLM** (primary, "hacky-but-works"): installs cleanly in an isolated `uv` venv —
  **vLLM 0.23.0 + torch 2.11.0+cu130, CUDA 13** — imports fine on Blackwell. Run as a
  subprocess (offline `LLM` API for benchmarking, or `vllm serve` OpenAI endpoint for
  the loader widget). PagedAttention + prebuilt CUDA kernels (no nvcc) = high tok/s.
  ⚠️ Bring-up was **not yet verified to generate** — the sandbox terminated during the
  first smoke test (`Qwen3-4B`), so vLLM end-to-end still needs confirmation.
- **transformers + flash-attn-4** (conventional fallback, "normal cell"): for bf16
  small models and MXFP4 (`gpt-oss`, via `kernels>=0.12,<0.13`).
- **gptqmodel TORCH backend**: works but slow; only needed for GPU+CPU offload of the
  235 B reference model.

Dead ends confirmed earlier: compressed-tensors W4A16 (decompresses to bf16, garbage),
AWQ-marlin (needs nvcc/`CUDA_HOME`).

## Benchmark plan (per size) — to run once the sandbox is back

Measure real decode tok/s (greedy, 256 tokens, warm) via vLLM offline API:

| Size | Candidate | Notes |
|---|---|---|
| 4 B | `Qwen/Qwen3-4B-Instruct-2507` | bf16, baseline small |
| 8 B | `Qwen/Qwen3-8B` / `meta-llama/Llama-3.1-8B-Instruct` | bf16 |
| 14 B | `Qwen/Qwen3-14B` | bf16 |
| 30 B (MoE) | `Qwen/Qwen3-30B-A3B-Instruct-2507` (+ AWQ) | the "fast 32B" hero (3B active) |
| 32 B (dense) | `Qwen/Qwen3-32B` (AWQ/GPTQ-4bit) | size the old agent maxed at — now fast |
| 80 B (MoE) | `Qwen/Qwen3-Next-80B-A3B-Instruct` (AWQ-4bit) | the sweet spot, newly accessible |

Reference (kept, from `notebook.py`): gpt-oss-120B (MXFP4, ~14 tok/s),
Qwen3-235B-A22B (GPTQ-Int4 + CPU offload, ~0.11 tok/s).

## ⛔ Critical finding: vLLM terminates the molab sandbox (2/2)

On **two separate sandboxes**, the instance was reclaimed (`HTTP 404 "sandbox
terminated"`) at the exact moment `vllm` starts loading a model — i.e. when the
vLLM **V1 EngineCore** spawns as a child process and runs its GPU memory-profiling
pass. Install + `import vllm` are fine; engine startup kills the sandbox.

Update — now **3/3**. A third sandbox died the same way even with the mitigations:
`VLLM_ENABLE_V1_MULTIPROCESSING=0` (in-process engine, no child proc),
`enforce_eager=True`, `gpu_memory_utilization=0.30`, and a 504 GB `/dev/shm`.
So it is **not** the child process or shm — the molab GPU watchdog reclaims the
sandbox during vLLM's CUDA init / memory-profiling pass. **vLLM is conclusively
unusable on molab; do not retry.**

**Decision:** treat vLLM as high-risk on molab. Primary engine pivots to
**in-process `transformers`** (proven stable here for hours — gpt-oss-120B and
Qwen3-235B both ran without killing the sandbox).

## Stable engine plan (no vLLM)

"Way faster than the old unquantized 32B" comes from:
- **bf16 + flash-attn-4** (sm120, already installed) for dense ≤32 B — fits in 96 GB
  (32 B bf16 ≈ 64 GB) and is far faster than fp32/no-flash.
- **MXFP4 + `kernels`** for gpt-oss MoE (20B/120B) — native Blackwell, fast, stable
  (120B measured ~14 tok/s, fully on GPU).
- **gptqmodel TORCH backend** only as a last resort (slow) for 4-bit that must fit.
- 70–80 B tier without vLLM: gpt-oss-120B (MXFP4, ~14 tok/s) is the stable fast
  option; dense 70 B needs 4-bit (slow here) or CPU offload.

## Status

- ✅ llmfit scout working; buckets + heroes identified.
- ✅ vLLM 0.23.0 installs & imports in isolated venv — but ⛔ **engine startup
  terminates the molab sandbox (2/2)**. Deprioritized.
- ⏳ Stable `transformers` per-size benchmarks: pending sandbox restart.
- ⏳ Notebook table + automated model-loader widget: pending benchmarks.

## Reproducible vLLM setup (isolated venv)

```bash
export UV_CACHE_DIR=/home/marimo/.uvcache
uv venv /home/marimo/vllm-venv --python 3.12
uv pip install --python /home/marimo/vllm-venv/bin/python vllm   # -> vllm 0.23.0, torch 2.11+cu130
export HF_HOME=/home/marimo/hfcache
# offline benchmark: vllm-venv/bin/python -c "from vllm import LLM, SamplingParams; ..."
```
