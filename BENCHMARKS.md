# Measured decode speed on RTX PRO 6000 Blackwell (96 GB)

Stable in-process `transformers` engine (no vLLM). Greedy decode, 256 new tokens,
warm run, short prompt. `dtype=bfloat16, attn_implementation="sdpa"` for dense
models; `dtype="auto"` (MXFP4) for gpt-oss.

| Model | Total params | Active | Precision | decode tok/s | VRAM | load s |
|---|---|---|---|---|---|---|
| Qwen3-4B-Instruct-2507 | 4 B | dense | **bf16** | **51.7** | 8.0 GB | 18 |
| Qwen3-8B | 8 B | dense | **bf16** | **51.6** | 16.4 GB | 37 |
| Qwen3-14B | 14 B | dense | **bf16** | **36.4** | 29.5 GB | 67 |
| Qwen3-32B | 32 B | dense | **bf16** | **17.7** | 65.5 GB | 147 |
| Qwen3-30B-A3B-Instruct-2507 | 30 B | 3 B (MoE) | **bf16** | **30.6** | 61.1 GB | 240 |
| openai/gpt-oss-20b | 20 B | 3.6 B (MoE) | **MXFP4** | **22.5** | 13.8 GB | 36 |
| openai/gpt-oss-120b (prior sandbox) | 117 B | 5.1 B (MoE) | MXFP4 | ~14* | 65 GB | ~33 |

\* measured on an earlier sandbox; reference value.

## Reading the numbers

- **4B–32B are unquantized bf16** — the honest baseline, same precision the prior
  agent used. 32B at 17.7 tok/s is the dense ceiling that fits comfortably (65 GB).
- **MoE is the architectural win:** Qwen3-30B-A3B (3 B active) hits **30.6 tok/s** —
  **1.7× faster than dense 32B** at the same size class, all in bf16.
- **gpt-oss MXFP4** runs natively on Blackwell (triton `kernels`), letting a 20 B MoE
  sit in **13.8 GB**; the 120 B sibling runs in ~65 GB at ~14 tok/s.

## What did NOT help (this box)

- **torch.compile (naive)**: `torch.compile(model.forward, fullgraph=True)` +
  static cache *regressed* 14B to **2.4 tok/s** (per-step recompilation / cudagraph
  marking issues). Not worth the fragility here; the real multiplier is a serving
  engine (vLLM), not compile.
- **4-bit for speed**: marlin needs nvcc (absent); gptqmodel torch backend is slow.
  On this box 4-bit's value is *fitting bigger models*, not raw speed — unless vLLM.

## Frontier summary (stable engine)

- **Fast & interactive (50+ tok/s):** ≤8 B dense bf16.
- **Sweet spot (30+ tok/s, capable):** Qwen3-30B-A3B (MoE bf16) — "32B-class, fast".
- **Big & usable (15–22 tok/s):** dense 32B bf16, gpt-oss-20B/120B MXFP4.
- **80B sweet spot (Qwen3-Next-80B-A3B):** needs 4-bit (40 GB) → only fast via vLLM;
  pending the guarded vLLM retry.

---

# Update — the static-KV-cache speed trick (fastest *stable* decode)

The biggest free win on this box is switching `transformers` to a **static KV cache**
(`model.generation_config.cache_implementation = "static"`). Measured decode tok/s
(greedy, 256 tok, warm), baseline (dynamic cache) vs static cache:

| Model | Type | baseline | **static cache** | speedup |
|---|---|---:|---:|---:|
| Qwen3-4B | dense bf16 | 54 | **112** | **2.07×** |
| Qwen3-8B | dense bf16 | 54 | **74** | 1.38× |
| Qwen3-14B | dense bf16 | 37 | **44** | 1.18× |
| Qwen3-32B | dense bf16 | 18 | **20** | 1.11× |
| **Qwen3-30B-A3B** | MoE bf16 | 32 | **85** | **2.68×** |
| gpt-oss-20B | MoE MXFP4 | 25 | 24 | 0.96× |

**Findings**
- Static cache helps most where per-step kernel-launch overhead dominates: **small
  dense and MoE** models. Big dense (32B) is bandwidth-bound, so it gains little.
- **Fastest capable model = Qwen3-30B-A3B at ~85 tok/s** (30B-class MoE, beats dense 8B).
- **MXFP4 (gpt-oss) does not benefit** — it manages its own cache.

**Critical caveat (and the fix):** static cache **auto-compiles keyed on tensor shape**,
so a naive setup recompiles (~50 s) on *every new prompt length* — fine for benchmarks,
unusable for interactive chat. The launcher fixes shapes: **left-pad every prompt to a
fixed length (512), always generate a fixed count (256), and warm up once at load.**
Verified: after warmup, different prompts/lengths all run at ~72–99 tok/s with **zero
recompiles**.

**`torch.compile(mode="reduce-overhead")` rejected:** ~0% gain over static cache on dense
models (90 s warmup wasted) and it **crashes on MoE** — CUDA-graph capture can't handle
the grouped-GEMM expert kernels (`Cannot copy between CPU and CUDA tensors during CUDA
graph capture`). The static-cache fast path is the better, simpler lever for both.

---

# Update — quantization is SLOWER than bf16 on this no-nvcc box

Tested Qwen3-8B decode (256 tok, warm) across quantizers vs bf16+static cache:

| Config | tok/s | VRAM |
|---|---:|---:|
| **bf16 + static cache** | **73.8** | 16.4 GB |
| bitsandbytes NF4 | 30.9 | 6.1 GB |
| bitsandbytes NF4 + static | 27.5 | 6.1 GB |
| FP8 (transformers FineGrained) | 16.4 | 9.4 GB |
| torchao FP8 weight-only | 10.0 | 9.4 GB |
| torchao FP8 dynamic-act | 10.1 | 9.4 GB |
| torchao int4 | conversion error | — |

**Conclusion:** with no `nvcc`, there is no fused/marlin int4 or tuned FP8 matmul, so every
quantizer falls back to dequant-then-matmul and runs **2–7× slower** than reading bf16
weights with a good matmul — even FP8, despite Blackwell's fp8 tensor cores. On molab,
quantization buys **VRAM, not speed**. Therefore the playground uses **bf16 + static cache**
for every model that fits, and reserves quantization for (a) gpt-oss's native MXFP4 and
(b) fitting models too big for bf16 (>~40 B, accepting the speed hit).
