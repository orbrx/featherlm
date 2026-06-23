# TODO — models & speed research

## Goal
Give molab users the **fastest possible experience for any model they pick**. Since molab
sandboxes are identical (RTX PRO 6000 Blackwell 96 GB, no nvcc), we curate the
best-known **load config (esp. quantization) for max decode speed** per model and expose a
clean API + widget. Every featured model should run "to the best of our knowledge."

## Fast-quantization research (the key open question)
On this no-nvcc box, quantization is NOT automatically faster (marlin needs nvcc; gptqmodel
torch backend is slow). Must **measure** the fastest *working* path per model:
- [ ] **bitsandbytes** NF4 / INT8 (prebuilt kernels — Blackwell sm_120 support?)
- [ ] **torchao** int4_weight_only / fp8 (compile-friendly tinygemm — Blackwell?)
- [ ] **MXFP4** on-the-fly for non-gpt-oss models (transformers Mxfp4Config + triton kernels)
- [ ] **bf16 + static KV cache** (current baseline; sometimes still fastest for small models)
- [ ] Combine winner with static cache + (where safe) torch.compile; verify correctness.
- [ ] Decide per-size: small (≤8B), mid (14–32B), MoE, and pick the fastest correct config.

## Models to add + test (availability + fastest quant per model)
*Many are 2026 releases — first confirm they're present on the molab image, then find the
fastest working config.*
- [ ] north-mini-code-1.0
- [ ] mistral-medium-3.5
- [ ] qwen3.6 (27B, 35B)
- [ ] all variants of qwen3.5
- [ ] variations of gemma 4
- [ ] lfm2.5 (LiquidAI)
- [ ] nemotron-3-super
- [ ] nemotron3 33B
- [ ] granite 4.1
- [ ] laguna xs.2
- [ ] mistral-3
- [ ] devstral-small-2
- [ ] qwen3-next (80B-A3B)
- [ ] deepseek-r1
- [ ] llama 3.1 / 3.2
- [ ] phi3
- [ ] phi4

## Notebook direction
- [ ] Move the measured-speed table OUT of the notebook → README/BENCHMARKS.
- [ ] Clean API: a small `Model`/`chat()` surface users can call in their own cells
      (load, generate, stream) + the widget on top.
- [ ] Apply the best-known fast config per featured model (quantized where it's faster).
- [ ] (Later) native tool-calling / ReAct loop (generalized run_python) + multi-turn history.
