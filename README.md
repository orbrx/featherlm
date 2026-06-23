# featherlm

**Lightweight LLM inference with batteries included.** Run optimal LLM inference in
environments where only `torch` is available — e.g. [molab](https://molab.run) GPU
notebooks with no CUDA toolchain.

`featherlm` loads any Hugging Face causal LM with the **fastest config for your GPU**, then
gives you `generate` / `stream` / `chat` (with tools) and per-family reasoning parsing — all
in one importable file.

## Install

```bash
pip install git+https://github.com/orbrx/featherlm.git
# with extras (gpt-oss MXFP4 + >120B offload):
pip install "featherlm[all] @ git+https://github.com/orbrx/featherlm.git"
```

featherlm assumes a GPU-matched `torch` is already installed (it never installs torch); it
brings `transformers` + `accelerate`.

## Quickstart

```python
import featherlm
llm = featherlm.load("Qwen/Qwen3-8B")             # any HF id; fastest config auto-picked
reasoning, answer = llm.generate("Explain entropy.", thinking=True)
for ev in llm.stream("Tell me a joke."):           # streaming generator
    print(ev["piece"], end="", flush=True)         # ev = {piece, raw, reasoning, answer}
```

- `load(model_id, kind=None, warmup=True) -> LLM`. `kind` auto-detects: `bf16` for most,
  `mxfp4` for gpt-oss; pass `kind="gptq_offload"` for >120B GPTQ-Int4 with GPU/CPU offload.
- `generate` / `stream` / `chat` return per-family **(reasoning, answer)** splits
  (Qwen `<think>`, Gemma channels, gpt-oss harmony).

## Tool-calling / ReAct

featherlm gives the primitives; you write the loop. `chat(messages, tools=)` is the
multi-turn step, `run_python` is a tiny code sandbox, and `parse_tool_calls` reads native
tool calls (Qwen/Hermes & Gemma dialects).

```python
import featherlm as fl
llm = fl.load("Qwen/Qwen3-8B")
msgs = [{"role": "user", "content": "Use run_python to compute the 15th Fibonacci number."}]
for _ in range(6):                                  # the ReAct loop is yours
    reasoning, answer = llm.chat(msgs, tools=[fl.python_tool()], thinking=True)
    msgs.append({"role": "assistant", "content": answer})
    calls = fl.parse_tool_calls(answer, llm.fam) \
            or ([{"name": "run_python", "arguments": {"code": fl.extract_code(answer)}}]
                if fl.extract_code(answer) else [])
    if not calls:
        break                                       # `answer` is the final response
    for c in calls:
        res = fl.run_python(c["arguments"].get("code", ""), env={"world": world})  # inject vars
        out = (res["stdout"] + ("\n" + res["stderr"] if res["stderr"] else "")).strip()
        msgs.append({"role": "tool", "name": "run_python", "content": out or "(no output)"})
```

`run_python(code, env=None, timeout=10)` runs in a fresh subprocess, injects `env` vars by
value, kills runaways, and never raises. Extend the registry with `featherlm.TOOLS["name"] = fn`.

## Model catalog & speed

`load()` picks the fastest **stable** path per model. Measured decode (greedy, 256 tokens, warm):

| Model | Execution path | tok/s |
|---|---|---:|
| Qwen3-4B / 8B / 14B / 32B, Phi-4 | bf16 + static KV cache | 112 / 74 / 44 / 20 |
| ★ Qwen3-30B-A3B (MoE, 3B active) | bf16 + static KV cache | **85** |
| gpt-oss-20B / 120B (MoE) | MXFP4 (native `kernels`) | 24 / ~14 |
| Gemma-3-27B\* | bf16 + static (eager attn) | ~24 |
| Qwen3-235B-A22B | GPTQ-Int4 + GPU/CPU offload | ~0.1 |

\* gated (needs an HF token). Any other model works by HF id.

The fast path is **bf16 + a static KV cache**: prompts are left-padded to a fixed shape and
warmed up once, so the cache graph compiles a single time and never recompiles (1.1–2.7× over
the dynamic cache). On a GPU without `nvcc`, quantization is *slower* than bf16 (no fused
int4/FP8 kernel), so it's used only to *fit* models too big for bf16. Numbers and method:
[`BENCHMARKS.md`](./BENCHMARKS.md).

## marimo playground

[`notebook.py`](./notebook.py) is a small marimo widget over featherlm — pick a model, toggle
thinking, stream the answer with a reasoning panel. On molab it installs featherlm into the
base env (keeping the preinstalled GPU torch). Open it on molab by replacing `github.com` with
`molab.marimo.io/github` in this repo's URL, or fork it from a shared link.
