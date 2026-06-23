"""featherlm — a tiny universal local-inference layer.

Load any Hugging Face causal LM with the **fastest config for your box**, then
`generate()` (blocking) or `stream()` (generator). Handles per-family sampling and
reasoning parsing for Qwen `<think>`, Gemma channels, and gpt-oss harmony.

Why it exists: on GPUs without a CUDA toolchain (e.g. molab's RTX PRO 6000), 4-bit/FP8
quantizers have no fused kernel and run *slower* than bf16 — so the fastest path is
**bf16 + a static KV cache** (compiled once at a fixed shape). featherlm picks that
automatically (MXFP4 for gpt-oss; optional GPTQ-Int4 + CPU offload for >120B models).

    import featherlm
    llm = featherlm.load("Qwen/Qwen3-8B")          # any HF id; fastest config auto-picked
    reasoning, answer = llm.generate("Explain entropy.", thinking=True)
    for ev in llm.stream("Tell me a joke."):       # streaming generator
        print(ev["piece"], end="", flush=True)

Requires a GPU-matched **torch** to already be installed (featherlm never installs it);
brings `transformers` + `accelerate`. Extras: `kernels` (gpt-oss MXFP4), `gptqmodel`
(GPTQ-Int4 offload).

On molab, add it the native way — a PEP-723 header + uv git source at the top of your
notebook (this is what molab's "add package from git" UI writes):

    # /// script
    # dependencies = ["featherlm", "kernels>=0.12,<0.13"]
    # [tool.uv.sources]
    # featherlm = { git = "https://github.com/orbrx/featherlm" }
    # ///

then just `import featherlm`. (Sanity-check once: `torch.cuda.is_available()` should still
be True on the molab GPU build — i.e. the deps were layered onto the base env, not isolated.)
"""
from __future__ import annotations

import os
import threading

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer,
                           set_seed)

__all__ = ["LLM", "load", "family", "split_reasoning", "DEFAULT_PAD",
           "BUDGET_PLAIN", "BUDGET_THINK",
           "python_tool", "run_python", "parse_tool_calls", "extract_code", "TOOLS"]

DEFAULT_PAD = 512     # left-pad prompts to this so the static-cache graph compiles once
BUDGET_PLAIN = 1024   # max_new_tokens, thinking off
BUDGET_THINK = 4096   # max_new_tokens, thinking on


def family(model_id: str) -> str:
    """Model family from its id: qwen | gemma | gpt-oss | other."""
    m = (model_id or "").lower()
    if "gpt-oss" in m:
        return "gpt-oss"
    if "gemma" in m:
        return "gemma"
    if "qwen" in m:
        return "qwen"
    return "other"


def split_reasoning(text: str, fam: str):
    """Split a reply into (reasoning, answer): gpt-oss harmony (analysis/assistantfinal),
    Gemma channels (<|channel>thought ... <channel|>), or Qwen <think> ... </think>."""
    if fam == "gpt-oss":
        if "assistantfinal" in text:
            r, _, a = text.partition("assistantfinal")
            return r.replace("analysis", "", 1).strip(), a.strip()
        if text.lstrip().startswith("analysis"):
            return text.strip()[len("analysis"):].strip(), ""
        return "", text.strip()
    if "<|channel>thought" in text:
        r, _, a = text.split("<|channel>thought", 1)[1].partition("<channel|>")
    elif "<think>" in text:
        r, _, a = text.split("<think>", 1)[1].partition("</think>")
    else:
        r, a = "", text
    for t in ("<turn|>", "<channel|>", "<|channel>final", "</think>", "<|im_end|>",
              "<end_of_turn>", "<|return|>", "<|end|>"):
        a = a.replace(t, "")
    return r.strip(), a.strip()


# --------------------------------------------------------------------------
# Native tool-calling helpers — build a ReAct loop on top (the loop is yours).
# --------------------------------------------------------------------------

def python_tool():
    """OpenAI-style schema for the built-in run_python sandbox tool."""
    return {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": "Run Python code in a sandbox and return its stdout/stderr. "
                           "Write code that computes the answer and prints it.",
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string", "description": "Python source to run."}},
                "required": ["code"],
            },
        },
    }


def run_python(code, env=None, timeout=10):
    """Tiny code sandbox: run `code` in a fresh subprocess, with optional `env` variables
    injected by value (e.g. env={"world": world}). Never raises into the caller; kills
    runaways. Returns {ok, stdout, stderr, timed_out, dur}."""
    import subprocess as _sub
    import time as _time
    src = ""
    for k, v in (env or {}).items():
        src += f"{k} = {v!r}\n"
    src += code
    t = _time.time()
    try:
        r = _sub.run([__import__("sys").executable, "-c", src],
                     capture_output=True, text=True, timeout=timeout)
        return {"ok": r.returncode == 0, "stdout": r.stdout, "stderr": r.stderr,
                "timed_out": False, "dur": _time.time() - t}
    except _sub.TimeoutExpired:
        return {"ok": False, "stdout": "", "stderr": f"TimeoutError: killed after {timeout}s",
                "timed_out": True, "dur": _time.time() - t}


# A registry you can extend: TOOLS["my_tool"] = my_callable
TOOLS = {"run_python": run_python}


def parse_tool_calls(text, fam="qwen"):
    """Parse native tool calls into [{name, arguments}].
      hermes (Qwen & most): <tool_call>{json}</tool_call>
      gemma:                <|tool_call>call:NAME{arg:<|"|>value<|"|>}<tool_call|>
    For gemma the delimiters are special tokens — keep them (featherlm streams gemma with
    skip_special_tokens=False, so they survive)."""
    import json as _json
    import re as _re
    calls = []
    if fam == "gemma":
        for m in _re.finditer(r"<\|tool_call>call:(\w+)\{(.*?)\}<tool_call\|>", text, _re.S):
            args = dict(_re.findall(r'(\w+):<\|"\|>(.*?)<\|"\|>', m.group(2), _re.S))
            calls.append({"name": m.group(1), "arguments": args})
        return calls
    for raw in _re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, _re.S):
        try:
            calls.append(_json.loads(raw))
        except Exception:
            pass
    return calls


def extract_code(text):
    """Fallback when a model doesn't emit <tool_call>: the last ```python/```json block."""
    import re as _re
    for b in reversed(_re.findall(r"```(?:python|json)?\s*\n?(.*?)```", text, _re.S)):
        b = b.strip()
        if b and not b.startswith("["):
            return b
    return None


def _gen_config(fam, think, budget, pad_id):
    cfg = dict(max_new_tokens=budget, pad_token_id=pad_id)
    if not think:
        cfg["do_sample"] = False
    elif fam == "gemma":
        cfg.update(do_sample=True, temperature=1.0, top_p=0.95, top_k=64)
    else:
        cfg.update(do_sample=True, temperature=0.6, top_p=0.95)
    return cfg


def _build_inputs(tok, prompt, think, fam, device, pad):
    kw = {"enable_thinking": think} if fam in ("qwen", "gemma") else {}
    s = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                add_generation_prompt=True, tokenize=False, **kw)
    ids = tok(s, return_tensors="pt")["input_ids"][0]
    if pad is None:
        mask = torch.ones_like(ids)
        return ({"input_ids": ids.unsqueeze(0).to(device),
                 "attention_mask": mask.unsqueeze(0).to(device)}, int(ids.shape[0]))
    ids = ids[-pad:] if ids.shape[0] > pad else ids
    L = int(ids.shape[0])
    pid = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    inp = torch.cat([torch.full((pad - L,), pid, dtype=ids.dtype), ids])
    mask = torch.cat([torch.zeros(pad - L, dtype=torch.long), torch.ones(L, dtype=torch.long)])
    return ({"input_ids": inp.unsqueeze(0).to(device),
             "attention_mask": mask.unsqueeze(0).to(device)}, pad)


class LLM:
    """A loaded model + the fastest generate/stream path for this box."""

    def __init__(self, model_id, kind=None, device="cuda", warmup=True,
                 cache_dir=None, pad=DEFAULT_PAD):
        self.id = model_id
        self.fam = family(model_id)
        self.device = device
        self.pad = pad
        self.cap = None  # per-model max_new_tokens cap (set for slow offload models)
        if kind is None:
            kind = "mxfp4" if "gpt-oss" in model_id.lower() else "bf16"
        self.kind = kind
        cache_dir = cache_dir or os.path.join(os.environ.get("HF_HOME", "~/.cache/huggingface"), "hub")

        if kind == "gptq_offload":
            self.model, self.tok, self.cap = self._load_gptq_offload(model_id)
            return

        self.tok = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        kw = dict(device_map=device, cache_dir=cache_dir)
        if kind == "mxfp4":
            kw["dtype"] = "auto"            # native MXFP4 (needs `kernels`)
        elif self.fam == "gemma":
            kw.update(dtype=torch.bfloat16, attn_implementation="eager")
        else:
            kw.update(dtype=torch.bfloat16, attn_implementation="sdpa")
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kw).eval()
        if kind == "bf16":
            # fast path: static KV cache compiles on first generate at a fixed shape;
            # warm it up here so every later call is fast with no recompile.
            self.model.generation_config.cache_implementation = "static"
            if warmup:
                enc, _ = _build_inputs(self.tok, "warm up the compiler", False, self.fam, device, self.pad)
                with torch.no_grad():
                    self.model.generate(**enc, **_gen_config(self.fam, False, BUDGET_PLAIN, self.tok.eos_token_id))
                torch.cuda.synchronize()

    @staticmethod
    def _load_gptq_offload(model_id, gpu_layers=70, n_layers=94):
        from gptqmodel import GPTQModel, BACKEND   # extra: pip install gptqmodel
        dm = {"model.embed_tokens": 0, "model.norm": 0, "model.rotary_emb": 0, "lm_head": 0}
        for i in range(n_layers):
            dm[f"model.layers.{i}"] = 0 if i < gpu_layers else "cpu"
        wrap = GPTQModel.load(model_id, backend=BACKEND.TORCH, device_map=dm)
        return wrap.model, wrap.tokenizer, 96

    def _prepare(self, prompt, thinking, max_tokens):
        fast = (self.kind == "bf16") and (not thinking)
        if self.kind == "bf16":
            self.model.generation_config.cache_implementation = "static" if fast else None
        budget = max_tokens or (BUDGET_THINK if thinking else BUDGET_PLAIN)
        if self.cap:
            budget = min(budget, self.cap)
        enc, start = _build_inputs(self.tok, prompt, thinking, self.fam, self.device,
                                   self.pad if fast else None)
        return enc, start, budget, fast

    def generate(self, prompt, thinking=False, seed=3407, max_tokens=None):
        """Blocking generate. Returns (reasoning, answer)."""
        enc, start, budget, _ = self._prepare(prompt, thinking, max_tokens)
        set_seed(int(seed))
        with torch.no_grad():
            out = self.model.generate(**enc, **_gen_config(self.fam, thinking, budget, self.tok.eos_token_id))
        raw = self.tok.decode(out[0, start:], skip_special_tokens=(self.fam != "gemma"))
        return split_reasoning(raw, self.fam)

    def stream(self, prompt, thinking=False, seed=3407, max_tokens=None):
        """Streaming generator. Yields dicts: {piece, raw, reasoning, answer}."""
        enc, _, budget, _ = self._prepare(prompt, thinking, max_tokens)
        set_seed(int(seed))
        streamer = TextIteratorStreamer(self.tok, skip_prompt=True,
                                        skip_special_tokens=(self.fam != "gemma"))
        cfg = _gen_config(self.fam, thinking, budget, self.tok.eos_token_id)
        threading.Thread(target=self.model.generate, daemon=True,
                         kwargs=dict(**enc, streamer=streamer, **cfg)).start()
        raw = ""
        for piece in streamer:
            raw += piece
            r, a = split_reasoning(raw, self.fam)
            yield {"piece": piece, "raw": raw, "reasoning": r, "answer": a}

    def chat(self, messages, tools=None, thinking=False, seed=3407, max_tokens=None, stream=False):
        """Multi-turn chat with optional native tool schemas — the building block for a
        ReAct loop (which you write yourself; see parse_tool_calls / run_python).

        `messages`: list of {"role": "user"|"assistant"|"tool", "content": ...}.
        `tools`:    list of OpenAI-style schemas, e.g. [featherlm.python_tool()].
        Returns (reasoning, answer); if stream=True, a generator of
        {piece, raw, reasoning, answer}. Uses the dynamic cache (flexible-length).

        Tip: append the returned `answer` back as the assistant turn and run
        parse_tool_calls(answer, llm.fam) — tool-call markers live in `answer`.
        """
        if self.kind == "bf16":
            self.model.generation_config.cache_implementation = None  # dynamic: messages vary in length
        budget = max_tokens or (BUDGET_THINK if thinking else BUDGET_PLAIN)
        if self.cap:
            budget = min(budget, self.cap)
        kw = {"enable_thinking": thinking} if self.fam in ("qwen", "gemma") else {}
        if tools:
            kw["tools"] = tools
        enc = self.tok.apply_chat_template(messages, add_generation_prompt=True,
                                           return_tensors="pt", return_dict=True, **kw).to(self.device)
        cfg = _gen_config(self.fam, thinking, budget, self.tok.eos_token_id)
        set_seed(int(seed))
        if stream:
            def _events():
                streamer = TextIteratorStreamer(self.tok, skip_prompt=True,
                                                skip_special_tokens=(self.fam != "gemma"))
                threading.Thread(target=self.model.generate, daemon=True,
                                 kwargs=dict(**enc, streamer=streamer, **cfg)).start()
                raw = ""
                for piece in streamer:
                    raw += piece
                    r, a = split_reasoning(raw, self.fam)
                    yield {"piece": piece, "raw": raw, "reasoning": r, "answer": a}
            return _events()
        start = enc["input_ids"].shape[1]
        with torch.no_grad():
            out = self.model.generate(**enc, **cfg)
        raw = self.tok.decode(out[0, start:], skip_special_tokens=(self.fam != "gemma"))
        return split_reasoning(raw, self.fam)


def load(model_id, **kw) -> "LLM":
    """Convenience: featherlm.load('Qwen/Qwen3-8B') -> LLM."""
    return LLM(model_id, **kw)
