import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")

with app.setup:
    # The reusable engine lives in `featherlm` (one file) — not this notebook. It installs
    # into molab's base env (keeping its GPU torch — featherlm declares no torch dep).
    import subprocess, sys, os, gc, time
    os.environ.setdefault("HF_HOME", "/home/marimo/hfcache")
    try:
        import featherlm
    except Exception:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--root-user-action=ignore",
                        "git+https://github.com/orbrx/featherlm.git"])
        import featherlm
    try:                              # gpt-oss MXFP4 extra (Blackwell-native kernels)
        import kernels  # noqa: F401
    except Exception:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--root-user-action=ignore",
                        "kernels>=0.12,<0.13"])
    import marimo as mo
    import torch
    from transformers import PreTrainedModel

    # Curated dropdown. featherlm.load() also takes ANY raw HF id.
    MODELS = {
        "Qwen3-4B-Instruct  · dense 4B · bf16":      dict(id="Qwen/Qwen3-4B-Instruct-2507", kind="bf16"),
        "Qwen3-8B           · dense 8B · bf16":      dict(id="Qwen/Qwen3-8B",               kind="bf16"),
        "Qwen3-14B          · dense 14B · bf16":     dict(id="Qwen/Qwen3-14B",              kind="bf16"),
        "Phi-4              · dense 14B · bf16":     dict(id="microsoft/phi-4",             kind="bf16"),
        "Qwen3-32B          · dense 32B · bf16":     dict(id="Qwen/Qwen3-32B",              kind="bf16"),
        "★ Qwen3-30B-A3B  · MoE 30B/3B act · bf16":  dict(id="Qwen/Qwen3-30B-A3B-Instruct-2507", kind="bf16"),
        "gpt-oss-20B    · MoE 20B/3.6B act · MXFP4": dict(id="openai/gpt-oss-20b",  kind="mxfp4"),
        "gpt-oss-120B   · MoE 117B/5.1B act · MXFP4":dict(id="openai/gpt-oss-120b", kind="mxfp4"),
        "Gemma-3-27B-it (gated*) · dense 27B · bf16":dict(id="google/gemma-3-27b-it", kind="bf16"),
        "⚠️ Qwen3-235B-A22B · GPTQ-Int4 · GPU+CPU offload (~0.1 tok/s, ~10min)":
            dict(id="Qwen/Qwen3-235B-A22B-GPTQ-Int4", kind="gptq_offload"),
    }


@app.cell
def _():
    mo.md(r"""
    # 🎛️ LLM Playground — molab · RTX PRO 6000 Blackwell (96 GB)

    **Pick a model → load → chat** (live streaming, with a reasoning panel). The whole
    engine is the reusable **[`featherlm`](./featherlm.py)** library — use it anywhere:

    ```python
    import featherlm
    llm = featherlm.load("Qwen/Qwen3-8B")          # any HF id; fastest config auto-picked
    reasoning, answer = llm.generate("Explain entropy.", thinking=True)
    for ev in llm.stream("Tell me a joke."):        # streaming generator
        print(ev["piece"], end="")
    ```

    Catalog, per-model execution paths, measured speeds, and publish/import notes:
    **[README](./README.md)**.
    """)
    return


@app.cell
def _():
    def _free_gb():
        f = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True).stdout.strip()
        return round(int(f) / 1024, 1) if f.isdigit() else "?"
    mo.md(f"## 🖥️ {torch.cuda.get_device_name(0)} · **{_free_gb()} GB free** / 96 GB · "
          f"CUDA {torch.version.cuda} · featherlm {featherlm.__name__}")
    return


@app.cell
def _():
    get_loaded, set_loaded = mo.state({"name": None, "llm": None})
    return get_loaded, set_loaded


@app.cell
def _():
    model_picker = mo.ui.dropdown(options=list(MODELS.keys()), value=list(MODELS.keys())[0],
                                  label="Pick a model")
    load_button = mo.ui.run_button(label="📥 Load model")
    mo.vstack([mo.md("### 🎛️ Model loader"), model_picker, load_button,
               mo.md("*Loading frees the previous model first. bf16 compiles once (~1 min); "
                     "the 235B is a ~10 min GPU+CPU offload.*")])
    return load_button, model_picker


@app.cell
def _(get_loaded, load_button, model_picker, set_loaded):
    mo.stop(not load_button.value, mo.callout("Pick a model and click **📥 Load model**.", kind="neutral"))
    _sel = model_picker.value
    _cur = get_loaded()
    if _cur["name"] == _sel and _cur["llm"] is not None:
        _out = mo.callout(f"✅ **{_sel}** already loaded · {torch.cuda.memory_allocated()/1e9:.1f} GB VRAM",
                          kind="success")
    else:
        set_loaded({"name": None, "llm": None})
        import warnings as _warnings                 # free every previous model (meta-device trick)
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            for _o in list(gc.get_objects()):
                try:
                    if isinstance(_o, PreTrainedModel):
                        _o.to("meta")
                except Exception:
                    pass
        gc.collect(); torch.cuda.empty_cache()
        _spec = MODELS[_sel]; _t0 = time.time()
        with mo.status.spinner(title=f"Loading {_spec['id']}…"):
            _llm = featherlm.load(_spec["id"], kind=_spec["kind"])
        set_loaded({"name": _sel, "llm": _llm})
        _out = mo.callout(f"✅ Loaded **{_sel}** ({_llm.fam}) in {time.time()-_t0:.0f}s · "
                          f"{torch.cuda.memory_allocated()/1e9:.1f} GB VRAM", kind="success")
    _out
    return


@app.cell
def _():
    prompt_box = mo.ui.text_area(value="Explain how Mixture-of-Experts routing works, with a short example.",
                                 label="Prompt", rows=3, full_width=True)
    think_switch = mo.ui.switch(value=False, label="🧠 Let the model think (slower, better)")
    seed_num = mo.ui.number(start=0, stop=99999, value=3407, label="🎲 seed")
    gen_button = mo.ui.run_button(label="🚀 Generate")
    mo.vstack([prompt_box, mo.hstack([think_switch, seed_num, gen_button], justify="start", gap=1.5)])
    return gen_button, prompt_box, seed_num, think_switch


@app.cell
def _(gen_button, get_loaded, prompt_box, seed_num, think_switch):
    _st = get_loaded()
    mo.stop(_st["llm"] is None, mo.callout("⬆️ Load a model above first.", kind="info"))
    mo.stop(not gen_button.value,
            mo.callout(f"Model **{_st['name'].split('·')[0].strip()}** ready — type a prompt and "
                       f"click **🚀 Generate**.", kind="info"))
    _llm = _st["llm"]

    def _scroll(text, h=300):
        _safe = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        _js = ("navigator.clipboard.writeText(this.nextElementSibling.querySelector('.rtxt').innerText);"
               "this.textContent='✓ Copied'")
        return mo.Html(
            f'<div style="position:relative"><button onclick="{_js}" style="position:absolute;top:6px;'
            f'right:10px;z-index:3;font-size:0.7rem;padding:2px 9px;border:1px solid rgba(0,0,0,0.15);'
            f'border-radius:6px;background:var(--marimo-background-color,#fff);cursor:pointer">📋 Copy</button>'
            f'<div style="max-height:{h}px;overflow:auto;display:flex;flex-direction:column-reverse;'
            f'border:1px solid rgba(0,0,0,0.12);border-radius:10px;background:rgba(0,0,0,0.03)">'
            f'<div class="rtxt" style="white-space:pre-wrap;font-family:ui-monospace,SFMono-Regular,Menlo,'
            f'monospace;font-size:0.8rem;line-height:1.5;padding:0.7rem 0.9rem">{_safe}</div></div></div>')

    def _panel(reasoning, answer, busy, foot=""):
        _b = []
        if reasoning:
            _b += [mo.md("🧠 **Reasoning** " + ("*(streaming…)*" if busy else "")), _scroll(reasoning, 240)]
        _b += [mo.md("💬 **Answer** " + ("*(streaming…)*" if busy and not reasoning else "")),
               _scroll(answer or " ", 380)]
        if foot:
            _b.append(mo.md(foot))
        return mo.vstack(_b)

    mo.output.replace(_panel("", "", True))
    _ev, _i, _t0 = None, 0, time.time()
    for _ev in _llm.stream(prompt_box.value, thinking=think_switch.value, seed=int(seed_num.value)):
        _i += 1
        if _i % 12 == 0:
            mo.output.replace(_panel(_ev["reasoning"], _ev["answer"], True))
    _dt = time.time() - _t0
    if _ev is not None:
        _ntok = len(_llm.tok(_ev["raw"], add_special_tokens=False)["input_ids"])
        _mode = "🧠 thinking" if think_switch.value else "fast"
        _foot = (f"**⏱ {_ntok} tokens in {_dt:.1f}s — {_ntok/max(_dt,1e-9):.1f} tok/s** · "
                 f"{_st['name'].split('·')[0].strip()} · {_mode}")
        mo.output.replace(_panel(_ev["reasoning"], _ev["answer"], False, _foot))
    return


if __name__ == "__main__":
    app.run()
