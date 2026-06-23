import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # 🦣 Running the Largest Possible LLM on a single RTX PRO 6000 (96 GB)

    **Goal:** push a single **NVIDIA RTX PRO 6000 Blackwell (96 GB VRAM)** + 160 GB RAM
    as far as it will go on open-weight LLMs.

    **What we achieved (largest → fastest):**

    | Model | Params | How it runs here | Speed |
    |---|---|---|---|
    | **Qwen3-235B-A22B** (GPTQ-Int4) | **235 B** | 70 layers on GPU + 24 on CPU (offload) | ~0.11 tok/s |
    | **gpt-oss-120B** (native MXFP4) | **117 B** | fully on GPU (65 GB) | ~8.5 tok/s |

    The interactive demo below runs **gpt-oss-120B** — 117 B parameters living entirely
    in VRAM, the *largest model that still runs at interactive speed* on this box.
    The 235 B run is documented at the bottom.

    *Key tricks:* Blackwell-native **MXFP4** triton kernels (no nvcc needed) for gpt-oss,
    and **GPTQ-Int4 + explicit GPU/CPU device-map offload** for the 235 B stretch.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import os, time, subprocess, torch

    # Writable HF cache on the fast overlay disk
    os.environ["HF_HOME"] = "/home/marimo/hfcache"
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    mo.md("✅ Environment ready — transformers + torch on CUDA " + torch.version.cuda)
    return AutoModelForCausalLM, AutoTokenizer, mo, subprocess, time, torch


@app.cell
def _(mo, subprocess, torch):
    def _hw():
        q = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free,driver_version",
             "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()
        name, tot, used, free, drv = [x.strip() for x in q.split(",")]
        ram = subprocess.run(["bash","-lc","free -g | awk '/Mem:/{print $2}'"], capture_output=True, text=True).stdout.strip()
        cap = torch.cuda.get_device_capability()
        return name, tot, used, free, drv, ram, cap

    _name,_tot,_used,_free,_drv,_ram,_cap = _hw()
    gpu_total_gb = round(int(_tot)/1024, 1)
    mo.md(f"""
    ## 🖥️ Hardware
    | | |
    |---|---|
    | **GPU** | {_name} |
    | **Compute capability** | sm_{_cap[0]}{_cap[1]} (Blackwell) |
    | **VRAM total / free** | {gpu_total_gb} GB / {round(int(_free)/1024,1)} GB |
    | **Driver / CUDA** | {_drv} / {torch.version.cuda} |
    | **System RAM** | {_ram} GB |
    | **GPU bf16 sanity** | {"✅ pass" if (torch.randn(512,512,device="cuda",dtype=torch.bfloat16)@torch.randn(512,512,device="cuda",dtype=torch.bfloat16)).isfinite().all().item() else "❌"} |
    """)
    return


@app.cell
def _(AutoModelForCausalLM, AutoTokenizer, mo, time, torch):
    # Load gpt-oss-120B (117B params, native MXFP4) fully into VRAM.
    # Blackwell-native MXFP4 triton kernels need no nvcc. Loads once (~30s).
    MODEL_ID = "openai/gpt-oss-120b"
    HF_CACHE = "/home/marimo/hfcache/hub"   # explicit: kernel baked HF_HUB_CACHE early
    _t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=HF_CACHE)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype="auto", device_map="cuda", cache_dir=HF_CACHE)
    _load_s = time.time() - _t0
    _vram = torch.cuda.memory_allocated() / 1e9
    mo.md(
        f"### 🧠 Loaded **`{MODEL_ID}`**\n\n"
        f"- **117 B** parameters (MoE, 128 experts) — quantized **MXFP4**\n"
        f"- Resident in **{_vram:.1f} GB** of VRAM, fully on GPU\n"
        f"- Load time: **{_load_s:.0f}s**"
    )
    return model, tok


@app.cell
def _(mo):
    # Interactive controls for the 117B model
    prompt_box = mo.ui.text_area(
        value="Explain, in 3 sentences, how a Mixture-of-Experts transformer routes each token to only a few experts.",
        label="Prompt", rows=3, full_width=True)
    max_tok = mo.ui.slider(32, 512, value=160, step=16, label="Max new tokens", show_value=True)
    gen_button = mo.ui.run_button(label="🚀 Generate with gpt-oss-120B")
    mo.vstack([prompt_box, max_tok, gen_button])
    return gen_button, max_tok, prompt_box


@app.cell
def _(gen_button, max_tok, mo, model, prompt_box, time, tok, torch):
    # Generates on the 117B model when the button is clicked
    mo.stop(not gen_button.value, mo.callout("Edit the prompt and click **🚀 Generate** above.", kind="info"))
    _messages = [{"role": "user", "content": prompt_box.value}]
    _inp = tok.apply_chat_template(_messages, add_generation_prompt=True,
                                   return_tensors="pt", return_dict=True).to(model.device)
    _t = time.time()
    with torch.no_grad():
        _out = model.generate(**_inp, max_new_tokens=int(max_tok.value), do_sample=False)
    _dt = time.time() - _t
    _n = int(_out.shape[1] - _inp["input_ids"].shape[1])
    _resp = tok.decode(_out[0, _inp["input_ids"].shape[1]:], skip_special_tokens=True)
    mo.md(f"**⏱ {_n} tokens in {_dt:.1f}s — {_n/_dt:.1f} tok/s** · `gpt-oss-120B` (117B, MXFP4, 65 GB VRAM)\n\n---\n\n{_resp}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 📊 How far this 96 GB GPU goes

    Empirically validated on this box (RTX PRO 6000 Blackwell, 96 GB VRAM + 160 GB RAM):

    | Model | Total params | Quant | Placement | Speed | Status |
    |---|---|---|---|---|---|
    | **Qwen3-235B-A22B** | **235 B** | GPTQ-Int4 | 70 layers GPU + 24 CPU | ~0.11 tok/s | ✅ ran, coherent |
    | **gpt-oss-120B** | **117 B** | MXFP4 | fully on GPU (65 GB) | ~8.5 tok/s | ✅ interactive (above) |

    **`llmfit` cross-check** (`llmfit recommend`): its curated catalog tops out near
    **~80–87 B** for a "Perfect" GPU fit at high speed (35–610 tok/s, e.g. Qwen3-Next-80B,
    MiniMax-class). It's a great *scout* for fast models, but doesn't list the 120 B / 235 B
    checkpoints — so the real hardware ceiling sits well above its recommendations.

    ### What made it work on Blackwell (sm_120, CUDA 13)
    - **MXFP4** (gpt-oss): Blackwell-native triton kernels via `kernels` 0.12.x — **no nvcc** needed, correct output at true 4-bit.
    - **GPTQ-Int4 + torch backend** (Qwen3-235B): `gptqmodel`'s pure-torch kernel sidesteps the marlin/nvcc requirement; an **explicit GPU/CPU `device_map`** offloads 24 of 94 layers to RAM so a 235 B model fits.
    - Dead ends: compressed-tensors **W4A16** silently decompresses to bf16 (≈470 GB) and produced garbage; **AWQ marlin** needs `CUDA_HOME`/nvcc (absent).

    **Bottom line:** the largest model that *runs* here is **Qwen3-235B-A22B (235 B)**;
    the largest that runs *fast/interactively* is **gpt-oss-120B (117 B)**, demoed live above.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 🐘 Reproducible recipe — the absolute-largest run (235 B)

    The interactive demo keeps **gpt-oss-120B** resident. To instead run the
    **235 B** Qwen3-235B-A22B (frees VRAM first, ~10 min load, ~0.1 tok/s), use:

    ```python
    import os, time, torch
    os.environ["HF_HOME"] = "/home/marimo/hfcache"
    from gptqmodel import GPTQModel, BACKEND

    MID = "Qwen/Qwen3-235B-A22B-GPTQ-Int4"   # 117 GB of GPTQ-Int4 weights
    N, GPU_LAYERS = 94, 70                     # 70 layers on GPU (~93 GB), 24 on CPU
    dm = {"model.embed_tokens": 0, "model.norm": 0, "model.rotary_emb": 0, "lm_head": 0}
    for i in range(N):
        dm[f"model.layers.{i}"] = 0 if i < GPU_LAYERS else "cpu"

    # TORCH backend avoids marlin's nvcc/CUDA_HOME requirement on Blackwell
    m = GPTQModel.load(MID, backend=BACKEND.TORCH, device_map=dm)
    tok = m.tokenizer
    msgs = [{"role": "user", "content": "Why can a 235B MoE run on one GPU server?"}]
    inp = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True)
    inp = {k: v.to("cuda") for k, v in inp.items()}
    out = m.model.generate(**inp, max_new_tokens=70, do_sample=False)
    print(tok.decode(out[0, inp["input_ids"].shape[1]:], skip_special_tokens=True))
    ```

    Measured here: **load 571 s**, **93.4 GB VRAM**, generation **~0.11 tok/s** —
    slow because the 24 CPU-offloaded layers are shuttled over PCIe every token, but
    the 235 B model genuinely runs and produces coherent text.
    """)
    return


if __name__ == "__main__":
    app.run()
