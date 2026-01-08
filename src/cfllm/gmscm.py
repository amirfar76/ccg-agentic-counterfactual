# cfllm/gmscm.py
# Minimal GM-SCM-compatible wrappers for this project:
# - We expose gmscm_action_pair and gmscm_report_pair with flexible signatures
# - Internally we use HF generate() with seeds to control randomness
# - This is *not* the full stepwise Gumbel-Max implementation; it is a stable
#   interface that unblocks your pipeline. You can later swap internals with the
#   paper's sampler without changing call sites.

import os
import torch
from typing import Optional, Tuple, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM

# Environment overrides (optional)
_DEFAULT_MODEL = os.environ.get("CFLLM_GMSCM_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Single-model cache to avoid repeated downloads/loads
_MODEL = None
_TOKENIZER = None
_MODEL_NAME = None

def _load_model(model_name: Optional[str] = None):
    global _MODEL, _TOKENIZER, _MODEL_NAME
    name = model_name or _DEFAULT_MODEL
    if _MODEL is not None and _MODEL_NAME == name:
        return _MODEL, _TOKENIZER

    # make tokenizers behave nicely on fork
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    _TOKENIZER = AutoTokenizer.from_pretrained(name, use_fast=True)
    # pad token falls back to eos if missing (common for chat LLMs)
    if _TOKENIZER.pad_token_id is None and _TOKENIZER.eos_token_id is not None:
        _TOKENIZER.pad_token = _TOKENIZER.eos_token

    dtype = torch.float16 if _DEVICE == "cuda" else torch.float32
    _MODEL = AutoModelForCausalLM.from_pretrained(
        name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    _MODEL.to(_DEVICE)
    _MODEL.eval()
    _MODEL_NAME = name
    return _MODEL, _TOKENIZER

def _seed_everything(seed: int):
    if seed is None:
        return
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

def _generate_once(
    prompt: str,
    *,
    model_name: Optional[str],
    seed: int,
    temperature: float,
    top_p: Optional[float],
    top_k: Optional[int],
    max_new_tokens: int,
) -> str:
    model, tok = _load_model(model_name)
    _seed_everything(seed)

    # basic chat-style prompt -> plain prompt (works fine for these small templates)
    inputs = tok(prompt, return_tensors="pt").to(_DEVICE)
    gen_args = dict(
        do_sample=True,
        temperature=float(temperature),
        max_new_tokens=int(max_new_tokens),
        pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
        eos_token_id=tok.eos_token_id,
    )
    if top_p is not None:
        gen_args["top_p"] = float(top_p)
    if top_k is not None:
        gen_args["top_k"] = int(top_k)

    with torch.no_grad():
        out = model.generate(**inputs, **gen_args)
    text = tok.decode(out[0], skip_special_tokens=True)

    # Try to strip the prompt to return only the completion
    try:
        in_text = tok.decode(inputs["input_ids"][0], skip_special_tokens=True)
        if text.startswith(in_text):
            text = text[len(in_text):].lstrip()
    except Exception:
        pass
    return text

# --------------------------------------------------------------------------
# Public wrappers expected by cfllm.llm
# --------------------------------------------------------------------------

def gmscm_action_pair(
    user_factual: str,
    user_cf: str,
    *,
    model_name: Optional[str] = None,
    seed: int = 0,
    temperature: float = 0.7,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    max_new_tokens: int = 160,
) -> Tuple[str, str, Dict[str, Any]]:
    """
    Returns (action_text_factual, action_text_cf, meta).
    For compatibility, both are generated the same way; llm.action_from_prompt
    typically calls this with the same prompt for both and only uses the first.
    """
    sys = (
        "You are a helpful assistant that outputs a short JSON dictionary describing an LTE simulation action. "
        "Prefer concise keys like num_ues, scheduler, traffic_mbps, duration_s. Do not add commentary."
    )
    pf = f"{sys}\nUSER:\n{user_factual}\nASSISTANT:\n"
    pc = f"{sys}\nUSER:\n{user_cf}\nASSISTANT:\n"

    a_f = _generate_once(
        pf,
        model_name=model_name,
        seed=seed,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_new_tokens=max_new_tokens,
    )
    a_c = _generate_once(
        pc,
        model_name=model_name,
        seed=seed + 1,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_new_tokens=max_new_tokens,
    )
    meta = {"engine": "gmscm", "model": model_name or _DEFAULT_MODEL}
    return a_f.strip(), a_c.strip(), meta

def gmscm_report_pair(
    system_prompt: str,
    facts_factual: str,
    facts_cf: str,
    *,
    model_name: Optional[str] = None,
    seed: int = 0,
    temperature: float = 0.7,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    max_new_tokens: int = 220,
) -> Tuple[str, str, Dict[str, Any]]:
    """
    Returns (report_text, report_text_cf, meta).
    In this project, callers often only use the first element. We nevertheless
    return two (second one with seed+1) for future-proofing.
    """
    pf = f"{system_prompt}\n\nFacts:\n{facts_factual}\n\nWrite the report:\n"
    pc = f"{system_prompt}\n\nFacts:\n{facts_cf}\n\nWrite the report:\n"

    y_f = _generate_once(
        pf,
        model_name=model_name,
        seed=seed,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_new_tokens=max_new_tokens,
    )
    y_c = _generate_once(
        pc,
        model_name=model_name,
        seed=seed + 1,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_new_tokens=max_new_tokens,
    )
    meta = {"engine": "gmscm", "model": model_name or _DEFAULT_MODEL}
    return y_f.strip(), y_c.strip(), meta
