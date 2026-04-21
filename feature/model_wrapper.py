from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import torch
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN", "")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.2-3B-Instruct")
MODEL_ID = MODEL_NAME

_tokenizer = None
_model = None


def _get_env(key: str, default=None, required: bool = False, cast=None):
    value = os.getenv(key, default)

    if required and value is None:
        raise ValueError(f"Missing required env var: {key}")

    if cast and value is not None:
        try:
            value = cast(value)
        except Exception as e:
            raise ValueError(f"Failed to cast env var {key}: {e}")

    return value


def _extract_json(text: str) -> Dict[str, Any]:
    if not text:
        return {"label": "error", "reason": "empty model output", "raw": text}

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1]
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    return {"label": "error", "reason": "could not parse JSON", "raw": text}


def load_model():
    global _tokenizer, _model

    if _tokenizer is not None and _model is not None:
        return _tokenizer, _model

    if HF_TOKEN:
        try:
            login(token=HF_TOKEN)
        except Exception as e:
            print(f"HF login warning: {e}")

    kwargs = {}
    if HF_TOKEN:
        kwargs["token"] = HF_TOKEN

    print(f"Loading tokenizer for {MODEL_ID}...")
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, **kwargs)

    if _tokenizer.pad_token_id is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    print(f"Loading model for {MODEL_ID}...")
    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        **kwargs,
    )
    _model.eval()

    print("Model loaded.")
    return _tokenizer, _model


def relation_model_wrapper(messages: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Expects chat-style messages:
    [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."},
    ]
    """
    tokenizer, model = load_model()

    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        )

        device = model.device if hasattr(model, "device") else next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=160,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated = tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        parsed = _extract_json(generated)
        parsed.setdefault("raw", generated)
        return parsed

    except Exception as e:
        return {
            "label": "error",
            "reason": f"model wrapper failure: {str(e)}",
            "raw": "",
        }