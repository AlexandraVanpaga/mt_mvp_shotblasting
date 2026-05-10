from __future__ import annotations

import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class QwenPostEditService:
    """Qwen2.5 Instruct for neural post-editing (separate from Marian MT)."""

    def __init__(
        self,
        model_name: str,
        max_new_tokens: int = 256,
        max_input_tokens: int = 2048,
    ) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.max_input_tokens = max_input_tokens

        self._tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        elif torch.cuda.is_available():
            dtype = torch.float16
        else:
            dtype = torch.float32

        # Pin whole model on GPU 0 when available (faster than fragmented CPU offload).
        device_map: dict[str, int] | None
        if torch.cuda.is_available():
            device_map = {"": 0}
        else:
            device_map = None

        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        if device_map is None:
            self._model.to("cpu")
        self._model.eval()

    def _model_device(self) -> torch.device:
        return next(self._model.parameters()).device

    @torch.inference_mode()
    def refine(self, source_en: str, draft_es: str, system_instructions: str) -> str:
        user = (
            "Below is a draft Spanish translation of English industrial / blast-equipment text.\n\n"
            f"English source:\n{source_en.strip()}\n\n"
            "Draft Spanish (may contain MT errors). Improve grammar and wording per the system "
            "instructions. Do not add or remove facts, numbers, dimensions, or product codes "
            "from the English source. Preserve catalog product names and glossary Spanish "
            "phrases exactly as in the draft (same spelling and capitalization).\n\n"
            f"{draft_es.strip()}\n\n"
            "Reply with ONLY the improved Spanish. No title, no quotes, no markdown fences."
        )
        messages = [
            {"role": "system", "content": system_instructions.strip()},
            {"role": "user", "content": user},
        ]
        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        )
        dev = self._model_device()
        inputs = {k: v.to(dev) for k, v in inputs.items()}

        out = self._model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self._tokenizer.pad_token_id,
        )
        new_tokens = out[0, inputs["input_ids"].shape[1] :]
        text = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return _strip_markdown_fences(text)


def _strip_markdown_fences(text: str) -> str:
    t = text.strip()
    m = re.match(r"^```(?:\w*)?\s*\n?(.*?)\n?```$", t, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return t
