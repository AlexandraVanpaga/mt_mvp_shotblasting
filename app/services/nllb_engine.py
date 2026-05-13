"""HuggingFace NLLB-200 wrapper (distilled-600M / distilled-1.3B / 3.3B).

NLLB-200 is Meta's multilingual seq2seq model; for EN→ES it produces
noticeably more idiomatic Spanish on out-of-domain text than Marian opus,
at the cost of latency (≈ 2-4× slower per segment on the same GPU).

Tokenization quirks:

* Source language is selected by passing ``src_lang`` to ``AutoTokenizer``.
* Target language is forced via ``forced_bos_token_id`` at generate time —
  the tokenizer's :py:meth:`convert_tokens_to_ids` returns the BOS id for
  ``spa_Latn`` (Latin-script Spanish). Without this, NLLB defaults to
  whatever language the tokenizer was initialised with and silently
  produces wrong-language output.

VRAM footprint on RTX 3060 / 12 GB (bf16):

* ``facebook/nllb-200-distilled-600M``   ≈ 1.3 GB
* ``facebook/nllb-200-distilled-1.3B``  ≈ 2.7 GB
* ``facebook/nllb-200-3.3B``             ≈ 6.8 GB (tight alongside Qwen 3B)
"""

from __future__ import annotations

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class NllbEngine:
    """EN→ES translation via NLLB-200 (HuggingFace)."""

    def __init__(
        self,
        model_name: str = "facebook/nllb-200-distilled-600M",
        device: str | None = None,
        src_lang: str = "eng_Latn",
        tgt_lang: str = "spa_Latn",
        dtype: str = "auto",
        max_new_tokens: int = 512,
        num_beams: int = 4,
    ) -> None:
        self.model_name = model_name
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        torch_dtype: torch.dtype
        if dtype == "auto":
            torch_dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        elif dtype == "fp16":
            torch_dtype = torch.float16
        elif dtype == "bf16":
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float32

        self._tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang=src_lang)
        # ``convert_tokens_to_ids`` works across slow + fast tokenizers; the
        # ``lang_code_to_id`` mapping was removed in transformers ≥ 4.47.
        self._tgt_bos_id = self._tokenizer.convert_tokens_to_ids(tgt_lang)
        if self._tgt_bos_id is None or self._tgt_bos_id == self._tokenizer.unk_token_id:
            raise ValueError(
                f"NLLB tokenizer for {model_name} does not know target language token "
                f"'{tgt_lang}'. Pass a valid Flores-200 code (e.g. 'spa_Latn')."
            )

        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, torch_dtype=torch_dtype
        ).to(self.device)
        self._model.eval()

    @torch.inference_mode()
    def translate(self, text: str) -> str:
        batch = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)
        generated = self._model.generate(
            **batch,
            forced_bos_token_id=self._tgt_bos_id,
            max_new_tokens=self.max_new_tokens,
            num_beams=self.num_beams,
        )
        return self._tokenizer.decode(generated[0], skip_special_tokens=True).strip()
