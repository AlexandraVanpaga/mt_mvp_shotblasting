from __future__ import annotations

import logging
from pathlib import Path

import torch
from fastapi import Depends

from app.core.config import Settings, settings
from app.services.ct2_engine import Ctranslate2Engine
from app.services.glossary import Glossary
from app.services.mt_engine import MarianEngine
from app.services.nllb_engine import NllbEngine
from app.services.postedit import PostEditor
from app.services.qwen_postedit import QwenPostEditService
from app.services.translation import MTEngine  # re-exported for FastAPI typing

logger = logging.getLogger(__name__)


def get_settings() -> Settings:
    return settings

_engine_singleton: MTEngine | None = None
_engine_singleton_key: tuple | None = None


def _mt_engine_cache_key(cfg: Settings) -> tuple:
    if cfg.mt_engine == "ctranslate2":
        return (
            "ctranslate2",
            str(cfg.ct2_model_dir.resolve()),
            cfg.ct2_compute_type,
            cfg.mt_model_name,
            cfg.device,
        )
    if cfg.mt_engine == "nllb":
        return (
            "nllb",
            cfg.mt_model_name,
            cfg.device,
            cfg.nllb_src_lang,
            cfg.nllb_tgt_lang,
            cfg.nllb_num_beams,
            cfg.nllb_dtype,
        )
    return ("marian_hf", cfg.mt_model_name, cfg.device)


def _resolve_mt_engine(cfg: Settings) -> MTEngine:
    """Return cached engine, with graceful CT2 → marian_hf fallback.

    If ``mt_engine == "ctranslate2"`` but the converted weights are missing or
    fail to load (e.g. corrupted ``model.bin``, unsupported compute type on the
    host), log a warning and fall back to the HuggingFace Marian engine instead
    of crashing the request. This keeps the container useful even when the
    operator forgot to mount ``./models``.
    """
    global _engine_singleton, _engine_singleton_key
    key = _mt_engine_cache_key(cfg)
    if _engine_singleton is None or _engine_singleton_key != key:
        if cfg.mt_engine == "ctranslate2":
            try:
                _engine_singleton = Ctranslate2Engine(
                    cfg.ct2_model_dir,
                    tokenizer_model_name=cfg.mt_model_name,
                    device=cfg.device,
                    compute_type=cfg.ct2_compute_type,
                )
            except (FileNotFoundError, RuntimeError, OSError) as exc:
                logger.warning(
                    "CTranslate2 engine unavailable (%s); falling back to "
                    "HuggingFace Marian (%s). To use CT2, convert the model "
                    "via scripts/convert_marian_to_ct2.py and mount it at %s.",
                    exc,
                    cfg.mt_model_name,
                    cfg.ct2_model_dir,
                )
                _engine_singleton = MarianEngine(cfg.mt_model_name, cfg.device)
                key = ("marian_hf", cfg.mt_model_name, cfg.device)
        elif cfg.mt_engine == "nllb":
            _engine_singleton = NllbEngine(
                model_name=cfg.mt_model_name,
                device=cfg.device,
                src_lang=cfg.nllb_src_lang,
                tgt_lang=cfg.nllb_tgt_lang,
                dtype=cfg.nllb_dtype,
                num_beams=cfg.nllb_num_beams,
            )
        else:
            _engine_singleton = MarianEngine(cfg.mt_model_name, cfg.device)
        _engine_singleton_key = key
    return _engine_singleton


def get_mt_engine(cfg: Settings = Depends(get_settings)) -> MTEngine:
    return _resolve_mt_engine(cfg)


_glossary_cache: dict[str, Glossary] = {}


def get_glossary(cfg: Settings = Depends(get_settings)) -> Glossary:
    key = str(cfg.glossary_path.resolve())
    if key not in _glossary_cache:
        _glossary_cache[key] = Glossary(Path(key))
    return _glossary_cache[key]


_qwen_singleton: QwenPostEditService | None = None
_qwen_singleton_key: tuple[str, int, int, bool] | None = None
# After a failed load we skip retrying the same config every request (OOM / killed during shards).
_qwen_load_aborted_key: tuple[str, int, int, bool] | None = None


def _resolve_qwen(cfg: Settings) -> QwenPostEditService | None:
    if not cfg.postedit_use_qwen:
        return None
    global _qwen_singleton, _qwen_singleton_key, _qwen_load_aborted_key
    key = (
        cfg.postedit_qwen_model,
        cfg.postedit_max_new_tokens,
        cfg.postedit_qwen_max_input_tokens,
        cfg.postedit_force_cpu,
    )
    if _qwen_load_aborted_key == key:
        return None
    if _qwen_singleton is None or _qwen_singleton_key != key:
        try:
            _qwen_singleton = QwenPostEditService(
                cfg.postedit_qwen_model,
                max_new_tokens=cfg.postedit_max_new_tokens,
                max_input_tokens=cfg.postedit_qwen_max_input_tokens,
                force_cpu=cfg.postedit_force_cpu,
            )
            _qwen_singleton_key = key
            _qwen_load_aborted_key = None
        except (torch.OutOfMemoryError, MemoryError) as exc:
            _qwen_singleton = None
            _qwen_singleton_key = None
            _qwen_load_aborted_key = key
            logger.error(
                "Qwen post-edit model failed to load (%s). Neural post-edit is disabled for this "
                "process. Fixes: set MT_MVP_POSTEDIT_FORCE_CPU=true (Qwen on CPU, MT can use GPU), "
                "set MT_MVP_POSTEDIT_USE_QWEN=false, close other GPU consumers (e.g. Docker), or "
                "use a smaller MT+Qwen footprint.",
                exc.__class__.__name__,
            )
            return None
    return _qwen_singleton


def get_qwen_postedit(cfg: Settings = Depends(get_settings)) -> QwenPostEditService | None:
    return _resolve_qwen(cfg)


def get_posteditor(
    cfg: Settings = Depends(get_settings),
    qwen: QwenPostEditService | None = Depends(get_qwen_postedit),
) -> PostEditor:
    return PostEditor(cfg.postedit_prompt_path, qwen=qwen)
