from __future__ import annotations

from pathlib import Path

from fastapi import Depends

from app.core.config import Settings, settings
from app.services.ct2_engine import Ctranslate2Engine
from app.services.glossary import Glossary
from app.services.mt_engine import MarianEngine
from app.services.postedit import PostEditor
from app.services.qwen_postedit import QwenPostEditService


def get_settings() -> Settings:
    return settings


_engine_singleton: Ctranslate2Engine | MarianEngine | None = None
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
    return ("marian_hf", cfg.mt_model_name, cfg.device)


def _resolve_mt_engine(cfg: Settings) -> Ctranslate2Engine | MarianEngine:
    global _engine_singleton, _engine_singleton_key
    key = _mt_engine_cache_key(cfg)
    if _engine_singleton is None or _engine_singleton_key != key:
        if cfg.mt_engine == "ctranslate2":
            _engine_singleton = Ctranslate2Engine(
                cfg.ct2_model_dir,
                tokenizer_model_name=cfg.mt_model_name,
                device=cfg.device,
                compute_type=cfg.ct2_compute_type,
            )
        else:
            _engine_singleton = MarianEngine(cfg.mt_model_name, cfg.device)
        _engine_singleton_key = key
    return _engine_singleton


def get_mt_engine(cfg: Settings = Depends(get_settings)) -> Ctranslate2Engine | MarianEngine:
    return _resolve_mt_engine(cfg)


_glossary_cache: dict[str, Glossary] = {}


def get_glossary(cfg: Settings = Depends(get_settings)) -> Glossary:
    key = str(cfg.glossary_path.resolve())
    if key not in _glossary_cache:
        _glossary_cache[key] = Glossary(Path(key))
    return _glossary_cache[key]


_qwen_singleton: QwenPostEditService | None = None
_qwen_singleton_key: tuple[str, int, int] | None = None


def _resolve_qwen(cfg: Settings) -> QwenPostEditService | None:
    if not cfg.postedit_use_qwen:
        return None
    global _qwen_singleton, _qwen_singleton_key
    key = (cfg.postedit_qwen_model, cfg.postedit_max_new_tokens, cfg.postedit_qwen_max_input_tokens)
    if _qwen_singleton is None or _qwen_singleton_key != key:
        _qwen_singleton = QwenPostEditService(
            cfg.postedit_qwen_model,
            max_new_tokens=cfg.postedit_max_new_tokens,
            max_input_tokens=cfg.postedit_qwen_max_input_tokens,
        )
        _qwen_singleton_key = key
    return _qwen_singleton


def get_qwen_postedit(cfg: Settings = Depends(get_settings)) -> QwenPostEditService | None:
    return _resolve_qwen(cfg)


def get_posteditor(
    cfg: Settings = Depends(get_settings),
    qwen: QwenPostEditService | None = Depends(get_qwen_postedit),
) -> PostEditor:
    return PostEditor(cfg.postedit_prompt_path, qwen=qwen)
