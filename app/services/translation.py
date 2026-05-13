"""Core EN→ES translation pipeline (shared by the HTTP router, Gradio UI, and the batch CLI).

The single source of truth for *what* a translation pass does is :func:`run_pipeline`.
Two thin wrappers sit on top of it:

* :func:`run_translate` adds the request/response shape used by the FastAPI router
  (debug payload, pydantic response model).
* ``scripts/translate_csv.py`` uses :func:`run_pipeline` directly to translate the
  full corpus row-by-row.

Keeping the pipeline body in one place means the batch run and the live API can
never drift: ALL-CAPS preprocessing, glossary placeholders, MT, post-edit, and
ALL-CAPS post-processing always execute in the same order with the same arguments.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from app.api.schemas import TranslateRequest, TranslateResponse
from app.core.config import Settings
from app.services.ct2_engine import Ctranslate2Engine
from app.services.glossary import Glossary
from app.services.mt_engine import MarianEngine
from app.services.nllb_engine import NllbEngine
from app.services.postedit import PostEditor
from app.services.text_case import postprocess_after_mt, preprocess_for_mt

# Union of every concrete MT backend. Imported by `app.api.deps`, `app.api.routes`
# and any batch script that wants a single type annotation for the engine handle.
MTEngine = Ctranslate2Engine | MarianEngine | NllbEngine


@dataclass(frozen=True)
class PipelineResult:
    """Output of one full translation pass.

    Carries enough information for both the HTTP debug payload and the batch CLI
    progress log (``was_uppercase`` is used to count ALL-CAPS rewrites; ``raw_mt``
    and ``protected`` are surfaced in the API debug response).
    """

    translation: str
    raw_mt: str
    protected: str
    source_for_mt: str
    was_uppercase: bool
    postedit_applied: bool


def run_pipeline(
    text: str,
    *,
    glossary: Glossary,
    engine: MTEngine,
    posteditor: PostEditor | None,
    apply_glossary: bool,
) -> PipelineResult:
    """Run one English→Spanish translation, engine-agnostic and HTTP-agnostic.

    Stages (executed in order):

      1. **ALL-CAPS detector** — sentence-case the source if it is mostly upper.
         Marian/NLLB were trained on mixed case and otherwise mistranslate long
         shouted safety warnings. The original register is restored at the end.
      2. **Glossary placeholder protection** — replace canonical EN terms with
         ``__GLSN__`` tokens that survive lowercasing and MT tokenization.
      3. **Machine translation** via the configured engine (CT2 Marian by default,
         optionally HF Marian or NLLB-200).
      4. **Glossary placeholder restore** — swap placeholders back to canonical
         Spanish targets.
      5. **Post-edit** (optional) — Qwen 2.5 Instruct sees normal-cased Spanish.
      6. **ALL-CAPS post-process** — UPPER-case the final output iff the source
         was ALL-CAPS in step 1.

    Pass ``posteditor=None`` to skip step 5 entirely.
    """
    source_for_mt, was_uppercase = preprocess_for_mt(text)

    if apply_glossary:
        protected, placeholders = glossary.protect_source(source_for_mt)
    else:
        protected, placeholders = source_for_mt, {}

    raw_mt = engine.translate(protected)
    after_glossary = (
        glossary.enforce_placeholders(raw_mt, placeholders) if apply_glossary else raw_mt
    )

    postedit_applied = False
    after_postedit = after_glossary
    if posteditor is not None:
        after_postedit = posteditor.edit(
            source_en=source_for_mt, target_es=after_glossary, glossary=glossary
        )
        postedit_applied = True

    final = postprocess_after_mt(after_postedit.strip(), was_uppercase)

    return PipelineResult(
        translation=final,
        raw_mt=raw_mt,
        protected=protected,
        source_for_mt=source_for_mt,
        was_uppercase=was_uppercase,
        postedit_applied=postedit_applied,
    )


def run_translate(
    body: TranslateRequest,
    cfg: Settings,
    glossary: Glossary,
    engine: MTEngine,
    posteditor: PostEditor,
) -> TranslateResponse:
    """HTTP-facing wrapper around :func:`run_pipeline`.

    Adds the ``include_debug`` debug payload and shapes the result as a
    :class:`TranslateResponse` for FastAPI.
    """
    result = run_pipeline(
        body.text,
        glossary=glossary,
        engine=engine,
        posteditor=posteditor if body.apply_postedit else None,
        apply_glossary=body.apply_glossary,
    )

    debug = None
    if body.include_debug:
        debug = _build_debug_payload(
            body=body,
            cfg=cfg,
            posteditor=posteditor,
            result=result,
        )

    return TranslateResponse(
        translation=result.translation,
        glossary_applied=body.apply_glossary,
        postedit_applied=result.postedit_applied,
        from_cache=False,
        debug=debug,
    )


def _build_debug_payload(
    *,
    body: TranslateRequest,
    cfg: Settings,
    posteditor: PostEditor,
    result: PipelineResult,
) -> dict:
    resolved_mt_device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    stages: list[str] = [
        "cache_bypassed_for_debug",
        "allcaps_sentence_case" if result.was_uppercase else "allcaps_sentence_case_skipped",
        "glossary_protect" if body.apply_glossary else "glossary_skipped",
        f"mt_{cfg.mt_engine}",
        "glossary_restore_placeholders" if body.apply_glossary else "glossary_restore_skipped",
        "postedit" if body.apply_postedit else "postedit_skipped",
        "allcaps_uppercase_restore"
        if result.was_uppercase
        else "allcaps_uppercase_restore_skipped",
    ]
    return {
        "pipeline_order_on_miss": [
            "cache_check",
            "allcaps_preprocess",
            "glossary_protect_source",
            "machine_translate",
            "glossary_restore_placeholders",
            "postedit",
            "allcaps_postprocess",
        ],
        "stages_executed_this_request": stages,
        "mt_engine": cfg.mt_engine,
        "mt_model": cfg.mt_model_name,
        "mt_resolved_device": resolved_mt_device,
        "cuda_available": torch.cuda.is_available(),
        "ct2_model_dir": str(cfg.ct2_model_dir) if cfg.mt_engine == "ctranslate2" else None,
        "ct2_compute_type": cfg.ct2_compute_type if cfg.mt_engine == "ctranslate2" else None,
        "glossary_path": str(cfg.glossary_path),
        "postedit_prompt_path": str(cfg.postedit_prompt_path),
        "postedit_prompt_chars": len(posteditor.instructions),
        "postedit_use_qwen": cfg.postedit_use_qwen,
        "postedit_force_cpu": cfg.postedit_force_cpu if cfg.postedit_use_qwen else None,
        "postedit_qwen_model": cfg.postedit_qwen_model if cfg.postedit_use_qwen else None,
        "postedit_max_new_tokens": cfg.postedit_max_new_tokens if cfg.postedit_use_qwen else None,
        "postedit_qwen_max_input_tokens": cfg.postedit_qwen_max_input_tokens
        if cfg.postedit_use_qwen
        else None,
        "allcaps_source_detected": result.was_uppercase,
        "source_after_allcaps_preprocess": result.source_for_mt if result.was_uppercase else None,
        "protected_source": result.protected if body.apply_glossary else None,
        "after_mt_before_placeholder_restore": result.raw_mt,
        "cache_hit": False,
    }
