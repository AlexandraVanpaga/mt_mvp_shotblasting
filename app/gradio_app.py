"""Gradio UI that drives the same translation pipeline as the HTTP API.

Why a Gradio app exists alongside the FastAPI frontend
------------------------------------------------------
* The vanilla ``frontend/`` page is the production UI used inside Docker.
* The Gradio app is meant for **demos to clients** — a single command spins
  up a public ``https://*.gradio.live`` link they can open from anywhere
  (no port forwarding, no Docker needed on their end).

Two entry points expose this UI:

* ``python -m app.gradio_app`` → standalone server on ``http://0.0.0.0:7860``.
  Pass ``--share`` to print a temporary public URL.
* ``app.main:create_app`` mounts the same Gradio app at ``/gradio`` so the
  FastAPI deployment exposes both the REST API (``/api/v1``), the vanilla web
  page (``/``), and the Gradio demo (``/gradio``) under one process.

The Gradio handler calls the *exact same* ``run_translate`` function the API
uses, so glossary, MT, and Qwen post-edit behaviour are guaranteed to match.
"""

from __future__ import annotations

import argparse
from typing import Any

from app.api.deps import _resolve_mt_engine, _resolve_qwen, get_glossary
from app.api.schemas import TranslateRequest, TranslateResponse
from app.core.config import settings
from app.services.postedit import PostEditor
from app.services.translate_cache import build_translate_cache_key, translate_cache
from app.services.translation import run_translate

_EXAMPLES = [
    [
        "Inspect the nylon blast nozzle holders and abrasive control valve "
        "before starting the blast cabinet.",
        True,
        True,
    ],
    [
        "The Spartan Helmet and breathing tube Spartan must be checked for "
        "damage after each shift in the blasting area.",
        True,
        True,
    ],
    [
        "Replace the Endurolite blast hoses if the outer cover shows wear "
        "before pressurizing the system.",
        True,
        True,
    ],
    [
        "Connect the Pet Cock valve on the rear of the helmet to the "
        "compressed airline using the supplied NPT fitting.",
        True,
        True,
    ],
]


def _components() -> tuple[Any, ...]:
    """Build the Gradio components lazily so importing this module is cheap."""
    import gradio as gr  # noqa: PLC0415 - kept local; gradio is a soft dep

    return (
        gr,
        gr.Textbox(
            label="English source",
            placeholder=(
                "Example: The Respirator inner lens gasket Titan must be inspected "
                "and replaced regularly to maintain a tight seal."
            ),
            lines=6,
        ),
        gr.Checkbox(value=True, label="Use equipment glossary"),
        gr.Checkbox(value=True, label="Post-edit (Spanish polish, Qwen 2.5)"),
        gr.Textbox(label="Spanish target", lines=6),
    )


_glossary = None
_engine = None
_qwen = None


def _ensure_pipeline_loaded() -> None:
    """Load MT / glossary / Qwen once and cache the singletons.

    Gradio's request handlers run on a worker thread per request; loading
    the models on the first request avoids paying for ``Settings`` / Marian
    at import time (which would prolong ``app.main`` startup).
    """
    global _glossary, _engine, _qwen
    if _glossary is None:
        _glossary = get_glossary(settings)
    if _engine is None:
        _engine = _resolve_mt_engine(settings)
    if _qwen is None and settings.postedit_use_qwen:
        _qwen = _resolve_qwen(settings)


def translate_handler(source_en: str, use_glossary: bool, use_postedit: bool) -> str:
    """Translate one EN string and return the Spanish output (or an error).

    Routes through the same ``translate_cache`` the REST API uses, so a
    repeated click on the same English text returns instantly without
    re-running Marian + Qwen.
    """
    if not source_en or not source_en.strip():
        return ""
    try:
        _ensure_pipeline_loaded()
        assert _glossary is not None and _engine is not None
        body = TranslateRequest(
            text=source_en,
            apply_glossary=use_glossary,
            apply_postedit=use_postedit,
            include_debug=False,
        )
        cache_key = build_translate_cache_key(body, settings)
        hit = translate_cache.get(cache_key)
        if hit is not None:
            return TranslateResponse.model_validate(hit).translation

        posteditor = PostEditor(settings.postedit_prompt_path, qwen=_qwen if use_postedit else None)
        out = run_translate(body, settings, _glossary, _engine, posteditor)
        translate_cache.set(cache_key, out.model_dump(mode="python"))
        return out.translation
    except Exception as exc:  # noqa: BLE001 - surface readable error to the user
        return f"[error] {type(exc).__name__}: {exc}"


def build_ui() -> Any:
    """Construct (but do not launch) the Gradio Blocks UI."""
    gr, source_box, use_glossary, use_postedit, target_box = _components()
    with gr.Blocks(title="Blast equipment EN→ES translator", theme=gr.themes.Soft()) as ui:
        gr.Markdown(
            """
            # Blast equipment · EN → ES

            Glossary-backed machine translation for shotblasting and PPE
            catalogue copy. The pipeline is **Glossary protect → Marian MT
            → Glossary restore → Qwen 2.5 post-edit → Glossary re-assert**.

            Toggle the checkboxes to compare MT-only with glossary protection
            and full post-editing.
            """
        )
        with gr.Row():
            with gr.Column():
                source_box.render()
                use_glossary.render()
                use_postedit.render()
                go = gr.Button("Translate to Spanish", variant="primary")
            with gr.Column():
                target_box.render()
        gr.Examples(
            examples=_EXAMPLES,
            inputs=[source_box, use_glossary, use_postedit],
            outputs=[target_box],
            fn=translate_handler,
            cache_examples=False,
            label="Example inputs",
        )
        go.click(
            fn=translate_handler,
            inputs=[source_box, use_glossary, use_postedit],
            outputs=[target_box],
        )
    return ui


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Bind address (default 0.0.0.0 — required for Docker port forwarding).",
    )
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument(
        "--share",
        action="store_true",
        help="Publish a temporary https://*.gradio.live link for client demos.",
    )
    args = parser.parse_args()

    ui = build_ui()
    ui.queue().launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
