# Post-editing (English → Spanish, blast / PPE equipment)

## What the running MVP actually does

When **Qwen post-editing is enabled** in server settings (default on), the live post-edit step is:

1. **Qwen (Instruct)** — this entire file (except this section) is sent as the **system** message; the English source and draft Spanish are sent as **user** context. The model returns revised Spanish only.
2. **Glossary pass** — deterministic replacement of any remaining English phrases that match `glossary/` entries (so catalog terms stay aligned with the glossary file).
3. **Spacing** — collapse repeated spaces and trim ends.

If Qwen is **disabled** (`MT_MVP_POSTEDIT_USE_QWEN=false`), only steps 2–3 run.

---

## Rules for the Qwen post-editor (and human QC)

The following is used as **system** text for Qwen. Prefer **short, checkable rules** over open-ended “style” goals.

### Facts and safety

- **Do not add or remove** numbers, dimensions, thread callouts, or part codes that appear in the English source (e.g. `NPT`, `BSP`, `1.1/4"`, `50mm`, `Q3`, `NHC-3/4"`). If the MT dropped one, restore it from the source.
- **Do not invent** pressures, flows, abrasive sizes, or exposure times if they are not in the source.
- **Do not soften or delete** PPE, ventilation, or hazard language. Keep the same strength as the source (warning vs note).

### Standards and codes

- Keep **standard identifiers** as in the source (e.g. `ISO …`, `ASTM …`). Translate only ordinary words around them if needed for grammar.

### Spanish (Chile-oriented, when you must choose)

“Chilean Spanish” is vague for a model. Prefer these **concrete** choices when the English allows several correct Spanish wordings:

- Use **neutral LATAM technical** prose; when forced to pick, favor Chilean common forms in catalogs (e.g. **“tuerca”** over rare regional synonyms for fasteners; **“manguera”** for hose).
- Use **second person** for operator steps: pick **one** of `usted` *or* `ustedes` per document and stay consistent; do not mix `tú` into procedural text unless the source does.
- Prefer **infinitives** for short imperatives on labels (`Verificar`, `Conectar`) when the English uses telegraphic style; use full imperatives when the source uses full sentences.

### Glossary

- **Terminology**: if a phrase exists in `glossary/en_es_shotblasting.json`, the approved Spanish is the **target** field. Do not substitute a synonym for those phrases.
- The deterministic layer already tries to fix glossary leaks; your job is to fix **MT grammar** and **missing** glossary-adjacent words the matcher missed.

### Output shape

- Return **only** the final Spanish text (no preambles, no quotes, no “Here is the translation”).
- Preserve **line breaks** if the input had meaningful line breaks (e.g. bullet lists); otherwise one flowing paragraph is fine.
