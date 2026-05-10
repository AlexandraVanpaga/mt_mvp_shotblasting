const $ = (id) => document.getElementById(id);

function fillExample(text) {
  $("src").value = text;
  $("src").focus();
}

document.querySelector(".chips")?.addEventListener("click", (ev) => {
  const btn = ev.target.closest(".chip");
  if (!btn || !btn.dataset.text) return;
  fillExample(btn.dataset.text);
});

async function translate() {
  const src = $("src").value.trim();
  const out = $("out");
  const err = $("err");
  const dbg = $("debug");
  const btn = $("go");
  const cacheNote = $("cacheNote");

  err.classList.add("hidden");
  dbg.classList.add("hidden");
  cacheNote.classList.add("hidden");
  err.textContent = "";
  dbg.textContent = "";

  if (!src) {
    err.textContent = "Paste or type English blast-equipment text first.";
    err.classList.remove("hidden");
    return;
  }

  btn.disabled = true;
  out.value = "Translating…";

  try {
    const res = await fetch("/api/v1/translate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text: src,
        apply_glossary: $("useGlossary").checked,
        apply_postedit: $("usePostedit").checked,
        include_debug: $("useDebug").checked,
      }),
    });

    if (!res.ok) {
      const t = await res.text();
      throw new Error(t || `HTTP ${res.status}`);
    }

    const data = await res.json();
    out.value = data.translation || "";

    if (data.from_cache) {
      cacheNote.classList.remove("hidden");
    }

    if ($("useDebug").checked && data.debug) {
      dbg.textContent = JSON.stringify(data.debug, null, 2);
      dbg.classList.remove("hidden");
    }
  } catch (e) {
    out.value = "";
    err.textContent = e instanceof Error ? e.message : String(e);
    err.classList.remove("hidden");
  } finally {
    btn.disabled = false;
  }
}

$("go").addEventListener("click", translate);
