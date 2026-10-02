const $ = (id) => document.getElementById(id);

const state = {
  languages: {},
  lastResult: null,
};

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || "Request failed");
  return data;
}

function toast(message) {
  const el = $("toast");
  el.textContent = message;
  el.classList.add("show");
  setTimeout(() => el.classList.remove("show"), 2600);
}

function populateLanguages(data) {
  state.languages = data.languages;
  const source = $("sourceLanguage");
  const target = $("targetLanguage");

  Object.entries(data.languages).forEach(([code, info]) => {
    const option = new Option(info.name, code);
    source.appendChild(option);
    target.appendChild(new Option(info.name, code));
  });
  target.value = "hin_Deva";
}

function updateCounts() {
  const text = $("inputText").value;
  const words = text.trim() ? text.trim().split(/\s+/).length : 0;
  $("inputCount").textContent = `${text.length} characters · ${words} words`;
}

function setLoading(loading) {
  const btn = $("translateBtn");
  btn.disabled = loading;
  btn.classList.toggle("loading", loading);
  btn.querySelector("span:first-child").textContent = loading ? "Translating…" : "Translate message";
}

function renderResult(result) {
  state.lastResult = result;
  $("outputArea").classList.remove("empty");
  $("outputArea").innerHTML = `<div class="translation-text" lang="${result.target_language}">${escapeHtml(result.translated_text)}</div>`;
  $("outputTitle").textContent = `${result.target_language_name} translation`;
  $("outputMeta").textContent = `${result.source_language_name} → ${result.target_language_name} · ${result.num_input_tokens} input tokens`;
  $("latencyMeta").textContent = `${result.latency_ms} ms`;

  $("copyBtn").disabled = false;
  $("downloadBtn").disabled = false;

  const confidence = Math.round((result.detection.confidence || 0) * 100);
  $("confidenceValue").textContent = `${confidence}%`;
  $("detectedLanguage").textContent = result.source_language_name;
  $("detectionNote").textContent = result.detection.note || "Source language detected successfully.";

  $("detectedHint").textContent = `Detected: ${result.source_language_name} (${confidence}% confidence)`;

  const list = $("candidateList");
  list.innerHTML = (result.detection.candidates || []).slice(0, 3).map(([code, prob]) => `
    <div class="candidate"><span>${state.languages[code]?.name || code}</span><b>${Math.round(prob * 100)}%</b></div>
  `).join("");

  if (result.protected_terms?.length) {
    toast(`${result.protected_terms.length} technical term(s) protected`);
  }
}

function escapeHtml(value) {
  return value.replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;"
  }[c]));
}

async function translate() {
  const text = $("inputText").value.trim();
  if (!text) {
    toast("Enter a support message first.");
    $("inputText").focus();
    return;
  }

  const source = $("sourceLanguage").value || null;
  const target = $("targetLanguage").value;
  setLoading(true);
  $("outputArea").classList.remove("empty");
  $("outputArea").innerHTML = `<div class="loading-state"><div class="spinner"></div><strong>Running NLLB inference…</strong><span>Detecting language, tokenising and decoding</span></div>`;

  try {
    const result = await api("/api/translate", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        text,
        target_language: target,
        source_language: source,
        protect_terms: $("protectTerms").checked,
        num_beams: Number($("beams").value),
        max_new_tokens: Number($("maxTokens").value),
        no_repeat_ngram_size: Number($("ngram").value),
        length_penalty: Number($("lengthPenalty").value),
      })
    });
    renderResult(result);
  } catch (err) {
    $("outputArea").innerHTML = `<div class="error-state"><strong>Translation unavailable</strong><span>${escapeHtml(err.message)}</span></div>`;
    toast(err.message);
  } finally {
    setLoading(false);
  }
}

async function boot() {
  try {
    const data = await api("/api/languages");
    populateLanguages(data);
    const health = await api("/api/health");
    const badge = $("healthBadge");
    if (health.model_loaded) {
      badge.className = "status-pill online";
      badge.innerHTML = "<i></i> Model ready";
    } else {
      badge.className = "status-pill error";
      badge.innerHTML = "<i></i> Model unavailable";
      toast("Model could not be loaded. Check README/model path.");
    }
  } catch (err) {
    $("healthBadge").className = "status-pill error";
    $("healthBadge").innerHTML = "<i></i> API unavailable";
    toast("Could not connect to the FastAPI backend.");
  }
}

$("inputText").addEventListener("input", updateCounts);
$("translateBtn").addEventListener("click", translate);
$("clearBtn").addEventListener("click", () => {
  $("inputText").value = "";
  updateCounts();
  $("outputArea").className = "output-area empty";
  $("outputArea").innerHTML = `<div class="empty-state"><div class="empty-icon">✦</div><strong>Your translation will appear here</strong><span>Choose a target language and translate a support message.</span></div>`;
  $("outputTitle").textContent = "Ready to translate";
  $("outputMeta").textContent = "NLLB-200 · awaiting input";
  $("latencyMeta").textContent = "";
  $("copyBtn").disabled = true;
  $("downloadBtn").disabled = true;
});

$("swapBtn").addEventListener("click", () => {
  const source = $("sourceLanguage");
  const target = $("targetLanguage");
  const currentSource = source.value;
  const currentTarget = target.value;

  if (!currentSource) {
    toast("Auto-detected source cannot be swapped until a translation is made.");
    return;
  }
  source.value = currentTarget;
  target.value = currentSource;

  if (state.lastResult?.translated_text) {
    $("inputText").value = state.lastResult.translated_text;
    updateCounts();
  }
});

document.querySelectorAll(".chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    $("inputText").value = chip.dataset.example;
    updateCounts();
    $("inputText").focus();
  });
});

$("copyBtn").addEventListener("click", async () => {
  if (!state.lastResult) return;
  await navigator.clipboard.writeText(state.lastResult.translated_text);
  toast("Translation copied to clipboard");
});

$("downloadBtn").addEventListener("click", () => {
  if (!state.lastResult) return;
  const blob = new Blob([state.lastResult.translated_text], {type: "text/plain;charset=utf-8"});
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `translation-${state.lastResult.target_language}.txt`;
  a.click();
  URL.revokeObjectURL(url);
  toast("Translation downloaded");
});

$("advancedToggle").addEventListener("click", () => {
  const body = $("advancedBody");
  body.classList.toggle("hidden");
  $("advancedChevron").textContent = body.classList.contains("hidden") ? "+" : "−";
});

const sliders = [
  ["beams", "beamOut", (v) => v],
  ["maxTokens", "tokensOut", (v) => v],
  ["ngram", "ngramOut", (v) => v],
  ["lengthPenalty", "lengthOut", (v) => Number(v).toFixed(1)],
];
sliders.forEach(([input, output, formatter]) => {
  $(input).addEventListener("input", () => $(output).value = formatter($(input).value));
});

boot();
updateCounts();
