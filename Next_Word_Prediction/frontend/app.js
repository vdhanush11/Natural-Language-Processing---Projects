
const state = { models: {}, selected: "Simple RNN" };

const $ = (id) => document.getElementById(id);

function showError(message) {
  $("error-box").textContent = message;
  $("error-box").classList.remove("hidden");
}
function clearError() {
  $("error-box").classList.add("hidden");
  $("error-box").textContent = "";
}
function setBusy(busy) {
  $("predict-btn").disabled = busy;
  $("generate-btn").disabled = busy || !state.models[state.selected]?.available;
}

function renderModels() {
  const host = $("model-list");
  host.innerHTML = "";
  Object.entries(state.models).forEach(([name, info]) => {
    const button = document.createElement("button");
    button.className = "model-option" + (name === state.selected ? " active" : "") + (!info.available ? " disabled" : "");
    button.disabled = !info.available;
    button.innerHTML = `
      <span class="badge ${info.available ? "" : "off"}">${info.available ? "READY" : "MISSING"}</span>
      <div class="model-name">${name}</div>
      <div class="model-meta">${info.kind}</div>`;
    button.onclick = () => {
      if (!info.available) return;
      state.selected = name;
      renderModels();
      updateHint();
      clearError();
    };
    host.appendChild(button);
  });
  $("chart-model").textContent = state.selected;
  updateHint();
}

function updateHint() {
  const info = state.models[state.selected];
  $("model-hint").textContent = info?.available
    ? `${state.selected} is ready for local inference.`
    : `${state.selected} checkpoint is not installed.`;
  setBusy(false);
}

function renderEvaluation(rows) {
  const host = $("evaluation");
  host.innerHTML = "";
  rows.forEach(row => {
    const model = document.createElement("div");
    model.className = "eval-row";
    const acc = row["Test Accuracy"] == null ? "—" : `${(row["Test Accuracy"]*100).toFixed(2)}%`;
    model.innerHTML = `<span>${row.Model}</span><span>${acc}</span>`;
    host.appendChild(model);
  });
}

function renderBars(predictions) {
  const host = $("prediction-bars");
  host.classList.remove("empty-state");
  host.innerHTML = "";
  predictions.forEach((p, i) => {
    const row = document.createElement("div");
    row.className = "bar-row";
    const label = (p.token || "").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    row.innerHTML = `
      <div class="bar-label" title="${label}">${label || "∅"}</div>
      <div class="bar-track"><div class="bar-fill" style="width:${Math.max(0.5, p.probability*100)}%"></div></div>
      <div class="bar-value">${p.display_probability}</div>`;
    host.appendChild(row);
  });
}

async function api(path, options={}) {
  const response = await fetch(path, {
    headers: {"Content-Type":"application/json"},
    ...options
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || "Request failed.");
  return data;
}

async function loadDashboard() {
  try {
    const health = await api("/api/health");
    state.models = health.models;
    const ready = Object.entries(state.models).find(([,v]) => v.available);
    if (ready) state.selected = ready[0];
    $("status-dot").style.background = "#66e3a4";
    $("status-text").textContent = `${Object.values(state.models).filter(v=>v.available).length}/4 models ready`;
    renderModels();
    const evaluation = await api("/api/evaluation");
    renderEvaluation(evaluation);
  } catch (err) {
    $("status-dot").style.background = "#ff7b8a";
    $("status-text").textContent = "Backend unavailable";
    showError(err.message);
  }
}

$("word-slider").addEventListener("input", e => $("word-count").textContent = e.target.value);
$("sample-btn").onclick = () => $("input-text").value = "the government announced";
$("predict-btn").onclick = async () => {
  clearError();
  const text = $("input-text").value.trim();
  if (!text) return showError("Please enter a sentence.");
  if (!state.models[state.selected]?.available) return showError("This model checkpoint is not available.");
  setBusy(true);
  try {
    const data = await api("/api/predict", {
      method:"POST",
      body: JSON.stringify({text, model:state.selected, top_k:5})
    });
    $("prediction-type").textContent = `Predicted ${data.prediction_type}`;
    $("predicted-next").textContent = data.predicted_next || "—";
    const p = data.top_predictions[0]?.probability ?? 0;
    $("confidence").textContent = `Top prediction confidence: ${(p*100).toFixed(2)}%`;
    $("chart-model").textContent = data.model;
    renderBars(data.top_predictions);
  } catch (err) { showError(err.message); }
  finally { setBusy(false); }
};
$("generate-btn").onclick = async () => {
  clearError();
  const text = $("input-text").value.trim();
  if (!text) return showError("Please enter a sentence.");
  if (!state.models[state.selected]?.available) return showError("This model checkpoint is not available.");
  setBusy(true);
  try {
    const data = await api("/api/generate", {
      method:"POST",
      body: JSON.stringify({text, model:state.selected, num_words:Number($("word-slider").value)})
    });
    $("generated-text").textContent = data.generated_text;
  } catch (err) { showError(err.message); }
  finally { setBusy(false); }
};
loadDashboard();
