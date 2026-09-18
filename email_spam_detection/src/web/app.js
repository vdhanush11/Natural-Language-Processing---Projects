const state = { overview: null };
const $ = (selector) => document.querySelector(selector);

async function request(path, options) {
  const response = await fetch(path, options);
  if (!response.ok) throw new Error(`Request failed: ${response.status}`);
  return response.json();
}

function formatDate(value) {
  if (!value) return "Unknown";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString([], { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" });
}

function renderOverview(data) {
  state.overview = data;
  $("#stat-scans").textContent = data.scans.toLocaleString();
  $("#stat-threats").textContent = data.threats.toLocaleString();
  $("#stat-rate").innerHTML = `${data.detection_rate}<sup>%</sup>`;
  $("#nav-count").textContent = data.quarantine;
}

async function loadOverview() {
  try { renderOverview(await request("/api/overview")); } catch (error) { console.error(error); }
}

async function loadQuarantine() {
  const body = $("#quarantine-body");
  try {
    const { items } = await request("/api/quarantine?limit=50");
    body.innerHTML = items.length ? items.map((item) => `<tr><td>${escapeHtml(item.text)}</td><td>${item.confidence == null ? "—" : `${Math.round(item.confidence * 100)}%`}</td><td>${formatDate(item.quarantined_at)}</td><td><span class="table-status">HELD</span></td></tr>`).join("") : `<tr><td colspan="4" class="table-empty">No threats have been held yet.</td></tr>`;
  } catch (error) { body.innerHTML = `<tr><td colspan="4" class="table-empty">Unable to load the quarantine ledger.</td></tr>`; }
}

function escapeHtml(value) { const div = document.createElement("div"); div.textContent = value || ""; return div.innerHTML; }

function setView(view) {
  const dashboard = $("#dashboard-view");
  const quarantine = $("#quarantine-view");
  const isQuarantine = view === "quarantine";
  dashboard.classList.toggle("hidden", isQuarantine);
  quarantine.classList.toggle("hidden", !isQuarantine);
  $("#view-label").textContent = isQuarantine ? "Quarantine" : "Command center";
  document.querySelectorAll(".nav-item").forEach((item) => item.classList.toggle("active", item.dataset.view === view));
  if (isQuarantine) loadQuarantine();
}

function showResult(result) {
  const isSpam = result.label === "spam";
  const percent = Math.round(result.confidence * 100);
  $("#empty-result").classList.add("hidden");
  $("#result-content").classList.remove("hidden");
  $("#result-icon").textContent = isSpam ? "!" : "✓";
  $("#result-icon").style.background = isSpam ? "var(--coral)" : "#6aa767";
  $("#result-eyebrow").textContent = isSpam ? "THREAT DETECTED" : "MESSAGE CLEARED";
  $("#result-title").innerHTML = isSpam ? "Hold this message." : "Looks safe to send.";
  $("#result-badge").textContent = isSpam ? "SPAM" : "HAM";
  $("#result-badge").style.background = isSpam ? "#f7c7bf" : "#cfe7c7";
  $("#result-badge").style.color = isSpam ? "#a94737" : "#477449";
  $("#confidence-value").textContent = `${percent}%`;
  $("#confidence-bar").style.width = `${percent}%`;
  $("#confidence-bar").style.background = isSpam ? "var(--coral)" : "#6aa767";
  $("#spam-score").textContent = `${Math.round((result.scores.spam || 0) * 100)}%`;
  $("#ham-score").textContent = `${Math.round((result.scores.ham || 0) * 100)}%`;
  $("#action-note").innerHTML = isSpam ? `<span>↳</span><p><strong>Action taken</strong><br>Message held in quarantine for review.</p>` : `<span>↳</span><p><strong>Action taken</strong><br>No threat found. Message cleared.</p>`;
}

$("#scan-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const textarea = $("#email-text");
  const button = event.target.querySelector("button[type=submit]");
  button.disabled = true;
  button.querySelector("span").textContent = "Analyzing...";
  try {
    const result = await request("/api/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text: textarea.value }) });
    showResult(result);
    await loadOverview();
  } catch (error) { alert("The model could not analyze this message. Check that the API is running."); }
  button.disabled = false;
  button.querySelector("span").textContent = "Analyze message";
});

$("#email-text").addEventListener("input", (event) => { $("#char-count").textContent = `${event.target.value.length.toLocaleString()} / 100,000`; });
document.querySelectorAll(".sample-button").forEach((button) => button.addEventListener("click", () => { $("#email-text").value = button.dataset.sample; $("#email-text").dispatchEvent(new Event("input")); $("#email-text").focus(); }));
$("#clear-result").addEventListener("click", () => { $("#result-content").classList.add("hidden"); $("#empty-result").classList.remove("hidden"); });
$("#refresh-btn").addEventListener("click", async () => { await loadOverview(); if (!$("#quarantine-view").classList.contains("hidden")) await loadQuarantine(); });
$("#quarantine-refresh").addEventListener("click", loadQuarantine);
document.querySelectorAll(".nav-item").forEach((item) => item.addEventListener("click", (event) => { event.preventDefault(); window.location.hash = item.dataset.view; setView(item.dataset.view); }));
window.addEventListener("hashchange", () => setView(window.location.hash.slice(1) || "dashboard"));
loadOverview();
setView(window.location.hash.slice(1) || "dashboard");
