const $ = (selector) => document.querySelector(selector);
const state = { categories: [] };

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || "The request could not be completed.");
  return data;
}

function setError(selector, message = "") { $(selector).textContent = message; }
function escapeHtml(value) { return String(value).replace(/[&<>\"]/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[char])); }
function categoryClass(category) { return `category-${category.replace(/[^a-z0-9]/gi, "-")}`; }

function renderResult(result) {
  const panel = $("#result-panel");
  panel.classList.remove("empty");
  panel.innerHTML = `<div class="result-top"><div><p class="eyebrow">Model result</p><h2>${escapeHtml(result.category)}</h2></div><span class="confidence">${Math.round(result.confidence * 100)}% match</span></div>
    <div class="confidence-bar"><span style="width:${Math.max(4, result.confidence * 100)}%"></span></div>
    <div class="result-meta"><span>Top predictions</span><span>${result.alternatives.length} categories compared</span></div>
    <div class="alternatives">${result.alternatives.map((item, index) => `<div class="alternative"><span class="rank">0${index + 1}</span><span class="category-dot ${categoryClass(item.category)}"></span><strong>${escapeHtml(item.category)}</strong><span class="alternative-bar"><i style="width:${Math.max(3, item.confidence * 100)}%"></i></span><span>${Math.round(item.confidence * 100)}%</span></div>`).join("")}</div>
    ${result.cleaned_text ? `<details class="cleaned"><summary>View cleaned text</summary><p>${escapeHtml(result.cleaned_text)}</p></details>` : ""}`;
}

async function loadOverview() {
  try {
    const [health, categories, stats] = await Promise.all([api("/api/health"), api("/api/categories"), api("/api/stats")]);
    state.categories = categories.categories;
    $("#status-dot").classList.add("online");
    $("#status-text").textContent = health.model_loaded ? "Model online" : "Model unavailable";
    $("#total-routed").textContent = stats.total_routed;
    $("#category-filter").innerHTML = '<option value="">All categories</option>' + state.categories.map((category) => `<option value="${escapeHtml(category)}">${escapeHtml(category)}</option>`).join("");
  } catch (error) {
    $("#status-text").textContent = "Offline";
    $("#status-dot").classList.add("offline");
  }
}

async function loadFeeds() {
  const list = $("#feed-list");
  list.innerHTML = '<div class="loading">Loading routed articles...</div>';
  try {
    const category = $("#category-filter").value;
    const data = await api(`/api/feeds?limit=50${category ? `&category=${encodeURIComponent(category)}` : ""}`);
    list.innerHTML = data.articles.length ? data.articles.map((article) => `<article class="feed-item"><div class="feed-item-head"><span class="category-pill ${categoryClass(article.category)}">${escapeHtml(article.category)}</span><time>${new Date(article.timestamp).toLocaleString()}</time></div><p>${escapeHtml(article.text)}</p></article>`).join("") : '<div class="empty-feed"><h3>No routed articles yet</h3><p>Classified stories will appear here when routing is enabled.</p></div>';
  } catch (error) { list.innerHTML = `<div class="empty-feed"><h3>Could not load feeds</h3><p>${escapeHtml(error.message)}</p></div>`; }
}

$("#predict-form").addEventListener("submit", async (event) => {
  event.preventDefault(); setError("#form-error");
  const button = event.currentTarget.querySelector(".primary"); button.disabled = true; button.querySelector("span").textContent = "Classifying...";
  try { renderResult(await api("/api/predict", { method: "POST", body: JSON.stringify({ text: $("#article-text").value, route: $("#route").checked, include_cleaned_text: $("#cleaned").checked, top_k: Number($("#top-k").value) }) })); loadOverview(); }
  catch (error) { setError("#form-error", error.message); }
  finally { button.disabled = false; button.querySelector("span").textContent = "Classify article"; }
});

$("#batch-submit").addEventListener("click", async () => {
  setError("#batch-error");
  const articles = $("#batch-text").value.split(/\n\s*\n|\n/).map((text) => text.trim()).filter(Boolean);
  if (!articles.length) return setError("#batch-error", "Add at least one article.");
  const button = $("#batch-submit"); button.disabled = true; button.firstChild.textContent = "Running... ";
  try { const data = await api("/api/predict/batch", { method: "POST", body: JSON.stringify({ articles: articles.map((text) => ({ text, route: $("#batch-route").checked })) }) }); $("#batch-results").innerHTML = data.results.map((result, index) => `<article class="batch-item"><span class="batch-index">${String(index + 1).padStart(2, "0")}</span><div><span class="category-pill ${categoryClass(result.category)}">${escapeHtml(result.category)}</span><p>${Math.round(result.confidence * 100)}% confidence</p></div></article>`).join(""); loadOverview(); }
  catch (error) { setError("#batch-error", error.message); }
  finally { button.disabled = false; button.firstChild.textContent = "Run batch "; }
});

document.querySelectorAll(".tab").forEach((tab) => tab.addEventListener("click", () => { document.querySelectorAll(".tab, .view").forEach((element) => element.classList.remove("active")); tab.classList.add("active"); $(`#${tab.dataset.view}-view`).classList.add("active"); if (tab.dataset.view === "feeds") loadFeeds(); }));
document.querySelectorAll(".sample").forEach((button) => button.addEventListener("click", () => { $("#article-text").value = button.dataset.sample; $("#article-text").focus(); }));
$("#refresh-feeds").addEventListener("click", loadFeeds);
$("#category-filter").addEventListener("change", loadFeeds);
loadOverview();
