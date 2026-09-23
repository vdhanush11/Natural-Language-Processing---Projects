const state = {
  categories: { main_categories: [], sub_categories: [] }
};

const $ = (id) => document.getElementById(id);

function toast(message) {
  const el = $("toast");
  el.textContent = message;
  el.classList.add("show");
  setTimeout(() => el.classList.remove("show"), 3200);
}

async function api(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options
  });

  const data = await response.json();

  if (!response.ok) {
    throw new Error(data.detail || "Request failed");
  }

  return data;
}

function money(value) {
  if (value === null || value === undefined || value === "") return "—";
  const n = Number(value);
  if (Number.isNaN(n)) return String(value);
  return `₹${n.toLocaleString("en-IN", { maximumFractionDigits: 0 })}`;
}

function productCard(product) {
  const image = product.image || "";
  const link = product.link || "";
  return `
    <article class="product-card">
      <img class="product-image"
           src="${escapeAttr(image)}"
           alt=""
           loading="lazy"
           onerror="this.style.visibility='hidden'">
      <div class="product-body">
        <div class="product-name">${escapeHtml(product.name || "Unnamed product")}</div>
        <div class="product-meta">
          ${escapeHtml(product.main_category || "Unknown")} ·
          ${escapeHtml(product.sub_category || "Unknown")}
        </div>
        <div class="score-row">
          <span class="score">${Number(product.similarity_score || 0).toFixed(4)} similarity</span>
          <span class="price">${money(product.discount_price)}</span>
        </div>
        <div class="score-row">
          <span class="product-meta">ID ${product.catalog_id} · Rating ${escapeHtml(product.ratings ?? "—")}</span>
          ${link ? `<a class="open-link" href="${escapeAttr(link)}" target="_blank" rel="noopener">View source ↗</a>` : ""}
        </div>
      </div>
    </article>
  `;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function escapeAttr(value) {
  return escapeHtml(value);
}

function fillSelect(id, values, firstLabel) {
  const select = $(id);
  select.innerHTML = `<option value="">${firstLabel}</option>` +
    values.map(v => `<option value="${escapeAttr(v)}">${escapeHtml(v)}</option>`).join("");
}

function updateSubCategories() {
  const main = $("main-category").value;
  const values = main
    ? state.categories.sub_categories_by_main[main] || []
    : state.categories.sub_categories;
  fillSelect("sub-category", values, "All sub-categories");
}

async function loadStatus() {
  try {
    const [status, categories] = await Promise.all([
      api("/api/status"),
      api("/api/categories")
    ]);

    state.categories = categories;
    fillSelect("main-category", categories.main_categories, "All categories");
    updateSubCategories();
    fillSelect("catalogue-category", categories.main_categories, "All categories");

    $("system-status").textContent =
      `${status.products.toLocaleString()} products loaded`;

    $("health-dot").classList.add("online");
    $("health-text").textContent = "API online";

    $("stats").innerHTML = `
      <div class="stat"><strong>${status.products.toLocaleString()}</strong><span>products</span></div>
      <div class="stat"><strong>${status.main_categories}</strong><span>categories</span></div>
      <div class="stat"><strong>${status.sub_categories}</strong><span>sub-categories</span></div>
    `;
  } catch (err) {
    $("system-status").textContent = "Backend unavailable";
    $("health-text").textContent = "Offline";
    toast(err.message);
  }
}

function getQueryPayload(compare = false) {
  const text = compare ? $("compare-query").value.trim() : $("query-text").value.trim();
  const idRaw = compare ? $("compare-catalog-id").value : $("catalog-id").value;
  const payload = {};

  if (text) {
    payload.query_text = text;
  } else if (idRaw !== "") {
    payload.catalog_id = Number(idRaw);
  }

  if (compare) {
    payload.top_n = Number($("compare-top-n").value);
  } else {
    payload.model = $("model").value;
    payload.top_n = Number($("top-n").value);
    payload.main_category = $("main-category").value || null;
    payload.sub_category = $("sub-category").value || null;
  }

  return payload;
}

async function recommend() {
  const payload = getQueryPayload(false);

  if (!payload.query_text && payload.catalog_id === undefined) {
    toast("Enter product text or a catalogue ID.");
    return;
  }

  $("results").innerHTML = `<div class="empty">Finding similar products…</div>`;

  try {
    const data = await api("/api/recommend", {
      method: "POST",
      body: JSON.stringify(payload)
    });

    const query = data.query;
    $("query-summary").classList.remove("hidden");
    $("query-summary").innerHTML =
      `<strong>${escapeHtml(query.name || query.text || "Query")}</strong>
       <span class="product-meta">Model: ${escapeHtml(data.model)} · ${data.recommendations.length} recommendations</span>`;

    $("results").innerHTML = data.recommendations.length
      ? data.recommendations.map(productCard).join("")
      : `<div class="empty">No matching products were found with the selected filters.</div>`;
  } catch (err) {
    $("results").innerHTML = `<div class="empty">${escapeHtml(err.message)}</div>`;
    toast(err.message);
  }
}

async function compareModels() {
  const payload = getQueryPayload(true);

  if (!payload.query_text && payload.catalog_id === undefined) {
    toast("Enter product text or a catalogue ID.");
    return;
  }

  $("compare-results").innerHTML = `<div class="empty">Running all three models. Semantic models may take longer on first use.</div>`;

  try {
    const data = await api("/api/compare", {
      method: "POST",
      body: JSON.stringify(payload)
    });

    $("compare-results").innerHTML = Object.entries(data).map(([model, result]) => {
      const items = result.recommendations || [];
      const note = model === "TF-IDF"
        ? "Lexical baseline"
        : model === "Word2Vec"
          ? "Dense word embeddings"
          : "Dense subword embeddings";

      return `
        <section class="model-column">
          <h3>${model}</h3>
          <div class="model-note">${note}</div>
          ${items.map((p, index) => `
            <div class="mini-result">
              <strong>${index + 1}. ${escapeHtml(p.name || "Unnamed product")}</strong>
              <span>${Number(p.similarity_score || 0).toFixed(4)}</span>
            </div>
          `).join("")}
        </section>
      `;
    }).join("");
  } catch (err) {
    $("compare-results").innerHTML = `<div class="empty">${escapeHtml(err.message)}</div>`;
    toast(err.message);
  }
}

async function loadCatalogue() {
  const search = $("catalogue-search").value.trim();
  const category = $("catalogue-category").value;

  $("catalogue-results").innerHTML = `<div class="empty">Loading catalogue…</div>`;

  try {
    const params = new URLSearchParams();
    if (search) params.set("search", search);
    if (category) params.set("main_category", category);
    params.set("limit", "50");

    const data = await api(`/api/products?${params.toString()}`);

    if (!data.products.length) {
      $("catalogue-results").innerHTML = `<div class="empty">No products found.</div>`;
      return;
    }

    $("catalogue-results").innerHTML = data.products.map(p => `
      <div class="catalogue-row">
        <img src="${escapeAttr(p.image || "")}" alt="" loading="lazy"
             onerror="this.style.visibility='hidden'">
        <div>
          <strong>${escapeHtml(p.name || "Unnamed product")}</strong>
          <span>ID ${p.catalog_id} · ${escapeHtml(p.main_category || "Unknown")} · ${escapeHtml(p.sub_category || "Unknown")}</span>
        </div>
        <button class="use-id" data-id="${p.catalog_id}">Use ID</button>
      </div>
    `).join("");

    document.querySelectorAll(".use-id").forEach(btn => {
      btn.addEventListener("click", () => {
        $("catalog-id").value = btn.dataset.id;
        $("query-text").value = "";
        switchView("recommendation");
        toast(`Catalogue ID ${btn.dataset.id} selected.`);
      });
    });
  } catch (err) {
    $("catalogue-results").innerHTML = `<div class="empty">${escapeHtml(err.message)}</div>`;
  }
}

function switchView(view) {
  document.querySelectorAll(".nav-item").forEach(item => {
    item.classList.toggle("active", item.dataset.view === view);
  });

  document.querySelectorAll(".view").forEach(section => {
    section.classList.toggle("active", section.id === `${view}-view`);
  });

  const titles = {
    recommendation: ["Product recommendations", "Find semantically similar products from the catalogue."],
    compare: ["Compare models", "Compare TF-IDF, Word2Vec and FastText on the same query."],
    catalogue: ["Catalogue explorer", "Browse products and use any catalogue ID as a query."]
  };

  $("page-title").textContent = titles[view][0];
  $("page-subtitle").textContent = titles[view][1];

  if (view === "catalogue") loadCatalogue();
}

document.querySelectorAll(".nav-item").forEach(item => {
  item.addEventListener("click", () => switchView(item.dataset.view));
});

$("recommend-form").addEventListener("submit", (event) => {
  event.preventDefault();
  recommend();
});

$("compare-btn").addEventListener("click", compareModels);
$("catalogue-search-btn").addEventListener("click", loadCatalogue);
$("main-category").addEventListener("change", updateSubCategories);

$("catalogue-search").addEventListener("keydown", event => {
  if (event.key === "Enter") loadCatalogue();
});

loadStatus();
