const jdInput = document.getElementById("jdInput");
const resumeInput = document.getElementById("resumeInput");
const jdDrop = document.getElementById("jdDrop");
const resumeDrop = document.getElementById("resumeDrop");
const jdBrowse = document.getElementById("jdBrowse");
const resumeBrowse = document.getElementById("resumeBrowse");
const jdFile = document.getElementById("jdFile");
const resumeCount = document.getElementById("resumeCount");
const topK = document.getElementById("topK");
const topKValue = document.getElementById("topKValue");
const screenButton = document.getElementById("screenButton");

const emptyState = document.getElementById("emptyState");
const loadingState = document.getElementById("loadingState");
const resultState = document.getElementById("resultState");
const resultMeta = document.getElementById("resultMeta");
const stats = document.getElementById("stats");
const candidateList = document.getElementById("candidateList");
const downloadButton = document.getElementById("downloadButton");

let latestResults = [];

jdBrowse.addEventListener("click", () => jdInput.click());
resumeBrowse.addEventListener("click", () => resumeInput.click());

jdDrop.addEventListener("click", event => {
  if (!event.target.closest("button")) jdInput.click();
});

resumeDrop.addEventListener("click", event => {
  if (!event.target.closest("button")) resumeInput.click();
});

topK.addEventListener("input", () => {
  topKValue.textContent = topK.value;
});

jdInput.addEventListener("change", () => {
  const file = jdInput.files[0];
  if (!file) return;
  jdFile.textContent = file.name;
  jdFile.classList.remove("hidden");
});

resumeInput.addEventListener("change", () => {
  updateResumeCount();
});

function updateResumeCount() {
  const count = resumeInput.files.length;
  resumeCount.textContent = `${count} resume${count === 1 ? "" : "s"} selected`;
}

function setupDropzone(zone, input, multiple) {
  ["dragenter", "dragover"].forEach(eventName => {
    zone.addEventListener(eventName, event => {
      event.preventDefault();
      zone.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach(eventName => {
    zone.addEventListener(eventName, event => {
      event.preventDefault();
      zone.classList.remove("dragover");
    });
  });

  zone.addEventListener("drop", event => {
    const files = [...event.dataTransfer.files];
    if (!files.length) return;

    if (multiple) {
      input.files = createFileList(files);
      updateResumeCount();
    } else {
      input.files = createFileList([files[0]]);
      const file = input.files[0];
      jdFile.textContent = file.name;
      jdFile.classList.remove("hidden");
    }
  });
}

function createFileList(files) {
  const dataTransfer = new DataTransfer();
  files.forEach(file => dataTransfer.items.add(file));
  return dataTransfer.files;
}

setupDropzone(jdDrop, jdInput, false);
setupDropzone(resumeDrop, resumeInput, true);

screenButton.addEventListener("click", runScreening);

async function runScreening() {
  if (!jdInput.files.length) {
    alert("Please upload a job description.");
    return;
  }

  if (!resumeInput.files.length) {
    alert("Please upload at least one resume.");
    return;
  }

  const formData = new FormData();
  formData.append("jd_file", jdInput.files[0]);

  [...resumeInput.files].forEach(file => {
    formData.append("resumes", file);
  });

  formData.append("top_k", topK.value);

  screenButton.disabled = true;
  emptyState.classList.add("hidden");
  resultState.classList.add("hidden");
  loadingState.classList.remove("hidden");

  try {
    const response = await fetch("/api/screen", {
      method: "POST",
      body: formData
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || "Screening failed.");
    }

    latestResults = data.all_results || [];
    renderResults(data);
  } catch (error) {
    loadingState.classList.add("hidden");
    emptyState.classList.remove("hidden");
    alert(error.message);
  } finally {
    screenButton.disabled = false;
  }
}

function renderResults(data) {
  loadingState.classList.add("hidden");
  resultState.classList.remove("hidden");

  resultMeta.textContent =
    `${data.candidate_count} candidates screened • ${data.model} • ${data.device}`;

  stats.innerHTML = `
    <div class="stat">
      <div class="stat-label">Candidates</div>
      <div class="stat-value">${data.candidate_count}</div>
    </div>
    <div class="stat">
      <div class="stat-label">Shortlisted</div>
      <div class="stat-value">${data.shortlist.length}</div>
    </div>
    <div class="stat">
      <div class="stat-label">JD sections</div>
      <div class="stat-value">${data.jd_sections.length}</div>
    </div>
  `;

  candidateList.innerHTML = data.shortlist.map(candidateCard).join("");
}

function candidateCard(item) {
  const tags = (item.skills || []).slice(0, 7)
    .map(skill => `<span class="tag">${escapeHtml(skill)}</span>`)
    .join("");

  const evidence = Object.entries(item.evidence || {})
    .filter(([, items]) => items && items.length)
    .map(([section, items]) => `
      <div class="evidence-section">
        <div class="evidence-title">${escapeHtml(section.replaceAll("_", " "))}</div>
        ${items.slice(0, 2).map(ev => `
          <div class="evidence-line">
            <span class="evidence-score">${Number(ev.sim).toFixed(3)}</span>
            ${escapeHtml(ev.resume_chunk)}
          </div>
        `).join("")}
      </div>
    `).join("");

  return `
    <article class="candidate-card">
      <div class="candidate-main">
        <div class="rank-badge">#${item.rank}</div>
        <div>
          <div class="candidate-name">${escapeHtml(item.name)}</div>
          <div class="candidate-file">${escapeHtml(item.file_name)}</div>
          <div class="candidate-reason">${escapeHtml(item.reason)}</div>
        </div>
        <div class="score-wrap">
          <div class="score">${Number(item.fit_score).toFixed(1)}</div>
          <div class="score-label">fit score</div>
        </div>
      </div>
      <div class="progress"><span style="width:${Math.max(0, Math.min(100, Number(item.fit_score)))}%"></span></div>
      <div class="card-footer">
        ${tags}
        <span class="tag">${Number(item.years).toFixed(1)} yrs detected</span>
        <button class="evidence-toggle" type="button">View evidence</button>
      </div>
      <div class="evidence">${evidence || '<div class="evidence-line">No section evidence available.</div>'}</div>
    </article>
  `;
}

candidateList.addEventListener("click", event => {
  const button = event.target.closest(".evidence-toggle");
  if (!button) return;

  const card = button.closest(".candidate-card");
  card.classList.toggle("open");
  button.textContent = card.classList.contains("open")
    ? "Hide evidence"
    : "View evidence";
});

downloadButton.addEventListener("click", async () => {
  if (!latestResults.length) return;

  const response = await fetch("/api/export-csv", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({results: latestResults})
  });

  if (!response.ok) {
    alert("Unable to export results.");
    return;
  }

  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "resume_screening_results.csv";
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
});

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}
