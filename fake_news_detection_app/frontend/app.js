const API_BASE = "http://127.0.0.1:8001/api";

const titleEl = document.getElementById("title");
const textEl = document.getElementById("text");
const errorEl = document.getElementById("error");
const resultEl = document.getElementById("result");
const explanationEl = document.getElementById("explanation");
const rawEl = document.getElementById("raw");
const healthBadge = document.getElementById("healthBadge");

const examples = [
  {
    title: "Central bank keeps interest rates unchanged",
    text: "The central bank announced its latest policy decision after a scheduled meeting. Officials said they would continue monitoring inflation, employment and economic growth before making further changes."
  },
  {
    title: "BREAKING: miracle cure hidden from the public",
    text: "SHOCKING insiders reveal a secret treatment that cures every disease overnight. Doctors supposedly hate this discovery and governments have allegedly hidden it for decades. Click now before it disappears."
  },
  {
    title: "NASA spacecraft sends new observations from the Sun",
    text: "A NASA spacecraft completed a close pass near the Sun and returned scientific observations about the solar environment. Researchers are analyzing the measurements to better understand solar wind."
  }
];

async function checkHealth() {
  try {
    const response = await fetch(`${API_BASE}/health`);
    const data = await response.json();
    healthBadge.textContent = data.model_loaded ? "Backend + model ready" : "Backend ready — model missing";
  } catch {
    healthBadge.textContent = "Backend offline";
  }
}

function setError(message) {
  errorEl.textContent = message;
  errorEl.classList.toggle("hidden", !message);
}

function percent(value) {
  return `${(value * 100).toFixed(1)}%`;
}

async function analyze() {
  setError("");

  const title = titleEl.value.trim();
  const text = textEl.value.trim();

  if ((title + " " + text).trim().length < 20) {
    setError("Please enter at least 20 meaningful characters.");
    return;
  }

  const button = document.getElementById("predictBtn");
  button.disabled = true;
  button.textContent = "Analyzing…";

  try {
    const response = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({title, text})
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || "Prediction failed.");
    }

    renderPrediction(data);
    await renderExplanation(title, text);
  } catch (error) {
    setError(error.message);
  } finally {
    button.disabled = false;
    button.textContent = "Analyze prediction";
  }
}

function renderPrediction(data) {
  resultEl.classList.remove("hidden");

  const label = document.getElementById("label");
  const confidence = document.getElementById("confidence");
  const predictionClass = data.label.toLowerCase() === "real" ? "real" : "fake";

  label.textContent = data.label;
  label.classList.remove("fake", "real");
  label.classList.add(predictionClass);

  confidence.classList.remove("fake", "real");
  confidence.classList.add(predictionClass);
  confidence.textContent = percent(data.confidence);

  const fake = data.probabilities.FAKE;
  const real = data.probabilities.REAL;

  document.getElementById("fakeProb").textContent = percent(fake);
  document.getElementById("realProb").textContent = percent(real);
  document.getElementById("fakeBar").style.width = percent(fake);
  document.getElementById("realBar").style.width = percent(real);

  document.getElementById("lengthMeta").textContent = `Cleaned characters: ${data.cleaned_text_length}`;
  document.getElementById("modelMeta").textContent = `Model: ${data.model}`;

  rawEl.textContent = JSON.stringify(data, null, 2);
}

async function renderExplanation(title, text) {
  const response = await fetch(`${API_BASE}/explain`, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({title, text, top_k: 8})
  });

  const data = await response.json();
  if (!response.ok) return;

  explanationEl.classList.remove("hidden");

  const list = document.getElementById("wordList");
  list.innerHTML = "";

  for (const item of data.influential_words) {
    const div = document.createElement("div");
    div.className = "word";
    div.textContent = item.word;

    const strong = document.createElement("strong");
    strong.textContent = `${item.impact >= 0 ? "+" : ""}${item.impact.toFixed(4)}`;

    div.appendChild(strong);
    list.appendChild(div);
  }

  rawEl.textContent = JSON.stringify(data, null, 2);
}

document.getElementById("predictBtn").addEventListener("click", analyze);

document.getElementById("clearBtn").addEventListener("click", () => {
  titleEl.value = "";
  textEl.value = "";
  resultEl.classList.add("hidden");
  explanationEl.classList.add("hidden");
  setError("");
  rawEl.textContent = "{}";
});

document.getElementById("exampleBtn").addEventListener("click", () => {
  const example = examples[Math.floor(Math.random() * examples.length)];
  titleEl.value = example.title;
  textEl.value = example.text;
});

document.getElementById("copyBtn").addEventListener("click", async () => {
  await navigator.clipboard.writeText(rawEl.textContent);
});

checkHealth();
