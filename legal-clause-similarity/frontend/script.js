async function searchClauses() {
    const queryElement = document.getElementById("query");
    const statusElement = document.getElementById("status");
    const resultsElement = document.getElementById("results");
    const headingCountElement = document.getElementById("headingCount");
    const activeFilterElement = document.getElementById("activeFilter");
    const searchButton = document.getElementById("searchButton");
    const query = queryElement.value.trim();
    const topK = Number(document.getElementById("topK").value);
    const clauseType = document.getElementById("clauseType").value;

    if (!query) {
        statusElement.textContent = "Add a clause to begin searching.";
        queryElement.focus();
        return;
    }

    searchButton.disabled = true;
    searchButton.querySelector("span").textContent = "Searching index...";
    statusElement.textContent = "Comparing semantic neighbors...";
    resultsElement.innerHTML = "<div class=\"empty-state\"><span class=\"empty-line\"></span><p>Searching 21,187 indexed clauses...</p></div>";

    try {
        const response = await fetch("/search", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query, top_k: topK, clause_type: clauseType })
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "Search failed.");
        statusElement.textContent = `Search complete · ${data.results.length} nearest matches`;
        headingCountElement.textContent = topK;
        activeFilterElement.textContent = clauseType === "all" ? "All clause types" : clauseType;
        resultsElement.innerHTML = data.results.map((result, index) => `
            <article class="result-card" style="animation-delay: ${index * 55}ms">
                <div class="result-header"><span class="rank">0${result.rank} / MATCH</span><span class="similarity">${result.similarity_percentage}% <small>similarity</small></span></div>
                <span class="clause-type">${escapeHtml(result.clause_type)}</span>
                <p class="clause-text">${escapeHtml(result.clause)}</p>
                <div class="result-meta"><span>Clause ${escapeHtml(result.clause_id)}</span><span>Cosine score ${result.similarity}</span></div>
            </article>
        `).join("");
    } catch (error) {
        console.error(error);
        statusElement.textContent = `Unable to search · ${error.message}`;
        resultsElement.innerHTML = "<div class=\"empty-state\"><span class=\"empty-line\"></span><p>The index is unavailable. Check that the API is running.</p></div>";
    } finally {
        searchButton.disabled = false;
        searchButton.querySelector("span").textContent = "Search clauses";
    }
}

function escapeHtml(value) {
    const div = document.createElement("div");
    div.textContent = value;
    return div.innerHTML;
}

const examples = {
    confidentiality: "Each party shall keep the other party's Confidential Information strictly confidential and shall use it only to perform this Agreement.",
    termination: "Either party may terminate this Agreement if the other party materially breaches its obligations. The breaching party shall have thirty days to cure the breach after receiving written notice.",
    indemnity: "Each party shall indemnify and hold harmless the other party from claims arising out of its breach of this Agreement. This obligation includes reasonable legal fees and court costs. The indemnified party shall promptly notify the indemnifying party of any claim."
};

const queryInput = document.getElementById("query");
const charCount = document.getElementById("charCount");
function updateCharCount() {
    const count = queryInput.value.length;
    charCount.textContent = `${count} character${count === 1 ? "" : "s"}`;
}
queryInput.addEventListener("input", updateCharCount);
queryInput.addEventListener("keydown", (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter") searchClauses();
});

document.querySelectorAll(".suggestion-card").forEach((card) => {
    card.addEventListener("click", () => {
        document.querySelectorAll(".suggestion-card").forEach((item) => item.classList.remove("is-selected"));
        card.classList.add("is-selected");
        queryInput.value = examples[card.dataset.example];
        updateCharCount();
        searchClauses();
    });
});

document.getElementById("topK").addEventListener("change", () => {
    document.getElementById("headingCount").textContent = document.getElementById("topK").value;
});

updateCharCount();
