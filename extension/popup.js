const SERVER_URL = "http://127.0.0.1:7331";

const statusEl = document.getElementById("status");
const errorEl = document.getElementById("error");
const resultsEl = document.getElementById("results");
const gradeEl = document.getElementById("grade");
const scoreEl = document.getElementById("score");
const labelEl = document.getElementById("label");
const dimsEl = document.getElementById("dimensions");
const wordsEl = document.getElementById("words");
const btn = document.getElementById("scoreBtn");

function showError(msg) {
  errorEl.textContent = msg;
  errorEl.style.display = "block";
  statusEl.textContent = "";
}

function scoreColor(score) {
  if (score >= 0.7) return "#3fb950";
  if (score >= 0.5) return "#d29922";
  return "#f85149";
}

function renderResults(data) {
  resultsEl.style.display = "block";
  errorEl.style.display = "none";
  statusEl.textContent = "";

  gradeEl.textContent = data.grade;
  gradeEl.className = "grade grade-" + data.grade;
  scoreEl.textContent = data.overall_score.toFixed(3);
  labelEl.textContent = data.label;
  wordsEl.textContent = data.word_count.toLocaleString() + " words";

  dimsEl.innerHTML = "";
  for (const [name, dim] of Object.entries(data.dimensions)) {
    const score = dim.score;
    const pct = Math.round(score * 100);
    const color = scoreColor(score);
    dimsEl.innerHTML += `
      <div class="dim">
        <span class="dim-name">${name}</span>
        <span class="dim-score" style="color:${color}">${score.toFixed(3)}</span>
      </div>
      <div class="bar"><div class="bar-fill" style="width:${pct}%;background:${color}"></div></div>
    `;
  }
}

btn.addEventListener("click", async () => {
  btn.disabled = true;
  statusEl.textContent = "Extracting page content...";
  errorEl.style.display = "none";
  resultsEl.style.display = "none";

  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (!tab || !tab.id) {
      showError("No active tab found.");
      btn.disabled = false;
      return;
    }

    const results = await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: () => document.documentElement.outerHTML,
    });

    const html = results[0].result;
    if (!html) {
      showError("Could not extract page HTML.");
      btn.disabled = false;
      return;
    }

    statusEl.textContent = "Scoring content...";

    const resp = await fetch(SERVER_URL + "/score", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ html: html, url: tab.url }),
    });

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      showError(err.error || "Server error: " + resp.status);
      btn.disabled = false;
      return;
    }

    const data = await resp.json();
    renderResults(data);
  } catch (e) {
    if (e.message && e.message.includes("Failed to fetch")) {
      showError("Cannot connect to distill server. Run: distill serve");
    } else {
      showError("Error: " + e.message);
    }
  }

  btn.disabled = false;
});
