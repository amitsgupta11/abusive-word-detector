/**
 * =====================================================
 *  GuardNLP — Frontend Logic
 *  Handles: API calls, result rendering, animations
 * =====================================================
 */

// ── DOM References ──
const textInput      = document.getElementById("textInput");
const charCount      = document.getElementById("charCount");
const analyzeBtn     = document.getElementById("analyzeBtn");
const resultPanel    = document.getElementById("resultPanel");
const verdictWrap    = document.getElementById("verdictWrap");
const verdictIcon    = document.getElementById("verdictIcon");
const verdictLabel   = document.getElementById("verdictLabel");
const safeBar        = document.getElementById("safeBar");
const abusiveBar     = document.getElementById("abusiveBar");
const safeScore      = document.getElementById("safeScore");
const abusiveScore   = document.getElementById("abusiveScore");
const confidenceVal  = document.getElementById("confidenceVal");
const processedText  = document.getElementById("processedText");
const errorMsg       = document.getElementById("errorMsg");

// ── Character Counter ──
textInput.addEventListener("input", () => {
  charCount.textContent = textInput.value.length;
});

// ── Submit on Enter (Ctrl/Cmd + Enter) ──
textInput.addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
    analyzeText();
  }
});

/**
 * analyzeText()
 * Sends input to the Flask /predict endpoint and renders the result.
 */
async function analyzeText() {
  const text = textInput.value.trim();

  // ── Input Validation ──
  if (!text) {
    showError("Please enter some text before analyzing.");
    return;
  }

  hideError();
  setLoading(true);
  hideResult();

  try {
    // ── API Call ──
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text })
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || "Server error. Please try again.");
    }

    // ── Render Result ──
    renderResult(data);

  } catch (err) {
    showError(err.message || "Could not connect to the server. Is Flask running?");
  } finally {
    setLoading(false);
  }
}

/**
 * renderResult(data)
 * Displays prediction result, confidence bars, and meta info.
 */
function renderResult(data) {
  const isAbusive = data.label === "Abusive";

  // ── Verdict Badge ──
  verdictIcon.textContent  = isAbusive ? "⚠️" : "✅";
  verdictLabel.textContent = data.label;
  verdictWrap.className    = "verdict-wrap " + (isAbusive ? "abusive-result" : "safe-result");

  // ── Confidence Bars (animated via CSS width transition) ──
  // Reset first for re-animation effect
  safeBar.style.width     = "0%";
  abusiveBar.style.width  = "0%";
  safeScore.textContent   = "—";
  abusiveScore.textContent= "—";

  // Trigger reflow to restart animation
  void safeBar.offsetWidth;

  requestAnimationFrame(() => {
    safeBar.style.width     = data.safe_prob + "%";
    abusiveBar.style.width  = data.abusive_prob + "%";
    safeScore.textContent   = data.safe_prob.toFixed(1) + "%";
    abusiveScore.textContent= data.abusive_prob.toFixed(1) + "%";
  });

  // ── Meta Info ──
  confidenceVal.textContent  = data.confidence + "%";
  processedText.textContent  = `"${data.processed_text}"`;

  // ── Show Panel ──
  showResult();
}

/**
 * clearAll()
 * Resets all inputs and result panels.
 */
function clearAll() {
  textInput.value           = "";
  charCount.textContent     = "0";
  hideResult();
  hideError();
  textInput.focus();
}

// ── Helpers ──

function setLoading(isLoading) {
  if (isLoading) {
    analyzeBtn.classList.add("loading");
    // Inject spinner HTML dynamically
    if (!analyzeBtn.querySelector(".spinner")) {
      const spinner = document.createElement("div");
      spinner.className = "spinner";
      analyzeBtn.prepend(spinner);
    }
    analyzeBtn.querySelector(".spinner").style.display = "block";
  } else {
    analyzeBtn.classList.remove("loading");
    const spinner = analyzeBtn.querySelector(".spinner");
    if (spinner) spinner.style.display = "none";
  }
}

function showResult() {
  resultPanel.classList.add("visible");
}

function hideResult() {
  resultPanel.classList.remove("visible");
}

function showError(msg) {
  errorMsg.textContent = "⚠ " + msg;
  errorMsg.classList.add("visible");
}

function hideError() {
  errorMsg.classList.remove("visible");
  errorMsg.textContent = "";
}
