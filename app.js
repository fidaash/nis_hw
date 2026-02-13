// app.js
import { pipeline } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.6/dist/transformers.min.js";

/**
 * Fully static, client-side sentiment analyzer + logger:
 * - Loads reviews_test.tsv via fetch (same folder as index.html)
 * - Parses TSV with Papa Parse (global Papa from index.html)
 * - Runs sentiment classification fully in-browser using Transformers.js
 * - Maps sentiment → business action (OFFER_COUPON / UPSELL / NO_OFFER)
 * - Optional: logs each analysis to Google Sheets via an Apps Script Web App endpoint
 *
 * Logging columns (exact):
 * 1) ts_iso
 * 2) review
 * 3) sentiment (label + confidence)
 * 4) meta (JSON containing client info + model details)
 * 5) action_taken (NEW: e.g. "OFFER_COUPON")
 */

const MODEL_ID = "Xenova/distilbert-base-uncased-finetuned-sst-2-english";
const TSV_PATH = "reviews_test.tsv";

// ---- Minimal shared state ----
let reviews = [];
let sentimentPipeline = null;

const sessionCounts = { POSITIVE: 0, NEGATIVE: 0, NEUTRAL: 0 };

const STORAGE_KEYS = {
  userId: "rsa_user_id",
  logEnabled: "rsa_log_enabled",
  logEndpoint: "rsa_log_endpoint",
};

const $ = (id) => document.getElementById(id);

// ---- Business Logic: Sentiment → Action ----
/**
 * Maps a sentiment bucket to a business action.
 * NEGATIVE → OFFER_COUPON  (customer upset, offer discount to retain)
 * POSITIVE → UPSELL        (customer happy, suggest premium/related product)
 * NEUTRAL  → NO_OFFER      (no strong signal, do nothing)
 */
function getBusinessAction(bucket) {
  if (bucket === "NEGATIVE") return "OFFER_COUPON";
  if (bucket === "POSITIVE") return "UPSELL";
  return "NO_OFFER";
}

/**
 * Returns a dynamic UI message based on the business action.
 */
function getActionMessage(action) {
  if (action === "OFFER_COUPON") {
    return {
      emoji: "🎟️",
      title: "We're sorry to hear that!",
      body: "As an apology, here's a 10% discount coupon for your next order: SORRY10",
      cls: "action-coupon",
    };
  }
  if (action === "UPSELL") {
    return {
      emoji: "⭐",
      title: "Glad you loved it!",
      body: "You might also enjoy our Premium plan — get even more features you'll love.",
      cls: "action-upsell",
    };
  }
  return {
    emoji: "💬",
    title: "Thanks for your feedback!",
    body: "We appreciate you taking the time to share your experience.",
    cls: "action-neutral",
  };
}

// ---- UI helpers ----
function setBusy(isBusy, message = "Working…") {
  const busy = $("busy");
  const btn = $("analyzeBtn");
  busy.style.display = isBusy ? "inline-flex" : "none";
  btn.disabled = isBusy || !sentimentPipeline || reviews.length === 0;

  if (isBusy) {
    busy.innerHTML = `<i class="fa-solid fa-spinner"></i> ${escapeHtml(message)}`;
  }
}

function setStatus(text, kind = "info") {
  const dot = $("statusDot");
  const statusText = $("statusText");
  statusText.textContent = text;

  dot.classList.remove("ready", "warn", "err");
  if (kind === "ready") dot.classList.add("ready");
  if (kind === "warn") dot.classList.add("warn");
  if (kind === "err") dot.classList.add("err");
}

function clearError() {
  const box = $("errorBox");
  box.textContent = "";
  box.classList.remove("show");
}

function showError(userMessage, err = null) {
  const box = $("errorBox");
  box.textContent = userMessage;
  box.classList.add("show");
  if (err) console.error(err);
}

function setReviewText(text) {
  $("reviewText").textContent = text || "";
}

function escapeHtml(str) {
  return String(str ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

// ---- Action message UI ----
function setActionUI(action) {
  const box = $("actionBox");
  if (!box) return;

  const msg = getActionMessage(action);
  box.className = `action-box ${msg.cls}`;
  box.innerHTML = `
    <div class="action-emoji">${msg.emoji}</div>
    <div class="action-text">
      <div class="action-title">${escapeHtml(msg.title)}</div>
      <div class="action-body">${escapeHtml(msg.body)}</div>
      <div class="action-tag">Action: <code>${escapeHtml(action)}</code></div>
    </div>
  `;
  box.style.display = "flex";
}

function clearActionUI() {
  const box = $("actionBox");
  if (!box) return;
  box.style.display = "none";
  box.innerHTML = "";
}

// ---- Reviews loading ----
async function loadReviews() {
  setStatus("Loading reviews TSV…", "info");

  try {
    const res = await fetch(TSV_PATH, { cache: "no-store" });
    if (!res.ok) throw new Error(`Failed to fetch ${TSV_PATH}: ${res.status} ${res.statusText}`);
    const tsvText = await res.text();

    const parsed = Papa.parse(tsvText, {
      header: true,
      delimiter: "\t",
      skipEmptyLines: true,
    });

    if (parsed.errors && parsed.errors.length > 0) {
      const first = parsed.errors[0];
      throw new Error(`TSV parse error: ${first.message || "Unknown"} (row ${first.row ?? "?"})`);
    }

    const rows = Array.isArray(parsed.data) ? parsed.data : [];
    const texts = rows
      .map((r) => (r && typeof r.text === "string" ? r.text.trim() : ""))
      .filter((t) => t.length > 0);

    if (texts.length === 0) {
      throw new Error("No valid review texts found. Ensure TSV has a 'text' column.");
    }

    reviews = texts;
    setStatus(`Reviews loaded (${reviews.length})`, "info");
  } catch (err) {
    reviews = [];
    setStatus("Reviews failed to load", "err");
    showError(
      `Could not load or parse ${TSV_PATH}. Make sure it exists next to index.html and contains a 'text' column.`,
      err
    );
  }
}

// ---- Model init ----
async function initModel() {
  setStatus("Loading sentiment model… (first load can take a while)", "info");
  try {
    sentimentPipeline = await pipeline("text-classification", MODEL_ID, {
      progress_callback: (p) => {
        if (!p) return;
        const msg =
          typeof p === "string"
            ? p
            : (p?.status ? `${p.status}${p?.file ? `: ${p.file}` : ""}` : "Loading model…");
        setStatus(msg, "info");
      },
    });

    setStatus("Sentiment model ready", "ready");
  } catch (err) {
    sentimentPipeline = null;
    setStatus("Model failed to load", "err");
    showError("Could not load the sentiment model in the browser. Check console for details.", err);
  }
}

// ---- Sentiment analysis ----
function pickRandomReview() {
  if (!reviews.length) return null;
  return reviews[Math.floor(Math.random() * reviews.length)] || null;
}

function normalizePipelineOutput(output) {
  if (Array.isArray(output) && output.length > 0 && output[0] && typeof output[0] === "object") {
    return output;
  }
  if (Array.isArray(output) && output.length > 0 && Array.isArray(output[0]) && output[0][0]) {
    return output[0];
  }
  throw new Error("Unexpected pipeline output format.");
}

function mapToBucket(topLabel, topScore) {
  const label = String(topLabel || "").toUpperCase();
  const score = Number(topScore);
  if (label === "POSITIVE" && score > 0.5) return "POSITIVE";
  if (label === "NEGATIVE" && score > 0.5) return "NEGATIVE";
  return "NEUTRAL";
}

function bucketToUI(bucket) {
  if (bucket === "POSITIVE") return { icon: "fa-thumbs-up", cls: "accentPos" };
  if (bucket === "NEGATIVE") return { icon: "fa-thumbs-down", cls: "accentNeg" };
  return { icon: "fa-question-circle", cls: "accentNeu" };
}

function setResultUI({ bucket, modelLabel, score, ms }) {
  const badge = $("resultBadge");
  const iconWrap = $("resultIcon");
  const labelEl = $("resultLabel");
  const metaEl = $("resultMeta");

  const pct = Math.max(0, Math.min(1, Number(score) || 0)) * 100;
  $("confidencePct").textContent = `${pct.toFixed(1)}%`;

  updateDonut(pct / 100, bucket);

  const { icon, cls } = bucketToUI(bucket);
  badge.classList.remove("accentPos", "accentNeg", "accentNeu");
  badge.classList.add(cls);

  iconWrap.innerHTML = `<i class="fa-solid ${icon}"></i>`;
  labelEl.textContent = `${bucket} (${pct.toFixed(1)}% confidence)`;

  metaEl.textContent = `Model: ${MODEL_ID} • Raw label: ${String(modelLabel)} • ${ms} ms`;
}

function updateDonut(progress01, bucket) {
  const arc = $("donutArc");
  const r = 48;
  const circumference = 2 * Math.PI * r;
  const clamped = Math.max(0, Math.min(1, progress01 || 0));
  const dash = clamped * circumference;
  const gap = circumference - dash;
  arc.setAttribute("stroke-dasharray", `${dash.toFixed(2)} ${gap.toFixed(2)}`);

  if (bucket === "POSITIVE") arc.setAttribute("stroke", "rgba(34,197,94,0.85)");
  else if (bucket === "NEGATIVE") arc.setAttribute("stroke", "rgba(239,68,68,0.85)");
  else arc.setAttribute("stroke", "rgba(163,163,163,0.75)");
}

async function analyzeRandomReview() {
  clearError();
  clearActionUI();

  if (!sentimentPipeline) {
    showError("Sentiment model is not ready yet. Please wait for it to finish loading.");
    return;
  }
  if (!reviews.length) {
    showError("No reviews are loaded. Please check reviews_test.tsv and reload the page.");
    return;
  }

  const review = pickRandomReview();
  if (!review) {
    showError("Could not pick a review. Please check the TSV content.");
    return;
  }

  setReviewText(review);
  setBusy(true, "Analyzing…");

  const t0 = performance.now();
  try {
    const raw = await sentimentPipeline(review);
    const normalized = normalizePipelineOutput(raw);

    const top = normalized
      .slice()
      .sort((a, b) => (Number(b?.score) || 0) - (Number(a?.score) || 0))[0];

    if (!top || typeof top.label !== "string" || typeof top.score !== "number") {
      throw new Error("Invalid top classification result.");
    }

    const bucket = mapToBucket(top.label, top.score);
    const ms = Math.round(performance.now() - t0);

    // ---- Business Logic ----
    const action_taken = getBusinessAction(bucket);

    setResultUI({ bucket, modelLabel: top.label, score: top.score, ms });
    setActionUI(action_taken);

    sessionCounts[bucket] += 1;
    drawDistributionChart();
    updateChartFooter();

    const sentimentStr = `${bucket} (${(top.score * 100).toFixed(1)}%)`;

    // ✅ REQUIRED LOGGING COLUMNS (now includes action_taken):
    await maybeLogToSheet({
      ts_iso: new Date().toISOString(),
      review,
      sentiment: sentimentStr,
      meta: buildMeta({
        bucket,
        modelLabel: top.label,
        score: top.score,
        ms,
      }),
      action_taken,
    });
  } catch (err) {
    showError("Analysis failed. Please try again (and check the console for details).", err);
  } finally {
    setBusy(false);
  }
}

// ---- Session chart (canvas) ----
function updateChartFooter() {
  $("chartFoot").textContent =
    `POSITIVE: ${sessionCounts.POSITIVE} • NEGATIVE: ${sessionCounts.NEGATIVE} • NEUTRAL: ${sessionCounts.NEUTRAL}`;
}

function drawDistributionChart() {
  const canvas = $("distChart");
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  const cssWidth = canvas.clientWidth || 900;
  const cssHeight = canvas.clientHeight || 280;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.floor(cssWidth * dpr);
  canvas.height = Math.floor(cssHeight * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  const w = cssWidth;
  const h = cssHeight;

  ctx.clearRect(0, 0, w, h);

  const labels = ["POSITIVE", "NEUTRAL", "NEGATIVE"];
  const values = [sessionCounts.POSITIVE, sessionCounts.NEUTRAL, sessionCounts.NEGATIVE];
  const maxV = Math.max(1, ...values);

  const padding = { left: 28, right: 18, top: 16, bottom: 34 };
  const innerW = w - padding.left - padding.right;
  const innerH = h - padding.top - padding.bottom;

  ctx.strokeStyle = "rgba(255,255,255,0.10)";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = padding.top + (innerH * i) / 4;
    ctx.beginPath();
    ctx.moveTo(padding.left, y);
    ctx.lineTo(w - padding.right, y);
    ctx.stroke();
  }

  const barGap = 22;
  const barW = (innerW - barGap * (labels.length - 1)) / labels.length;

  labels.forEach((lab, i) => {
    const v = values[i];
    const x = padding.left + i * (barW + barGap);
    const barH = (v / maxV) * (innerH - 6);
    const y = padding.top + innerH - barH;

    const grad = ctx.createLinearGradient(0, y, 0, y + barH);
    if (lab === "POSITIVE") {
      grad.addColorStop(0, "rgba(34,197,94,0.75)");
      grad.addColorStop(1, "rgba(34,197,94,0.18)");
    } else if (lab === "NEGATIVE") {
      grad.addColorStop(0, "rgba(239,68,68,0.75)");
      grad.addColorStop(1, "rgba(239,68,68,0.18)");
    } else {
      grad.addColorStop(0, "rgba(163,163,163,0.65)");
      grad.addColorStop(1, "rgba(163,163,163,0.14)");
    }

    roundRect(ctx, x, y, barW, barH, 12);
    ctx.fillStyle = grad;
    ctx.fill();

    ctx.strokeStyle = "rgba(255,255,255,0.10)";
    ctx.lineWidth = 1;
    roundRect(ctx, x, y, barW, barH, 12);
    ctx.stroke();

    ctx.fillStyle = "rgba(255,255,255,0.85)";
    ctx.font = "700 13px ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial";
    ctx.textAlign = "center";
    ctx.textBaseline = "bottom";
    ctx.fillText(String(v), x + barW / 2, y - 6);

    ctx.fillStyle = "rgba(255,255,255,0.65)";
    ctx.font = "650 12px ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial";
    ctx.textBaseline = "top";
    ctx.fillText(lab, x + barW / 2, padding.top + innerH + 10);
  });
}

function roundRect(ctx, x, y, w, h, r) {
  const radius = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + radius, y);
  ctx.arcTo(x + w, y, x + w, y + h, radius);
  ctx.arcTo(x + w, y + h, x, y + h, radius);
  ctx.arcTo(x, y + h, x, y, radius);
  ctx.arcTo(x, y, x + w, y, radius);
  ctx.closePath();
}

// ---- Logging (required columns: ts_iso, review, sentiment, meta, action_taken) ----
function getOrCreateUserId() {
  const existing = localStorage.getItem(STORAGE_KEYS.userId);
  if (existing) return existing;

  const id = `u_${Math.random().toString(16).slice(2)}_${Date.now().toString(16)}`;
  localStorage.setItem(STORAGE_KEYS.userId, id);
  return id;
}

function buildMeta(extra = {}) {
  return {
    userId: getOrCreateUserId(),
    page: location.href,
    referrer: document.referrer || "",
    userAgent: navigator.userAgent,
    language: navigator.language || "",
    timeZone: Intl.DateTimeFormat().resolvedOptions().timeZone || "",
    screen: { w: window.screen.width, h: window.screen.height, dpr: window.devicePixelRatio || 1 },
    model: MODEL_ID,
    app: { name: "review-sentiment-explorer", version: "2.0.0" },
    ...extra,
  };
}

function isLoggingEnabled() {
  return localStorage.getItem(STORAGE_KEYS.logEnabled) === "1";
}

function getLogEndpoint() {
  return (localStorage.getItem(STORAGE_KEYS.logEndpoint) || "").trim();
}

async function maybeLogToSheet({ ts_iso, review, sentiment, meta, action_taken }) {
  if (!isLoggingEnabled()) return;
  const endpoint = getLogEndpoint();
  if (!endpoint) return;

  // action_taken is now included as the 5th column
  const payload = { ts_iso, review, sentiment, meta, action_taken };
  const body = JSON.stringify(payload);

  try {
    if (navigator.sendBeacon) {
      const ok = navigator.sendBeacon(
        endpoint,
        new Blob([body], { type: "text/plain;charset=utf-8" })
      );
      if (ok) return;
    }

    await fetch(endpoint, {
      method: "POST",
      mode: "no-cors",
      body,
      cache: "no-store",
      keepalive: true,
    });
  } catch (err) {
    console.warn("Google Sheet logging error:", err);
  }
}

// ---- Logging UI ----
function syncLoggingUIFromStorage() {
  const sw = $("logSwitch");
  const endpoint = $("logEndpoint");

  const enabled = isLoggingEnabled();
  sw.classList.toggle("on", enabled);
  sw.setAttribute("aria-checked", enabled ? "true" : "false");

  endpoint.value = getLogEndpoint();
}

function toggleLogging() {
  const enabled = !isLoggingEnabled();
  localStorage.setItem(STORAGE_KEYS.logEnabled, enabled ? "1" : "0");
  syncLoggingUIFromStorage();
}

function attachLoggingHandlers() {
  const sw = $("logSwitch");
  const endpoint = $("logEndpoint");

  sw.addEventListener("click", toggleLogging);
  sw.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      toggleLogging();
    }
  });

  endpoint.addEventListener("change", () => {
    localStorage.setItem(STORAGE_KEYS.logEndpoint, endpoint.value.trim());
  });
}

// ---- Reset ----
function resetSession() {
  sessionCounts.POSITIVE = 0;
  sessionCounts.NEGATIVE = 0;
  sessionCounts.NEUTRAL = 0;

  $("confidencePct").textContent = "—";
  updateDonut(0, "NEUTRAL");

  const badge = $("resultBadge");
  badge.classList.remove("accentPos", "accentNeg", "accentNeu");
  badge.classList.add("accentNeu");
  $("resultIcon").innerHTML = `<i class="fa-regular fa-circle-question"></i>`;
  $("resultLabel").textContent = "No result yet";
  $("resultMeta").textContent = "Click the button to run sentiment analysis.";

  setReviewText('Review will appear here after you click "Analyze random review".');
  clearActionUI();
  drawDistributionChart();
  updateChartFooter();
  clearError();
}

function updateAnalyzeButtonState() {
  $("analyzeBtn").disabled = !(sentimentPipeline && reviews.length > 0);
}

// ---- Bootstrap ----
async function bootstrap() {
  attachLoggingHandlers();
  syncLoggingUIFromStorage();

  updateDonut(0, "NEUTRAL");
  drawDistributionChart();
  updateChartFooter();

  $("analyzeBtn").addEventListener("click", analyzeRandomReview);
  $("resetBtn").addEventListener("click", resetSession);

  await loadReviews();
  await initModel();
  updateAnalyzeButtonState();

  if (reviews.length === 0 || !sentimentPipeline) {
    setStatus("Ready with issues (see error message)", "warn");
  }
}

document.addEventListener("DOMContentLoaded", bootstrap);
window.addEventListener("resize", drawDistributionChart);
