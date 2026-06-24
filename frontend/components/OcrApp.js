/* eslint-disable @next/next/no-img-element */
"use client";

import { useEffect, useRef } from "react";
import {
  abs,
  processFiles,
  recognizeRegions,
  recognizeLines,
  fetchModels,
  setRecognitionModel,
  setDetectionModel,
  reprocessPage,
} from "@/lib/api";

/* Khmer Document OCR — UI shell + imperative pipeline controller.
 *
 * The staged scan/recognition animation is heavily DOM-driven (creating overlay
 * boxes, sweeping scan lines, fitting fonts to measured widths), so the markup
 * is rendered once by React and all behaviour is wired imperatively in a single
 * mount effect (StrictMode double-invoke is disabled in next.config). Containers
 * that the controller fills (overlay, steps, transcript, predocPage, …) are left
 * empty in JSX so React never fights the imperative DOM updates. */
export default function OcrApp() {
  const started = useRef(false);

  useEffect(() => {
    if (started.current) return; // guard against any double-invoke
    started.current = true;
    runController();
  }, []);

  return (
    <>
      <header className="topbar">
        <div className="brand">
          <span className="logo-mark" />
          <div>
            <h1>Khmer Document OCR</h1>
            <p>Layout analysis · Text detection · Recognition</p>
          </div>
        </div>
        <div className="topbar-actions">
          {/* Dropdown: which recognition model the backend pipeline is using.
              Native <details> so it works without React state; the controller
              fills #modelList from GET /api/models on mount. */}
          <details id="modelInfo" className="model-info">
            <summary>
              <span className="model-info-dot" />
              Models
            </summary>
            <div id="modelList" className="model-info-panel">
              <p className="model-info-loading">Loading…</p>
            </div>
          </details>
          <button id="resetBtn" className="ghost-btn" hidden>
            New document
          </button>
        </div>
      </header>

      {/* Upload screen */}
      <main id="uploadView" className="upload-view">
        <div id="dropzone" className="dropzone">
          <div className="dz-inner">
            <div className="dz-icon">
              <svg viewBox="0 0 56 56" width="56" height="56" fill="none" aria-hidden="true">
                <rect x="2" y="2" width="52" height="52" rx="15" stroke="currentColor" strokeWidth="2" />
                <path d="M28 41 V18 M18 28 L28 18 L38 28" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </div>
            <h2>Drop document images here</h2>
            <p>or click to browse — one or many pages (PNG, JPG, WEBP, TIFF)</p>
            <input id="fileInput" type="file" accept="image/*" multiple hidden />
          </div>
        </div>
        <div id="thumbStrip" className="thumb-strip" />
        <button id="runBtn" className="primary-btn" disabled>
          Analyze document
        </button>
        <p className="hint">
          The system runs layout analysis, line detection, and recognition, then
          rebuilds an editable page.
        </p>
      </main>

      {/* Workspace */}
      <main id="workspace" className="workspace" hidden>
        <aside className="pages-rail" id="pagesRail" />

        <section className="stage-wrap">
          <div className="stage-toolbar">
            <div className="view-toggle">
              <button className="seg active" data-view="scan" id="viewScan">
                Analysis
              </button>
              <button className="seg" data-view="predoc" id="viewPredoc" disabled>
                Pre-document
              </button>
            </div>
            <div className="edit-tools">
              <span className="stage-meta" id="stageMeta" />
              <label className="pd-border-ctl" id="scanBorderCtl" hidden>
                <span>Border</span>
                <input id="scanBorder" type="range" min="0" max="6" step="0.5" defaultValue="1.5" />
              </label>
              <label className="pd-border-ctl" id="scanZoomCtl" hidden>
                <span>Size</span>
                <input id="scanZoom" type="range" min="0.25" max="3" step="0.05" defaultValue="1" />
                <span id="scanZoomLabel" className="zoom-label">100%</span>
              </label>
              <div className="view-toggle box-vis" id="boxVis" hidden>
                <button className="seg active" data-vis="all">All</button>
                <button className="seg" data-vis="detect">Detect</button>
                <button className="seg" data-vis="layout">Layout</button>
                <button className="seg" data-vis="none">None</button>
              </div>
              <div className="view-toggle box-vis" id="predocVis" hidden>
                <button className="seg active" data-pvis="all">All</button>
                <button className="seg" data-pvis="boxes">Boxes</button>
                <button className="seg" data-pvis="none">None</button>
              </div>
              <label className="pd-border-ctl" id="predocBorderCtl" hidden>
                <span>Border</span>
                <input id="predocBorder" type="range" min="0" max="6" step="0.5" defaultValue="1.5" />
              </label>
              <label className="pd-border-ctl" id="predocZoomCtl" hidden>
                <span>Size</span>
                <input id="predocZoom" type="range" min="0.25" max="3" step="0.05" defaultValue="1" />
                <span id="zoomLabel" className="zoom-label">100%</span>
              </label>
              <button id="editBtn" className="mini-btn" hidden>
                Edit detection
              </button>
              <div className="view-toggle edit-modes" id="editLayers" hidden>
                <button className="seg active" id="layerLayout">Layout</button>
                <button className="seg" id="layerLines">Lines</button>
              </div>
              <div className="view-toggle edit-modes" id="editModes" hidden>
                <button className="seg active" id="modeSelect">Select</button>
                <button className="seg" id="modeDraw">Add box</button>
              </div>
              <span id="editHint" className="edit-hint" hidden>
                Layout edits block regions · Lines edits text-line boxes — Select to move/resize/delete, Add box to draw
              </span>
              <button id="applyEditBtn" className="primary-btn sm" hidden>
                Apply &amp; re-run
              </button>
              <button id="cancelEditBtn" className="ghost-btn sm" hidden>
                Cancel
              </button>
            </div>
          </div>

          <div className="stage-scroll" id="scanView">
            <div className="stage" id="stage">
              <img id="pageImg" className="page-img" alt="page" />
              <div id="overlay" className="overlay" />
              <div id="scanline" className="scanline" hidden />
            </div>
          </div>

          <div className="stage-scroll" id="predocView" hidden>
            <div className="zoom-box" id="predocZoomBox">
              <div className="doc-page" id="predocPage" />
            </div>
          </div>
        </section>

        <aside className="pipeline">
          <div className="pipeline-steps" id="steps" />
          <div className="stats" id="stats" />
          <div className="transcript-head">
            <h3>Recognized structure</h3>
            <div className="transcript-actions">
              <button className="mini-btn" id="copyBtn">
                Copy
              </button>
              <button className="mini-btn" id="downloadBtn">
                Download
              </button>
            </div>
          </div>
          <div className="transcript" id="transcript" />
        </aside>
      </main>

      <div id="toast" className="toast" hidden />
    </>
  );
}

/* ───────────────────── Imperative controller ───────────────────── */
function runController() {
  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  const escapeHtml = (s) =>
    String(s).replace(/[&<>"']/g, (c) => (
      { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
    ));

  // Populate the header "Models" dropdown with the backend's active models.
  async function loadModelInfo() {
    const list = $("#modelList");
    if (!list) return;
    try {
      const info = await fetchModels();
      const models = info.models || [];
      const activeModel = info.active_model || "";
      const activeDetModel = info.active_det_model || "";

      const recogModels = models.filter((m) => m.stage === "Recognition");
      const detModels = models.filter((m) => m.stage === "Detection");

      const recogOptions = recogModels.map((m) => `
        <option value="${escapeHtml(m.id)}"${m.id === activeModel ? " selected" : ""}>
          ${escapeHtml(m.label || m.name)}
        </option>`).join("");
      const detOptions = detModels.map((m) => `
        <option value="${escapeHtml(m.id)}"${m.id === activeDetModel ? " selected" : ""}>
          ${escapeHtml(m.label || m.name)}
        </option>`).join("");

      const rows = models.map((m) => `
        <div class="model-row${m.active ? " active" : ""}">
          <span class="model-stage">${escapeHtml(m.stage)}</span>
          <span class="model-name" title="${escapeHtml(m.path || "")}">${escapeHtml(m.label || m.name)}</span>
          <span class="model-name subtle">${escapeHtml(m.name)}</span>
          <span class="model-kind">${escapeHtml(m.kind)}</span>
        </div>`).join("");

      const hasPage = state.current >= 0 && state.animated.has(state.current);
      list.innerHTML = `
        <label class="model-picker">
          <span>Recognition model</span>
          <select id="recogModelSelect">${recogOptions}</select>
        </label>
        <label class="model-picker">
          <span>Detection model</span>
          <select id="detModelSelect">${detOptions}</select>
        </label>
        ${rows}
        <div class="model-foot">
          <span>Device: <b>${escapeHtml((info.device || "cpu").toUpperCase())}</b></span>
          <span>${info.warm ? "Models loaded" : "Loads on first run"}</span>
          ${hasPage ? `<button id="rerunPageBtn" class="mini-btn">Re-run page</button>` : ""}
        </div>`;

      const recogSelect = $("#recogModelSelect", list);
      if (recogSelect) {
        recogSelect.addEventListener("change", async () => {
          recogSelect.disabled = true;
          try {
            await setRecognitionModel(recogSelect.value);
            toast("Recognition model changed");
            await loadModelInfo();
          } catch (err) {
            toast("Model switch failed: " + err.message);
            recogSelect.disabled = false;
          }
        });
      }

      const detSelect = $("#detModelSelect", list);
      if (detSelect) {
        detSelect.addEventListener("change", async () => {
          detSelect.disabled = true;
          try {
            await setDetectionModel(detSelect.value);
            toast("Detection model changed");
            await loadModelInfo();
          } catch (err) {
            toast("Model switch failed: " + err.message);
            detSelect.disabled = false;
          }
        });
      }

      const rerunBtn = $("#rerunPageBtn", list);
      if (rerunBtn) {
        rerunBtn.addEventListener("click", async () => {
          const page = state.pages[state.current];
          if (!page) return;
          rerunBtn.disabled = true;
          rerunBtn.textContent = "Re-running…";
          $("#modelInfo").removeAttribute("open");
          try {
            const res = await reprocessPage(page.page_id);
            page.blocks = res.blocks;
            page.width = res.width;
            page.height = res.height;
            state.animated.delete(state.current);
            state.token++;
            $("#viewPredoc").disabled = true;
            await animatePage(page, state.token);
          } catch (err) {
            toast("Re-run failed: " + err.message);
            rerunBtn.disabled = false;
            rerunBtn.textContent = "Re-run page";
          }
        });
      }
    } catch (err) {
      list.innerHTML = `<p class="model-info-loading">Unavailable — is the backend running?</p>`;
    }
  }
  loadModelInfo();

  const TYPE_LABEL = {
    title: "Title", paragraph: "Paragraph", header: "Header",
    caption: "Caption", image: "Image / Logo",
  };
  const STEPS = [
    { key: "layout", title: "Layout analysis", idle: "Detecting document structure" },
    { key: "detect", title: "Text detection", idle: "Locating text lines" },
    { key: "recognize", title: "Recognition", idle: "Reading each line" },
    { key: "build", title: "Build document", idle: "Reconstructing editable page" },
  ];

  const state = {
    files: [], pages: [], current: -1, animated: new Set(), token: 0,
    view: "scan", editing: false, editMode: "select",
    editLayer: "layout", editBoxes: [], editLineBoxes: [],
    boxVis: "all", predocVis: "all", predocBorder: 1.5, predocZoom: 1,
    scanBorder: 1.5, scanZoom: 1, scanBaseW: 0,
  };

  function textLines(block) {
    return (block.lines || []).filter((line) => line.text);
  }

  function shouldRecognizeBlock(block) {
    return block.type !== "image" && block.type !== "header";
  }

  function blockStatusText(block) {
    if (block.type === "image") return "▣ cropped at original coordinates";
    if (block.type === "header") return "▣ cropped from original (not recognized)";
    return "";
  }

  function paragraphTabIndent(block, line, lines = textLines(block)) {
    if (block.type !== "paragraph" || !line.box || !block.box) return 0;
    const savedIndent = Number(line.indent);
    if (Number.isFinite(savedIndent) && savedIndent > 0) return savedIndent;
    if (line.tab === false) return 0;

    const heights = lines
      .map((l) => l.box && Number(l.box.height))
      .filter((h) => Number.isFinite(h) && h > 0)
      .sort((a, b) => a - b);
    if (!heights.length) return 0;

    const threshold = Math.max(4, heights[Math.floor(heights.length / 2)] * 0.5);
    const freeLeft = line.box.x - block.box.x;
    return freeLeft > threshold ? Math.round(freeLeft) : 0;
  }

  function lineText(block, line, lines = textLines(block)) {
    const text = line.text || "";
    if (!text) return "";
    return (paragraphTabIndent(block, line, lines) ? "\t" : "") + text;
  }

  function blockText(block) {
    const lines = textLines(block);
    return lines.map((line) => lineText(block, line, lines)).filter(Boolean).join("\n");
  }

  /* ── Upload screen ── */
  const dropzone = $("#dropzone");
  const fileInput = $("#fileInput");
  const runBtn = $("#runBtn");

  dropzone.addEventListener("click", () => fileInput.click());
  ["dragover", "dragenter"].forEach((ev) =>
    dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.add("drag"); }));
  ["dragleave", "drop"].forEach((ev) =>
    dropzone.addEventListener(ev, () => dropzone.classList.remove("drag")));
  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    addFiles([...e.dataTransfer.files].filter((f) => f.type.startsWith("image/")));
  });
  fileInput.addEventListener("change", () => addFiles([...fileInput.files]));

  function addFiles(files) {
    if (!files.length) return;
    state.files = state.files.concat(files);
    renderThumbs();
    runBtn.disabled = state.files.length === 0;
  }

  function renderThumbs() {
    const strip = $("#thumbStrip");
    strip.innerHTML = "";
    state.files.forEach((f, i) => {
      const el = document.createElement("div");
      el.className = "thumb";
      const img = document.createElement("img");
      img.src = URL.createObjectURL(f);
      el.appendChild(img);
      const n = document.createElement("span");
      n.textContent = i + 1;
      el.appendChild(n);
      strip.appendChild(el);
    });
  }

  runBtn.addEventListener("click", runPipeline);
  $("#resetBtn").addEventListener("click", () => location.reload());

  /* ── Run the pipeline ── */
  async function runPipeline() {
    if (!state.files.length) return;
    runBtn.disabled = true;
    runBtn.textContent = "Analyzing…";
    let data;
    try {
      data = await processFiles(state.files);
    } catch (e) {
      toast("Processing failed: " + e.message);
      runBtn.disabled = false;
      runBtn.textContent = "Analyze document";
      return;
    }
    state.pages = data.pages;
    enterWorkspace();
    selectPage(0);
  }

  function enterWorkspace() {
    $("#uploadView").hidden = true;
    $("#workspace").hidden = false;
    $("#resetBtn").hidden = false;
    buildSteps();
    buildRail();
  }

  /* ── Pages rail ── */
  function buildRail() {
    const rail = $("#pagesRail");
    rail.innerHTML = "";
    state.pages.forEach((p, i) => {
      const el = document.createElement("div");
      el.className = "rail-thumb";
      el.innerHTML =
        `<img src="${abs(p.image_url)}" alt="page ${i + 1}">` +
        `<span class="dot"></span><span class="pno">${i + 1}</span>`;
      el.addEventListener("click", () => selectPage(i));
      rail.appendChild(el);
    });
  }
  function markRail() {
    $$("#pagesRail .rail-thumb").forEach((el, i) => {
      el.classList.toggle("active", i === state.current);
      el.querySelector(".dot").classList.toggle("done", state.animated.has(i));
    });
  }

  /* ── View switching ── */
  $("#viewScan").addEventListener("click", () => setView("scan"));
  $("#viewPredoc").addEventListener("click", () => setView("predoc"));

  function setView(view) {
    // Leaving the Analysis view abandons any in-progress detection edits.
    if (state.editing && view !== "scan") exitEdit(true);
    state.view = view;
    $("#scanView").hidden = view !== "scan";
    $("#predocView").hidden = view !== "predoc";
    $("#viewScan").classList.toggle("active", view === "scan");
    $("#viewPredoc").classList.toggle("active", view === "predoc");
    // Elements can only be measured once their view is visible (a hidden
    // container reports zero size), so fit fonts to their boxes on show.
    if (view === "predoc") { fitPredocBoxes($("#predocPage")); applyPredocZoom(); }
    applyPredocBoxes();
    applyPredocBorder();
    applyScanBorder();
    updateEditBtn();
  }

  function updateEditBtn() {
    const showTools =
      state.view === "scan" && state.animated.has(state.current) && !state.editing;
    $("#editBtn").hidden = !showTools;
    // The All/Detect/Layout/None visibility control lives by the Edit button.
    $("#boxVis").hidden = !showTools;
    // The Analysis border + size sliders apply whenever a page is shown in scan
    // view.
    const showScanTools = state.view === "scan" && state.current >= 0 && !state.editing;
    $("#scanBorderCtl").hidden = !showScanTools;
    $("#scanZoomCtl").hidden = !showScanTools;
    // The predoc box-visibility toggle + border slider only apply to the
    // Pre-document view.
    const showPredocTools =
      state.view === "predoc" && state.animated.has(state.current);
    $("#predocVis").hidden = !showPredocTools;
    $("#predocBorderCtl").hidden = !showPredocTools;
    $("#predocZoomCtl").hidden = !showPredocTools;
  }

  // Predoc box visibility: All (outlines + type tags) / Boxes (outlines only) /
  // None (clean document) — the recognized text and image crops always
  // stay in place.
  function applyPredocBoxes() {
    const vis = state.predocVis;
    const pg = $("#predocPage");
    pg.classList.toggle("hide-boxes", vis === "none");
    pg.classList.toggle("hide-tags", vis === "boxes");
    $$("#predocVis .seg").forEach((s) =>
      s.classList.toggle("active", s.dataset.pvis === vis));
  }
  $$("#predocVis .seg").forEach((s) =>
    s.addEventListener("click", () => {
      state.predocVis = s.dataset.pvis;
      applyPredocBoxes();
    }));

  // Predoc box border thickness: drives a CSS variable on the page so every
  // region outline scales together (0 = no border).
  function applyPredocBorder() {
    $("#predocPage").style.setProperty("--pd-border-w", state.predocBorder + "px");
  }
  $("#predocBorder").addEventListener("input", (e) => {
    state.predocBorder = parseFloat(e.target.value);
    applyPredocBorder();
  });

  // Analysis overlay border thickness: layout boxes use this value directly,
  // while detected line boxes stay slightly lighter so they remain readable.
  function applyScanBorder() {
    const border = state.scanBorder;
    const ov = $("#overlay");
    ov.style.setProperty("--scan-border-w", border + "px");
    ov.style.setProperty("--scan-line-border-w", Math.max(0, border * 0.67) + "px");
  }
  $("#scanBorder").addEventListener("input", (e) => {
    state.scanBorder = parseFloat(e.target.value);
    applyScanBorder();
  });

  // Predoc size slider. The page is scaled with a top-left transform; a sizer
  // box grows with it so the scroll container can reach every corner.
  // offsetWidth (used by fitPredocBoxes) is unaffected by transforms, so
  // font-fitting stays correct at any size.
  function applyPredocZoom() {
    const z = state.predocZoom;
    const pg = $("#predocPage");
    pg.style.transformOrigin = "top left";
    pg.style.transform = `scale(${z})`;
    const page = state.pages[state.current];
    if (page) {
      $("#predocZoomBox").style.width = page.width * z + "px";
      $("#predocZoomBox").style.height = page.height * z + "px";
    }
    $("#zoomLabel").textContent = Math.round(z * 100) + "%";
  }
  $("#predocZoom").addEventListener("input", (e) => {
    state.predocZoom = parseFloat(e.target.value);
    applyPredocZoom();
  });

  // Analysis size slider. The page image (and its percentage-positioned overlay
  // boxes) scale by setting an explicit width relative to the natural fit width
  // captured on load; the scroll container handles anything larger than the view.
  function applyScanZoom() {
    const img = $("#pageImg");
    if (!state.scanBaseW) return;
    img.style.maxWidth = "none";
    img.style.width = state.scanBaseW * state.scanZoom + "px";
    $("#scanZoomLabel").textContent = Math.round(state.scanZoom * 100) + "%";
  }
  // Measure the image's natural fit width (at 100%) before applying any size.
  function captureScanBase() {
    const img = $("#pageImg");
    img.style.width = "";
    img.style.maxWidth = "";
    state.scanBaseW = img.getBoundingClientRect().width;
    applyScanZoom();
  }
  $("#scanZoom").addEventListener("input", (e) => {
    state.scanZoom = parseFloat(e.target.value);
    applyScanZoom();
  });

  // Overlay visibility: All / Detect (lines only) / Layout (blocks only) / None.
  function setBoxVis(vis) {
    state.boxVis = vis;
    const showLayout = vis === "all" || vis === "layout";
    const showLines = vis === "all" || vis === "detect";
    const ov = $("#overlay");
    ov.classList.toggle("hide-bbox", !showLayout);
    ov.classList.toggle("hide-lbox", !showLines);
    $$("#boxVis .seg").forEach((s) =>
      s.classList.toggle("active", s.dataset.vis === vis));
  }
  function applyBoxVis() { setBoxVis(state.boxVis); }
  $$("#boxVis .seg").forEach((s) =>
    s.addEventListener("click", () => setBoxVis(s.dataset.vis)));

  function selectPage(i) {
    if (i === state.current && state.animated.has(i)) return;
    state.current = i;
    state.token++;
    markRail();
    const page = state.pages[i];
    const img = $("#pageImg");
    img.onload = () => {
      captureScanBase();
      if (state.animated.has(i)) renderFinal(page);
      else animatePage(page, state.token);
    };
    img.src = abs(page.image_url);
    $("#stageMeta").textContent = `Page ${i + 1} · ${page.width} × ${page.height} px`;
    setView("scan");
    $("#viewPredoc").disabled = !state.animated.has(i);
  }

  /* ── Steps panel ── */
  function buildSteps() {
    const wrap = $("#steps");
    wrap.innerHTML = "";
    STEPS.forEach((s, n) => {
      const el = document.createElement("div");
      el.className = "step idle";
      el.id = "step-" + s.key;
      el.innerHTML =
        `<div class="ic">${n + 1}</div>` +
        `<div class="body"><h4>${s.title}</h4><p>${s.idle}</p></div>`;
      wrap.appendChild(el);
    });
  }
  function setStep(key, status, sub) {
    const el = $("#step-" + key);
    if (!el) return;
    el.classList.remove("idle", "run", "done");
    el.classList.add(status);
    if (sub) el.querySelector("p").textContent = sub;
  }
  function resetSteps() {
    STEPS.forEach((s) => setStep(s.key, "idle", s.idle));
  }
  function setStats({ blocks, lines, images }) {
    $("#stats").innerHTML =
      `<div class="stat"><b>${blocks}</b><span>Blocks</span></div>` +
      `<div class="stat"><b>${lines}</b><span>Lines</span></div>` +
      `<div class="stat"><b>${images}</b><span>Figures</span></div>`;
  }

  /* ── Overlay helpers (percentage coords) ── */
  const pct = (v, total) => (v / total) * 100 + "%";
  function placeBox(el, box, page) {
    el.style.left = pct(box.x, page.width);
    el.style.top = pct(box.y, page.height);
    el.style.width = pct(box.width, page.width);
    el.style.height = pct(box.height, page.height);
  }
  function clearStage() {
    $("#overlay").innerHTML = "";
    const sl = $("#scanline");
    sl.hidden = true;
    sl.classList.remove("sweep");
  }
  async function fullSweep() {
    const sl = $("#scanline");
    sl.hidden = false;
    sl.classList.remove("sweep");
    void sl.offsetWidth;
    sl.classList.add("sweep");
    await sleep(1250);
    sl.hidden = true;
  }
  async function boxSweep(box, page) {
    const sb = document.createElement("div");
    sb.className = "scan-box";
    placeBox(sb, box, page);
    $("#overlay").appendChild(sb);
    await sleep(480);
    sb.remove();
  }

  /* ── Staged animation ── */
  async function animatePage(page, token) {
    const live = () => token === state.token;
    clearStage();
    resetSteps();
    $("#transcript").innerHTML = "";

    const textBlocks = page.blocks.filter((b) => b.type !== "image");
    const imageBlocks = page.blocks.filter((b) => b.type === "image");
    const totalLines = textBlocks.reduce((n, b) => n + b.lines.length, 0);
    const recognitionBlocks = textBlocks.filter(shouldRecognizeBlock);
    const totalRecognizedLines = recognitionBlocks.reduce((n, b) => n + b.lines.length, 0);
    setStats({ blocks: page.blocks.length, lines: totalLines, images: imageBlocks.length });

    setStep("layout", "run", "Scanning page…");
    await fullSweep();
    if (!live()) return;

    const overlay = $("#overlay");
    const boxEls = new Map();
    let shown = 0;
    for (const block of page.blocks) {
      const el = document.createElement("div");
      el.className = `bbox t-${block.type}`;
      placeBox(el, block.box, page);
      el.innerHTML = `<span class="tag">${TYPE_LABEL[block.type] || block.type}</span>`;
      overlay.appendChild(el);
      boxEls.set(block.id, el);
      requestAnimationFrame(() => el.classList.add("show"));
      shown++;
      setStep("layout", "run", `${shown} / ${page.blocks.length} regions`);
      await sleep(70);
      if (!live()) return;
    }
    setStep("layout", "done", `${page.blocks.length} regions found`);

    setStep("detect", "run", "Detecting lines…");
    const lineEls = [];
    let detected = 0;
    for (const block of textBlocks) {
      const bEl = boxEls.get(block.id);
      bEl && bEl.classList.add("active");
      await boxSweep(block.box, page);
      if (!live()) return;
      for (const line of block.lines) {
        const le = document.createElement("div");
        le.className = "lbox";
        placeBox(le, line.box, page);
        overlay.appendChild(le);
        lineEls.push({ el: le, block, line });
        requestAnimationFrame(() => le.classList.add("show"));
        detected++;
        setStep("detect", "run", `${detected} / ${totalLines} lines`);
        await sleep(38);
        if (!live()) return;
      }
      bEl && bEl.classList.remove("active");
    }
    setStep("detect", "done", `${totalLines} text lines`);

    setStep("recognize", "run",
      totalRecognizedLines ? "Reading lines…" : "No non-header text to read");
    const trWrap = $("#transcript");
    let read = 0;
    for (const block of page.blocks) {
      const tr = document.createElement("div");
      tr.className = "tr-block" +
        (block.type === "image" || block.type === "header" ? " is-image" : "");
      tr.innerHTML =
        `<div class="tr-type">${TYPE_LABEL[block.type] || block.type}</div>` +
        `<div class="tr-text"></div>`;
      trWrap.appendChild(tr);
      requestAnimationFrame(() => tr.classList.add("show"));

      if (!shouldRecognizeBlock(block)) {
        tr.querySelector(".tr-text").textContent = blockStatusText(block);
        continue;
      }
      const target = tr.querySelector(".tr-text");
      const myLines = lineEls.filter((l) => l.block.id === block.id);
      const blockLines = textLines(block);
      for (const { el, line } of myLines) {
        el.classList.add("reading");
        target.textContent += (target.textContent ? "\n" : "") + lineText(block, line, blockLines);
        trWrap.scrollTop = trWrap.scrollHeight;
        read++;
        setStep("recognize", "run", `${read} / ${totalRecognizedLines} lines read`);
        await sleep(55);
        if (!live()) return;
        el.classList.remove("reading");
      }
    }
    setStep("recognize", "done", `${totalRecognizedLines} lines recognized`);

    setStep("build", "run", "Laying out page…");
    buildPredoc(page);
    await sleep(250);
    if (!live()) return;
    setStep("build", "done", "Editable document ready");

    state.animated.add(state.current);
    markRail();
    $("#viewPredoc").disabled = false;
    updateEditBtn();
    applyBoxVis();
    toast("Document ready — open the Pre-document tab to edit");
  }

  /* Re-render an already-animated page (no animation). */
  function renderFinal(page) {
    clearStage();
    resetSteps();
    STEPS.forEach((s) => setStep(s.key, "done"));
    $("#transcript").innerHTML = "";

    const overlay = $("#overlay");
    for (const block of page.blocks) {
      const el = document.createElement("div");
      el.className = `bbox t-${block.type} show`;
      placeBox(el, block.box, page);
      el.innerHTML = `<span class="tag">${TYPE_LABEL[block.type] || block.type}</span>`;
      overlay.appendChild(el);
      for (const line of block.lines || []) {
        const le = document.createElement("div");
        le.className = "lbox show";
        placeBox(le, line.box, page);
        overlay.appendChild(le);
      }
    }
    const textBlocks = page.blocks.filter((b) => b.type !== "image");
    const totalLines = textBlocks.reduce((n, b) => n + b.lines.length, 0);
    setStats({ blocks: page.blocks.length, lines: totalLines,
      images: page.blocks.length - textBlocks.length });

    const trWrap = $("#transcript");
    for (const block of page.blocks) {
      const tr = document.createElement("div");
      tr.className = "tr-block show" +
        (block.type === "image" || block.type === "header" ? " is-image" : "");
      const txt = shouldRecognizeBlock(block) ? blockText(block) : blockStatusText(block);
      tr.innerHTML =
        `<div class="tr-type">${TYPE_LABEL[block.type] || block.type}</div>` +
        `<div class="tr-text"></div>`;
      tr.querySelector(".tr-text").textContent = txt;
      trWrap.appendChild(tr);
    }
    buildPredoc(page);
    $("#viewPredoc").disabled = false;
    updateEditBtn();
    applyBoxVis();
  }

  /* ── Detection editor (Analysis view) ──
   * Lets you correct the layout by hand: move/resize boxes, draw new ones,
   * delete wrong ones, and change a block's type. Applying sends the edited
   * regions back to the server, which re-detects + re-recognises each one and
   * returns fresh blocks that rebuild every view. */
  const EDIT_TYPES = [
    ["paragraph", "Paragraph"], ["title", "Title"], ["header", "Header"],
    ["caption", "Caption"], ["image", "Image / Logo"],
  ];
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

  // Map a pointer event to page-pixel coordinates inside the overlay.
  function ptToPage(e, page) {
    const r = $("#overlay").getBoundingClientRect();
    return {
      x: clamp(((e.clientX - r.left) / r.width) * page.width, 0, page.width),
      y: clamp(((e.clientY - r.top) / r.height) * page.height, 0, page.height),
    };
  }

  $("#editBtn").addEventListener("click", enterEdit);
  $("#applyEditBtn").addEventListener("click", applyEdits);
  $("#cancelEditBtn").addEventListener("click", () => {
    exitEdit(false);
    renderFinal(state.pages[state.current]);
  });
  $("#modeSelect").addEventListener("click", () => setMode("select"));
  $("#modeDraw").addEventListener("click", () => setMode("draw"));
  $("#layerLayout").addEventListener("click", () => setLayer("layout"));
  $("#layerLines").addEventListener("click", () => setLayer("lines"));
  // Drawing a new box: pointer-down on the overlay while the Add-box tool is on.
  // (In draw mode the boxes are click-through, so the overlay gets the event.)
  $("#overlay").addEventListener("pointerdown", (e) => {
    if (state.editing && state.editMode === "draw") startDraw(e);
  });

  // The array currently being edited: layout block boxes, or text-line boxes.
  function items() {
    return state.editLayer === "lines" ? state.editLineBoxes : state.editBoxes;
  }

  // Switch between the Select tool (move/resize/delete) and the Add-box tool.
  function setMode(mode) {
    state.editMode = mode;
    $("#modeSelect").classList.toggle("active", mode === "select");
    $("#modeDraw").classList.toggle("active", mode === "draw");
    $("#overlay").classList.toggle("draw", mode === "draw");
  }

  // Switch which layer is being edited: the layout block boxes or the text lines.
  function setLayer(layer) {
    state.editLayer = layer;
    $("#layerLayout").classList.toggle("active", layer === "layout");
    $("#layerLines").classList.toggle("active", layer === "lines");
    setMode("select");
    renderEditBoxes(state.pages[state.current]);
  }

  function enterEdit() {
    const page = state.pages[state.current];
    if (!page) return;
    state.editing = true;
    // Snapshot both layers up front so toggling between them keeps your edits.
    state.editBoxes = page.blocks.map((b) => ({
      type: b.type === "image" ? "image" : b.type,
      x: b.box.x, y: b.box.y, w: b.box.width, h: b.box.height,
    }));
    state.editLineBoxes = [];
    page.blocks.forEach((b) => {
      if (b.type === "image") return;
      (b.lines || []).forEach((l) => state.editLineBoxes.push({
        x: l.box.x, y: l.box.y, w: l.box.width, h: l.box.height,
      }));
    });
    state.editLayer = "layout";
    $("#scanline").hidden = true;
    $("#overlay").classList.add("editing");
    $("#editBtn").hidden = true;
    $("#editLayers").hidden = false;
    $("#editModes").hidden = false;
    $("#applyEditBtn").hidden = false;
    $("#cancelEditBtn").hidden = false;
    $("#editHint").hidden = false;
    $("#layerLayout").classList.add("active");
    $("#layerLines").classList.remove("active");
    setMode("select");
    renderEditBoxes(page);
  }

  // Tear down editor UI. caller re-renders the page (apply/cancel both do).
  function exitEdit() {
    state.editing = false;
    $("#overlay").classList.remove("editing", "draw");
    $("#editLayers").hidden = true;
    $("#editModes").hidden = true;
    $("#applyEditBtn").hidden = true;
    $("#cancelEditBtn").hidden = true;
    $("#editHint").hidden = true;
    updateEditBtn();
  }

  function renderEditBoxes(page) {
    const overlay = $("#overlay");
    overlay.innerHTML = "";
    const lines = state.editLayer === "lines";

    // In Lines mode, draw the block regions faintly as non-interactive context.
    if (lines) {
      state.editBoxes.forEach((b) => {
        const ctx = document.createElement("div");
        ctx.className = `ebox-ctx t-${b.type}`;
        placeBox(ctx, { x: b.x, y: b.y, width: b.w, height: b.h }, page);
        overlay.appendChild(ctx);
      });
    }

    items().forEach((b, i) => {
      const el = document.createElement("div");
      el.className = lines ? "ebox eline" : `ebox t-${b.type}`;
      el.dataset.i = i;
      placeBox(el, { x: b.x, y: b.y, width: b.w, height: b.h }, page);

      // Layout boxes carry a type selector; line boxes inherit their block type.
      if (!lines) {
        const tag = document.createElement("span");
        tag.className = "ebox-tag";
        const sel = document.createElement("select");
        EDIT_TYPES.forEach(([v, l]) => {
          const o = document.createElement("option");
          o.value = v; o.textContent = l; o.selected = v === b.type;
          sel.appendChild(o);
        });
        sel.addEventListener("pointerdown", (e) => e.stopPropagation());
        sel.addEventListener("change", () => {
          state.editBoxes[i].type = sel.value;
          el.className = `ebox t-${sel.value}`;
        });
        tag.appendChild(sel);
        el.appendChild(tag);
      }

      const del = document.createElement("button");
      del.className = "ebox-del";
      del.textContent = "✕";
      del.title = "Delete box";
      del.addEventListener("pointerdown", (e) => e.stopPropagation());
      del.addEventListener("click", (e) => {
        e.stopPropagation();
        items().splice(i, 1);
        renderEditBoxes(page);
      });
      el.appendChild(del);

      ["nw", "ne", "sw", "se"].forEach((h) => {
        const hd = document.createElement("span");
        hd.className = "ehandle " + h;
        hd.addEventListener("pointerdown", (e) => startGesture(e, i, h));
        el.appendChild(hd);
      });

      el.addEventListener("pointerdown", (e) => {
        if (e.target === el) startGesture(e, i, "move"); // box body only
      });
      overlay.appendChild(el);
    });
  }

  // Run a drag gesture, capturing the pointer on the overlay so every move/up
  // routes to us even over other boxes or outside the page (and the scroll
  // container can't steal the gesture). onMove(dx, dy) gets page-pixel deltas.
  function runDrag(e, onMove, onUp) {
    e.preventDefault();
    e.stopPropagation();
    const overlay = $("#overlay");
    const page = state.pages[state.current];
    const start = ptToPage(e, page);
    const pid = e.pointerId;
    try { overlay.setPointerCapture(pid); } catch (_) { /* ignore */ }
    function move(ev) {
      const p = ptToPage(ev, page);
      onMove(p.x - start.x, p.y - start.y, p);
    }
    function up(ev) {
      overlay.removeEventListener("pointermove", move);
      overlay.removeEventListener("pointerup", up);
      overlay.removeEventListener("pointercancel", up);
      try { overlay.releasePointerCapture(pid); } catch (_) { /* ignore */ }
      onUp && onUp(ev);
    }
    overlay.addEventListener("pointermove", move);
    overlay.addEventListener("pointerup", up);
    overlay.addEventListener("pointercancel", up);
  }

  function placeEditBox(i) {
    const b = items()[i];
    const el = $("#overlay").querySelector(`.ebox[data-i="${i}"]`);
    if (el && b) placeBox(el, { x: b.x, y: b.y, width: b.w, height: b.h },
      state.pages[state.current]);
  }

  // Move or resize the i-th box of the active layer (mode: "move" | corner).
  function startGesture(e, i, mode) {
    const page = state.pages[state.current];
    const arr = items();
    const o = { ...arr[i] };
    runDrag(e, (dx, dy) => {
      const b = { ...o };
      if (mode === "move") {
        b.x = clamp(o.x + dx, 0, Math.max(0, page.width - o.w));
        b.y = clamp(o.y + dy, 0, Math.max(0, page.height - o.h));
      } else {
        if (mode.includes("e")) b.w = o.w + dx;
        if (mode.includes("s")) b.h = o.h + dy;
        if (mode.includes("w")) { b.x = o.x + dx; b.w = o.w - dx; }
        if (mode.includes("n")) { b.y = o.y + dy; b.h = o.h - dy; }
        b.w = Math.max(8, b.w); b.h = Math.max(8, b.h);
        b.x = clamp(b.x, 0, page.width - 8);
        b.y = clamp(b.y, 0, page.height - 8);
        b.w = Math.min(b.w, page.width - b.x);
        b.h = Math.min(b.h, page.height - b.y);
      }
      arr[i] = b;
      placeEditBox(i);
    });
  }

  // Draw a brand-new box from an empty-space pointer-down (into the active layer).
  function startDraw(e) {
    const page = state.pages[state.current];
    const arr = items();
    const a = ptToPage(e, page);
    const i = arr.push({ type: "paragraph", x: a.x, y: a.y, w: 0, h: 0 }) - 1;
    renderEditBoxes(page);
    runDrag(e, (dx, dy, p) => {
      arr[i] = {
        type: "paragraph",
        x: Math.min(a.x, p.x), y: Math.min(a.y, p.y),
        w: Math.abs(p.x - a.x), h: Math.abs(p.y - a.y),
      };
      placeEditBox(i);
    }, () => {
      const b = arr[i];
      if (!b || b.w < 10 || b.h < 10) arr.splice(i, 1); // ignore stray clicks
      setMode("select"); // drop back to Select so the new box is editable at once
      renderEditBoxes(page);
    });
  }

  function applyEdits() {
    return state.editLayer === "lines" ? applyLineEdits() : applyLayoutEdits();
  }

  async function runReRun(label, fn) {
    $("#applyEditBtn").disabled = true;
    toast(label);
    let res;
    try {
      res = await fn();
    } catch (err) {
      toast("Re-run failed: " + err.message);
      $("#applyEditBtn").disabled = false;
      return;
    }
    $("#applyEditBtn").disabled = false;
    const page = state.pages[state.current];
    page.blocks = res.blocks;
    page.width = res.width;
    page.height = res.height;
    exitEdit();
    renderFinal(page);
    toast("Detection updated");
  }

  // Layout layer: send block regions; the server re-detects + recognises lines.
  async function applyLayoutEdits() {
    const page = state.pages[state.current];
    if (!page) return;
    const regions = state.editBoxes
      .filter((b) => b.w >= 6 && b.h >= 6)
      .map((b) => ({
        type: b.type,
        box: {
          x: Math.round(b.x), y: Math.round(b.y),
          width: Math.round(b.w), height: Math.round(b.h),
        },
      }));
    if (!regions.length) { toast("Add at least one box first"); return; }
    await runReRun("Re-running detection…", () =>
      recognizeRegions(page.page_id, regions));
  }

  // Lines layer: keep the block boxes, assign each edited line box to the block
  // it falls in (nearest if it falls in none), and recognise those exact lines.
  async function applyLineEdits() {
    const page = state.pages[state.current];
    if (!page) return;
    const blockBoxes = state.editBoxes; // the (possibly edited) block layer
    const center = (b) => ({ cx: b.x + b.w / 2, cy: b.y + b.h / 2 });
    const inBox = (cx, cy, bb) =>
      cx >= bb.x && cx <= bb.x + bb.w && cy >= bb.y && cy <= bb.y + bb.h;

    const payload = blockBoxes.map((b) => ({
      type: b.type,
      box: { x: Math.round(b.x), y: Math.round(b.y),
             width: Math.round(b.w), height: Math.round(b.h) },
      lines: [],
    }));
    const textIdx = blockBoxes
      .map((b, i) => (b.type === "image" ? -1 : i))
      .filter((i) => i >= 0);

    for (const ln of state.editLineBoxes) {
      if (ln.w < 6 || ln.h < 6) continue;
      const { cx, cy } = center(ln);
      let bi = textIdx.find((i) => inBox(cx, cy, blockBoxes[i]));
      if (bi === undefined) {
        // fall back to the nearest text block by centre distance
        let best = Infinity;
        for (const i of textIdx) {
          const c = center(blockBoxes[i]);
          const d = (c.cx - cx) ** 2 + (c.cy - cy) ** 2;
          if (d < best) { best = d; bi = i; }
        }
      }
      if (bi === undefined) continue;
      payload[bi].lines.push({
        x: Math.round(ln.x), y: Math.round(ln.y),
        width: Math.round(ln.w), height: Math.round(ln.h),
      });
    }
    await runReRun("Re-reading lines…", () =>
      recognizeLines(page.page_id, payload));
  }

  /* ── Pre-document: LAYOUT regions + recognized text on the detected lines ──
   * A blank white page sized 1:1 to the input. For every layout block a labelled
   * outline box is drawn at its exact detected coordinate (so you see the
   * detection), and each recognized line is placed on the SAME line as it was
   * detected (at the line's own x — not snapped left or right). Picture blocks
   * show the crop. Font is uniform per block and fits the page width. */
  function buildPredoc(page) {
    const pg = $("#predocPage");
    pg.style.width = page.width + "px";
    pg.style.height = page.height + "px";
    pg.innerHTML = "";
    for (const block of page.blocks) {
      // Layout region outline (with type label).
      const box = document.createElement("div");
      box.className = `pd-box t-${block.type}`;
      box.style.left = block.box.x + "px";
      box.style.top = block.box.y + "px";
      box.style.width = block.box.width + "px";
      box.style.height = block.box.height + "px";
      if ((block.type === "image" || block.type === "header") && block.crop) {
        const im = document.createElement("img");
        im.className = "pd-crop";
        im.src = abs(block.crop.url);
        box.appendChild(im);
      }
      pg.appendChild(box);

      // Type label sits on the page (above the box) so the box's overflow:hidden
      // — which clips the text to the box — does not also hide the label.
      const tag = document.createElement("span");
      tag.className = "pd-tag";
      tag.textContent = TYPE_LABEL[block.type] || block.type;
      tag.style.left = block.box.x + "px";
      tag.style.top = block.box.y + "px";
      pg.appendChild(tag);
      if (block.type === "image" || block.type === "header") continue;

      // Recognized text lives INSIDE the layout box (which is overflow:hidden),
      // so the text area stays within the box. Lines are spread evenly across the
      // box height (top->bottom) and positioned relative to the box.
      const lines = textLines(block);
      if (!lines.length) continue;
      const slot = block.box.height / lines.length;   // equal vertical share per line
      const baseFont = Math.max(7, slot * 0.62);
      // Paragraph tab rule: render the same leading tab character as the
      // recognized preview, and preserve it with white-space: pre.
      lines.forEach((line, i) => {
        const el = document.createElement("div");
        el.className = `doc-text t-${block.type}`;
        el.contentEditable = "true";
        el.spellcheck = false;
        el.dataset.block = block.id;
        el.dataset.boxw = Math.max(10, block.box.width);
        el.textContent = lineText(block, line, lines);
        el.style.left = "0px";
        el.style.paddingLeft = "0.45em";     // small inset so text isn't on the edge
        el.style.top = (i * slot) + "px";    // fill box top -> bottom
        el.style.height = slot + "px";
        el.style.lineHeight = slot + "px";
        el.style.whiteSpace = "pre";
        el.style.tabSize = "4";
        el.style.textAlign = "left";
        el.style.fontSize = baseFont.toFixed(1) + "px";
        box.appendChild(el);               // child of the layout box -> clipped to it
      });
    }
    fitPredocBoxes(pg);
    applyPredocZoom();
  }

  // Shrink each block uniformly so its widest line fits exactly inside the layout
  // box width — the recognized text occupies the box, never spilling past it.
  async function fitPredocBoxes(pg) {
    if (document.fonts && document.fonts.ready) {
      try { await document.fonts.ready; } catch (e) { /* ignore */ }
    }
    const groups = new Map();
    for (const el of pg.querySelectorAll(".doc-text")) {
      const id = el.dataset.block || "_";
      if (!groups.has(id)) groups.set(id, []);
      groups.get(id).push(el);
    }
    for (const els of groups.values()) {
      let ratio = 1;
      let measurable = true;
      for (const el of els) {
        const w = el.offsetWidth;
        if (!w) { measurable = false; break; } // hidden -> refit on show
        const avail = (parseFloat(el.dataset.boxw) || 0) - 4; // box width, small right inset
        if (avail > 0 && w > avail) ratio = Math.min(ratio, avail / w);
      }
      if (!measurable || ratio >= 1) continue;
      for (const el of els) {
        const fs = parseFloat(el.style.fontSize);
        el.style.fontSize = Math.max(6, fs * ratio * 0.99).toFixed(2) + "px";
      }
    }
  }

  /* ── Copy / download ── */
  function pageText(page) {
    return page.blocks
      .filter(shouldRecognizeBlock)
      .map((b) => blockText(b))
      .filter(Boolean)
      .join("\n\n");
  }
  $("#copyBtn").addEventListener("click", async () => {
    const page = state.pages[state.current];
    if (!page) return;
    await navigator.clipboard.writeText(pageText(page));
    toast("Text copied");
  });
  $("#downloadBtn").addEventListener("click", () => {
    const page = state.pages[state.current];
    if (!page) return;
    const blob = new Blob([pageText(page)], { type: "text/plain;charset=utf-8" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `page_${state.current + 1}.txt`;
    a.click();
  });

  /* ── Toast ── */
  let toastTimer;
  function toast(msg) {
    const t = $("#toast");
    t.textContent = msg;
    t.hidden = false;
    requestAnimationFrame(() => t.classList.add("show"));
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => {
      t.classList.remove("show");
      setTimeout(() => (t.hidden = true), 250);
    }, 2600);
  }
}
