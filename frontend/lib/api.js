// API helpers for talking to the FastAPI backend.

export const API_BASE =
  (process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000").replace(/\/$/, "");

// Page-image and crop URLs from the API are relative (/storage/...); prefix them
// with the backend origin so the Next.js app (different port) can load them.
export function abs(url) {
  if (!url) return url;
  return url.startsWith("/") ? `${API_BASE}${url}` : url;
}

// Which models the backend pipeline is using -> { device, warm, models: [...] }.
export async function fetchModels() {
  const res = await fetch(`${API_BASE}/models`);
  if (!res.ok) throw new Error(`Server error ${res.status}`);
  return res.json();
}

// Switch the active recognition model. The next OCR run will load this model.
export async function setRecognitionModel(modelId) {
  const res = await fetch(`${API_BASE}/models`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model_id: modelId }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Server error ${res.status}`);
  }
  return res.json();
}

// Switch the active detection model. The next OCR run will load this model.
export async function setDetectionModel(modelId) {
  const res = await fetch(`${API_BASE}/det-models`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model_id: modelId }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Server error ${res.status}`);
  }
  return res.json();
}

// Re-run the full pipeline on a previously uploaded page (uses the stored image).
export async function reprocessPage(pageId) {
  const res = await fetch(`${API_BASE}/reprocess`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ page_id: pageId }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Server error ${res.status}`);
  }
  return res.json();
}

// Upload one or many images and get back the staged per-page OCR structure.
export async function processFiles(files) {
  const form = new FormData();
  files.forEach((f) => form.append("files", f));
  const res = await fetch(`${API_BASE}/process`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Server error ${res.status}`);
  }
  return res.json();
}

// Re-run detection + recognition for manually edited regions of one page.
// regions: [{ type, box: { x, y, width, height } }] -> { width, height, blocks }.
export async function recognizeRegions(pageId, regions) {
  const res = await fetch(`${API_BASE}/regions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ page_id: pageId, regions }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Server error ${res.status}`);
  }
  return res.json();
}

// Re-recognise a page from hand-edited text-line boxes (the line editor).
// blocks: [{ type, box, lines: [{ x, y, width, height }] }] -> { width, height, blocks }.
export async function recognizeLines(pageId, blocks) {
  const res = await fetch(`${API_BASE}/lines`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ page_id: pageId, blocks }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Server error ${res.status}`);
  }
  return res.json();
}

// Switch the active model of any pipeline stage ("layout" | "textdet" | "rec").
export async function setStageModel(stage, modelId) {
  const res = await fetch(`${API_BASE}/pipeline/models`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ stage, model_id: modelId }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Server error ${res.status}`);
  }
  return res.json();
}
