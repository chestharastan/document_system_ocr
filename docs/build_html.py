"""Render docs/ThaOCR_Book.md as a browsable page, docs/ThaOCR_Book.html (figures, tables, Mermaid diagrams, a chapter sidebar).

    python3 docs/build_html.py

The Markdown is embedded in the page and rendered in the browser with marked + mermaid (jsDelivr), so the page works
when opened straight from disk. Without an internet connection it falls back to showing the raw Markdown.
Re-run after editing the book.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
md = (HERE / "ThaOCR_Book.md").read_text(encoding="utf-8")

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ThaOCR: the book</title>
<style>
:root{--ink:#1f2328;--mut:#59636e;--line:#d1d9e0;--soft:#f6f8fa;--acc:#0969da}
*{box-sizing:border-box}
body{margin:0;color:var(--ink);background:#fff;font:16px/1.65 "Noto Sans","Noto Sans Khmer","Khmer OS","Khmer OS Siemreap",system-ui,sans-serif}
#side{position:fixed;top:0;left:0;bottom:0;width:290px;overflow:auto;padding:22px 16px 40px 22px;border-right:1px solid var(--line);background:#fbfcfd;font-size:13.5px;line-height:1.45}
#side h4{margin:0 0 12px;font-size:15px}
#side a{display:block;color:var(--ink);text-decoration:none;padding:3px 6px;border-radius:6px}
#side a.part{margin-top:12px;font-weight:600;color:var(--mut);text-transform:uppercase;font-size:11.5px;letter-spacing:.04em}
#side a.ch{padding-left:10px}
#side a:hover,#side a.on{background:#eaeef2}
main{margin-left:290px;padding:34px 48px 120px;max-width:1240px}
main h1{font-size:2em;border-bottom:1px solid var(--line);padding-bottom:.25em;margin:1.6em 0 .6em}
main h1:first-child{margin-top:0}
main h2{font-size:1.5em;border-bottom:1px solid var(--line);padding-bottom:.2em;margin:1.8em 0 .6em}
main h3{font-size:1.2em;margin:1.5em 0 .5em}
main h4{font-size:1.05em;margin:1.3em 0 .4em}
p,li{max-width:62em}
a{color:var(--acc)}
table{border-collapse:collapse;margin:14px 0 18px;font-size:14px;display:block;overflow-x:auto;max-width:100%}
th,td{border:1px solid var(--line);padding:5px 10px;vertical-align:top}
th{background:var(--soft);font-weight:600}
tr:nth-child(2n) td{background:#fcfdfe}
code{background:#eff1f3;border-radius:5px;padding:.1em .35em;font:85% ui-monospace,"JetBrains Mono","Noto Sans Mono",monospace}
pre{background:var(--soft);border:1px solid var(--line);border-radius:8px;padding:14px;overflow:auto;font-size:13px;line-height:1.45}
pre code{background:none;padding:0}
img{max-width:100%;border:1px solid var(--line);border-radius:8px;margin:8px 0}
blockquote{border-left:4px solid var(--line);margin:0;padding:0 1em;color:var(--mut)}
hr{border:0;border-top:1px solid var(--line);margin:2.2em 0}
.mermaid{margin:16px 0;text-align:center}
#raw{white-space:pre-wrap}
@media (max-width:900px){#side{display:none}main{margin-left:0;padding:20px}}
</style>
<script src="https://cdn.jsdelivr.net/npm/marked@12/marked.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"></script>
</head><body>
<nav id="side"><h4>ThaOCR: the book</h4><div id="toc"></div></nav>
<main id="c"></main>
<script>
const MD = __MD__;
const main = document.getElementById("c");
const slug = t => t.trim().toLowerCase().replace(/[^\\w\\- ]/g, "").replace(/ /g, "-");
if (!window.marked) {
  main.innerHTML = "<p><b>Offline:</b> the Markdown renderer could not load, so the raw text is shown.</p><pre id='raw'></pre>";
  document.getElementById("raw").textContent = MD;
} else {
  main.innerHTML = marked.parse(MD);
  const toc = document.getElementById("toc"), links = [];
  main.querySelectorAll("h1,h2,h3,h4").forEach(h => {
    h.id = slug(h.textContent);
    if (h.tagName === "H1" || h.tagName === "H2") {
      const a = document.createElement("a");
      a.href = "#" + h.id; a.textContent = h.textContent;
      a.className = h.tagName === "H1" ? "part" : "ch";
      toc.appendChild(a); links.push([h, a]);
    }
  });
  main.querySelectorAll("pre code.language-mermaid").forEach(c => {
    const d = document.createElement("div"); d.className = "mermaid"; d.textContent = c.textContent; c.parentElement.replaceWith(d);
  });
  if (window.mermaid) { mermaid.initialize({startOnLoad: false, theme: "neutral"}); mermaid.run(); }
  const mark = () => { let cur = null; for (const [h, a] of links) { if (h.getBoundingClientRect().top < 120) cur = a; }
    links.forEach(([, a]) => a.classList.toggle("on", a === cur)); };
  addEventListener("scroll", mark, {passive: true}); mark();
  if (location.hash) { const e = document.getElementById(decodeURIComponent(location.hash.slice(1))); if (e) setTimeout(() => e.scrollIntoView(), 50); }
}
</script></body></html>
"""

out = HERE / "ThaOCR_Book.html"
out.write_text(PAGE.replace("__MD__", json.dumps(md)), encoding="utf-8")
print("wrote", out, f"{out.stat().st_size / 1e3:.0f} kB")
