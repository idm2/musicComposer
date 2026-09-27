"""songsheet.pdf — the page a guitarist plays from: chord diagrams across the top, then the lyrics with chords above
the words, section by section. Built as HTML from the same chord sheet as chords.txt, printed to PDF by a headless
Chromium browser (Chrome or Edge), so the two can never disagree."""
import html
import re
import shutil
import subprocess
from pathlib import Path

from ..models import Transcription
from . import guitar
from .chordsheet import render_chordsheet

BROWSERS = ("chrome", "msedge", "chromium", "google-chrome",
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
CHORD_TOKEN = re.compile(r"[A-G][#b]?(m|maj7|m7|7|sus[24]|dim|aug|6|9|add9|m7b5)?(/[A-G][#b]?)?\??")


def diagram_svg(name: str, frets: list[str]) -> str:
    """A chord box: six strings, five frets, dots for fingers, x/o above the nut, "7fr" when the shape sits high."""
    fretted = [int(f) for f in frets if f not in ("x", "0")]
    base = 1 if not fretted or max(fretted) <= guitar.CHART_FRETS else min(fretted)
    x0, y0, sx, sy = 14, 30, 12, 14
    parts = [f'<text x="44" y="12" class="nm">{html.escape(name)}</text>']
    for i in range(6):
        parts.append(f'<line x1="{x0 + i * sx}" y1="{y0}" x2="{x0 + i * sx}" y2="{y0 + 5 * sy}"/>')
    for k in range(6):
        w = 3 if k == 0 and base == 1 else 1
        parts.append(f'<line x1="{x0}" y1="{y0 + k * sy}" x2="{x0 + 5 * sx}" y2="{y0 + k * sy}" stroke-width="{w}"/>')
    for i, f in enumerate(frets):
        x = x0 + i * sx
        if f == "x":
            parts.append(f'<text x="{x}" y="{y0 - 5}" class="mk">×</text>')
        elif f == "0":
            parts.append(f'<circle cx="{x}" cy="{y0 - 8}" r="3.2" class="open"/>')
        else:
            parts.append(f'<circle cx="{x}" cy="{y0 + (int(f) - base + 0.5) * sy}" r="4.4"/>')
    if base > 1:
        parts.append(f'<text x="{x0 + 5 * sx + 5}" y="{y0 + sy * 0.8}" class="fr">{base}fr</text>')
    return f'<svg viewBox="0 0 88 104" width="88" height="104">{"".join(parts)}</svg>'


def _is_chord_row(line: str) -> bool:
    tokens = line.split()
    return bool(tokens) and all(CHORD_TOKEN.fullmatch(t) for t in tokens)


def build_html(title: str, t: Transcription) -> str:
    sheet = render_chordsheet(title, t).splitlines()
    meta = next((l for l in sheet if l.startswith("Key:")), "")
    meta = re.sub(r" \(confidence [0-9.]+\)", "", meta)
    meta = " · ".join(p.strip() for p in re.split(r"\s{2,}", meta) if p.strip())
    note = next((l for l in sheet if l.startswith("Transcribed from")), "")
    body_start = next((i for i, l in enumerate(sheet) if l.startswith("[")), len(sheet))
    diagrams, seen = [], set()
    for c in sorted(t.analysis.chords, key=lambda c: c.onset):
        frets = guitar.voicing(c)
        if c.symbol not in seen and frets:
            seen.add(c.symbol)
            diagrams.append(diagram_svg(c.symbol, frets))
    blocks, current = [], []
    for line in sheet[body_start:]:
        if line.startswith("[") and line.endswith("]"):
            if current:
                blocks.append(current)
            current = [f'<h3>{html.escape(line[1:-1])}</h3>']
        elif _is_chord_row(line):
            current.append(f'<div class="ch">{html.escape(line)}</div>')
        elif line.strip():
            current.append(f'<div class="ly">{html.escape(line)}</div>')
    if current:
        blocks.append(current)
    sections = "".join(f'<section>{"".join(b)}</section>' for b in blocks)
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(title)}</title><style>
@page {{ size: A4; margin: 14mm 14mm 16mm; }}
body {{ font-family: "Segoe UI", Arial, sans-serif; color: #111; }}
h1 {{ font-size: 24pt; margin: 0 0 2pt; }}
.meta {{ font-size: 10.5pt; color: #444; margin-bottom: 8pt; }}
.charts {{ display: flex; flex-wrap: wrap; gap: 4pt 10pt; padding: 6pt 0 8pt; border-top: 1px solid #ccc;
          border-bottom: 1px solid #ccc; margin-bottom: 10pt; }}
svg line {{ stroke: #222; }} svg circle {{ fill: #111; }} svg circle.open {{ fill: none; stroke: #222; stroke-width: 1.2; }}
svg .nm {{ font: bold 12px "Segoe UI", Arial; text-anchor: middle; }} svg .mk {{ font: 11px Arial; text-anchor: middle; }}
svg .fr {{ font: 9px Arial; }}
section {{ break-inside: avoid; margin-bottom: 9pt; }}
h3 {{ font-size: 10.5pt; margin: 0 0 2pt; color: #7a3b00; text-transform: uppercase; letter-spacing: .05em; }}
.ch, .ly {{ font-family: Consolas, "Courier New", monospace; font-size: 11pt; white-space: pre; line-height: 1.25; }}
.ch {{ font-weight: bold; color: #b04a00; }}
.note {{ font-size: 8.5pt; color: #666; margin-top: 10pt; }}
</style></head><body>
<h1>{html.escape(title)}</h1>
<div class="meta">{html.escape(meta)} · standard tuning, no capo</div>
<div class="charts">{"".join(diagrams)}</div>
{sections}
<div class="note">{html.escape(note)} Chords marked ? are less certain — check them by ear.</div>
</body></html>
"""


def find_browser() -> str | None:
    for b in BROWSERS:
        found = shutil.which(b) or (b if Path(b).exists() else None)
        if found:
            return found
    return None


def write_songsheet(title: str, t: Transcription, html_path: Path, pdf_path: Path) -> bool:
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(build_html(title, t), encoding="utf-8")
    browser = find_browser()
    if browser is None:
        print(f"! no Chrome/Edge found — wrote {html_path} (open it and print to PDF) but no songsheet PDF.")
        return False
    proc = subprocess.run([browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                           f"--print-to-pdf={pdf_path.resolve()}", html_path.resolve().as_uri()],
                          capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
    if proc.returncode != 0 or not pdf_path.exists():
        print(f"! browser could not print {html_path}:\n{proc.stderr[-600:]}")
        return False
    return True
