"""Write the landscape as a ranked table and a crowding-versus-momentum scatter.

    python landscape_report.py     # outputs/landscape.md and outputs/landscape.html

Plain Markdown and one self-contained HTML file with an inline SVG: no libraries to load.
"""
import html
import json

import config

SIGNAL_COLORS = {"GO": "#1a7f37", "WAIT": "#9a6700", "SKIP": "#8c959f"}


def last_change(field):
    for h in reversed(field.get("history") or []):
        if h.get("event") == "changed":
            what = ", ".join(f"{k} {h[k][0]} -> {h[k][1]}" for k in ("quadrant", "signal") if k in h)
            return f"{h['date']}: {what}"
    return ""


def fmt(x, pct=False):
    if x is None:
        return "–"
    return f"{x:.0%}" if pct else f"{x:.1f}"


def ranked(doc):
    scored = [f for f in doc["fields"] if f.get("scores") and f.get("status") == "active"]
    return sorted(scored, key=lambda f: -f["scores"]["entry_score"])


def markdown(doc):
    lines = ["# Research landscape", "",
             f"Counts source: {doc['meta'].get('counts_source', 'pubmed_connector')}. "
             "Entry score = 0.40 opportunity + 0.35 feasibility + 0.25 fit. "
             "Warrant gap = share of extracted papers without external validation.", "",
             "| # | Field | Signal | Regime | Entry | Momentum | Crowding | Warrant gap | Last change |",
             "|---|---|---|---|---|---|---|---|---|"]
    for i, f in enumerate(ranked(doc), start=1):
        q = f.get("quality") or {}
        lines.append(f"| {i} | {f['label']} | {f.get('signal', '')} | {f['quadrant']} | "
                     f"{fmt(f['scores']['entry_score'])} | {fmt(f['metrics']['momentum'])} | "
                     f"{fmt(f['metrics']['crowding'])} | {fmt(q.get('warrant_gap'), pct=True)} | "
                     f"{last_change(f)} |")
    provisional = [f for f in doc["fields"] if f.get("status") == "provisional"]
    if provisional:
        lines += ["", "## Provisional (not ranked)", ""]
        lines += [f"- **{f['id']}**: {f.get('note', '')}" for f in provisional]
    return "\n".join(lines) + "\n"


def scatter(doc, width=760, height=520, pad=56):
    """Crowding (x) against momentum (y), both percentiles, with the quadrant lines at 50."""
    def x(v):
        return pad + v / 100 * (width - 2 * pad)

    def y(v):
        return height - pad - v / 100 * (height - 2 * pad)

    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Crowding versus momentum">',
             f'<line x1="{x(50)}" y1="{y(0)}" x2="{x(50)}" y2="{y(100)}" class="mid"/>',
             f'<line x1="{x(0)}" y1="{y(50)}" x2="{x(100)}" y2="{y(50)}" class="mid"/>',
             f'<rect x="{x(0)}" y="{y(100)}" width="{x(100) - x(0)}" height="{y(0) - y(100)}" class="frame"/>',
             f'<text x="{x(2)}" y="{y(96)}" class="q">Blooming</text>',
             f'<text x="{x(98)}" y="{y(96)}" class="q" text-anchor="end">Hot and contested</text>',
             f'<text x="{x(2)}" y="{y(3)}" class="q">Quiet / whitespace</text>',
             f'<text x="{x(98)}" y="{y(3)}" class="q" text-anchor="end">Crowded, decelerating</text>',
             f'<text x="{width / 2}" y="{height - 14}" class="axis" text-anchor="middle">'
             'Crowding (percentile of 2025 AI papers) →</text>',
             f'<text x="16" y="{height / 2}" class="axis" text-anchor="middle" '
             f'transform="rotate(-90 16 {height / 2})">Momentum (percentile of growth) →</text>']
    fields = sorted(ranked(doc), key=lambda f: -f["metrics"]["momentum"])
    # Boxes (left, text baseline, width) labels must avoid: every dot, then each label drawn.
    placed = [(x(f["metrics"]["crowding"]) - 9, y(f["metrics"]["momentum"]) + 4, 18) for f in fields]
    for f in fields:
        m = f["metrics"]
        color = SIGNAL_COLORS.get(f.get("signal"), "#57606a")
        r = 9 if f.get("signal") == "GO" else 6
        cx, cy = x(m["crowding"]), y(m["momentum"])
        w = 6.2 * len(f["label"])
        left = m["crowding"] > 60          # right-hand labels go left of the dot
        lx = cx - r - 3 - w if left else cx + r + 3
        ly = cy + 4
        own = (cx - 9, cy + 4, 18)
        while any(abs(ly - py) < 13 and lx < px + pw and px < lx + w
                  for px, py, pw in placed if (px, py, pw) != own):
            ly += 13
        placed.append((lx, ly, w))
        record = {k: f.get(k) for k in ("id", "label", "signal", "quadrant", "scores", "metrics",
                                         "quality", "mvp_example", "open_data")}
        leader = (f'<line x1="{cx:.1f}" y1="{cy:.1f}" x2="{(lx + w if left else lx):.1f}" '
                  f'y2="{ly - 4:.1f}" class="lead"/>' if ly != cy + 4 else "")
        parts.append(
            f'<g class="pt" data-field="{html.escape(json.dumps(record))}">{leader}'
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" fill="{color}"/>'
            f'<text x="{lx:.1f}" y="{ly:.1f}">{html.escape(f["label"])}</text></g>')
    parts.append("</svg>")
    return "\n".join(parts)


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Research landscape</title>
<style>
body {{ font: 14px/1.45 system-ui, sans-serif; margin: 24px; color: #1f2328; background: #fff; }}
svg {{ max-width: 100%; height: auto; }}
.mid {{ stroke: #d0d7de; stroke-dasharray: 4 4; }}
.frame {{ fill: none; stroke: #d0d7de; }}
.q {{ fill: #8c959f; font-size: 12px; }}
.axis {{ fill: #57606a; font-size: 12px; }}
.pt text {{ font-size: 11px; fill: #1f2328; }}
.lead {{ stroke: #d0d7de; }}
.pt {{ cursor: pointer; }}
.pt:hover circle {{ stroke: #000; stroke-width: 2; }}
#card {{ white-space: pre-wrap; font: 12px ui-monospace, monospace; background: #f6f8fa;
         padding: 12px; border-radius: 6px; max-width: 760px; min-height: 3em; }}
.legend span {{ margin-right: 14px; }}
</style></head><body>
<h1>Research landscape</h1>
<p class="legend"><span style="color:#1a7f37">● GO</span><span style="color:#9a6700">● WAIT</span>
<span style="color:#8c959f">● SKIP</span> Hover or tap a field to see its record.</p>
{svg}
<div id="card">Hover a field.</div>
<script>
document.querySelectorAll('.pt').forEach(function (g) {{
  function show() {{ document.getElementById('card').textContent =
    JSON.stringify(JSON.parse(g.dataset.field), null, 2); }}
  g.addEventListener('mouseenter', show); g.addEventListener('click', show);
}});
</script>
</body></html>
"""


def write_reports(doc, out_dir=None):
    out_dir = out_dir or config.OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    md, page = out_dir / "landscape.md", out_dir / "landscape.html"
    md.write_text(markdown(doc), encoding="utf-8")
    page.write_text(PAGE.format(svg=scatter(doc)), encoding="utf-8")
    return md, page


if __name__ == "__main__":
    import landscape
    landscape.ensure_live()
    for path in write_reports(landscape.load_fields()):
        print(f"Wrote {path}")
