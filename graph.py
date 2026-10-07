"""Milestone 5: a Litmaps-style map of the relevant papers plus Obsidian-ready notes.

    python graph.py --topic "..."

Writes outputs/graph.json, outputs/graph.html, and outputs/notes/<slug>.md.
"""
import argparse
import json
import math
import random
import re
from itertools import combinations

import networkx as nx
from networkx.algorithms.community import louvain_communities

import config
import db
from fetch import load_references

# Categorical colors in a fixed order (largest community first); extras fold into gray.
COMMUNITY_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                    "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
OTHER_COLOR = "#9a9893"
X_PER_YEAR = 140   # pixels between years on the map
Y_PER_LOG = 160    # pixels per factor of 10 in citations


def coupling_weight(refs_a, refs_b):
    """Bibliographic coupling: how many references two papers share."""
    return len(set(refs_a) & set(refs_b))


def relevant_papers(conn, topic):
    """LLM score >= 0.75, or labeled relevant. A human 'no' overrides the LLM."""
    labels = db.labels_for_topic(conn, topic)
    llm_scores = db.scores_for_topic(conn, topic, "llm")
    keep = []
    for p in db.papers_for_topic(conn, topic):
        label = labels.get(p["paper_id"])
        llm_ok = (llm_scores.get(p["paper_id"]) or 0) >= config.LLM_RELEVANT_CUTOFF
        if label == 1 or (label is None and llm_ok):
            keep.append(p)
    return keep


def build_graph(conn, papers):
    """Nodes = papers. Edges = within-set citations and coupling (>= 2 shared refs)."""
    g = nx.Graph()
    ids = {p["paper_id"] for p in papers}
    g.add_nodes_from(ids)

    cites = [(r["src"], r["dst"]) for r in conn.execute(
        "SELECT src, dst FROM edges WHERE kind='cites'") if r["src"] in ids and r["dst"] in ids]
    for src, dst in cites:
        g.add_edge(src, dst, cites=True, weight=1.0)

    refs = {pid: load_references(pid) for pid in ids}
    coupling = []
    for a, b in combinations(sorted(ids), 2):
        w = coupling_weight(refs[a], refs[b])
        if w >= config.COUPLING_MIN_SHARED:
            coupling.append((a, b, w))
            db.add_edge(conn, a, b, "coupling", w)
            # Louvain sees one weighted edge per pair; citation and coupling add up.
            prev = g.get_edge_data(a, b, {}).get("weight", 0.0)
            g.add_edge(a, b, weight=prev + w)
    conn.commit()
    return g, cites, coupling


def communities(g):
    """Louvain community id per paper, 0 = largest community. Seeded for reproducibility."""
    groups = louvain_communities(g, weight="weight", seed=config.RANDOM_SEED)
    groups = sorted(groups, key=lambda c: (-len(c), min(c)))
    return {pid: i for i, group in enumerate(groups) for pid in group}


def slugify(title, used):
    base = re.sub(r"[^a-z0-9]+", "-", (title or "untitled").lower()).strip("-")[:60].strip("-")
    slug, n = base or "paper", 2
    while slug in used:
        slug, n = f"{base}-{n}", n + 1
    used.add(slug)
    return slug


def relevance(pid, scores, labels):
    """Size nodes by LLM score; fall back to a cross-encoder, then to the label."""
    for sc in ("llm", "bge", "minilm"):
        if scores[sc].get(pid) is not None:
            return scores[sc][pid]
    return float(labels.get(pid, 0.5))


def layout(papers):
    """Preset positions: x = year, y = log10(citations + 1), plus a little jitter."""
    rng = random.Random(config.RANDOM_SEED)
    years = [p["year"] for p in papers if p["year"]]
    fallback_year = sorted(years)[len(years) // 2] if years else 2020
    pos = {}
    for p in papers:
        year = p["year"] or fallback_year
        x = year * X_PER_YEAR + rng.uniform(-45, 45)
        y = -math.log10((p["citation_count"] or 0) + 1) * Y_PER_LOG + rng.uniform(-18, 18)
        pos[p["paper_id"]] = {"x": round(x, 1), "y": round(y, 1)}
    return pos


def build_elements(papers, cites, coupling, comm, scores, labels, extractions, slugs):
    pos = layout(papers)
    nodes = []
    for p in papers:
        pid = p["paper_id"]
        c = comm.get(pid, 0)
        ext = extractions.get(pid)
        nodes.append({
            "data": {
                "id": pid, "slug": slugs[pid], "title": p["title"], "year": p["year"],
                "venue": p["venue"], "citations": p["citation_count"],
                "authors": json.loads(p["authors_json"] or "[]")[:6],
                "community": c,
                "color": COMMUNITY_COLORS[c] if c < len(COMMUNITY_COLORS) else OTHER_COLOR,
                "size": round(16 + 34 * relevance(pid, scores, labels), 1),
                "scores": {sc: scores[sc].get(pid) for sc in scores},
                "label": labels.get(pid),
                "extraction": ext[0] if ext else None,
                "source_text_kind": ext[1] if ext else None,
                "url": f"https://www.semanticscholar.org/paper/{pid}",
            },
            "position": pos[pid],
        })
    edges = [{"data": {"id": f"c:{s}:{d}", "source": s, "target": d, "kind": "cites"}}
             for s, d in cites]
    edges += [{"data": {"id": f"k:{a}:{b}", "source": a, "target": b, "kind": "coupling",
                        "weight": w}} for a, b, w in coupling]

    years = sorted({p["year"] for p in papers if p["year"]})
    axis = [{"data": {"id": f"year-{y}", "axis": str(y)}, "classes": "axis",
             "position": {"x": y * X_PER_YEAR, "y": 70}} for y in years]
    return nodes, edges, axis


def write_notes(papers, cites, coupling, slugs, extractions, scores, topic):
    """One markdown note per paper, linked with [[wikilinks]] for Obsidian."""
    config.NOTES_DIR.mkdir(parents=True, exist_ok=True)
    for old in config.NOTES_DIR.glob("*.md"):
        old.unlink()  # stale notes from an earlier run would leave dangling links
    cited_by_me, coupled = {}, {}
    for s, d in cites:
        cited_by_me.setdefault(s, []).append(d)
    for a, b, w in coupling:
        coupled.setdefault(a, []).append((b, w))
        coupled.setdefault(b, []).append((a, w))

    for p in papers:
        pid = p["paper_id"]
        lines = [
            "---",
            f"title: {json.dumps(p['title'])}",
            f"year: {p['year']}",
            f"citations: {p['citation_count']}",
            f"topic: {json.dumps(topic)}",
            f"semantic_scholar: https://www.semanticscholar.org/paper/{pid}",
            "---",
            f"# {p['title']}",
            "",
            f"**{p['venue'] or 'venue unknown'}, {p['year']}** · "
            + " · ".join(f"{sc}: {scores[sc][pid]:.2f}" for sc in scores
                         if scores[sc].get(pid) is not None),
            "",
        ]
        ext = extractions.get(pid)
        if ext:
            data, kind = ext
            lines.append(f"_Extracted from {kind.replace('_', ' ')}._\n")
            lines += extraction_markdown(data)
        else:
            lines += ["## Abstract", "", p["abstract"] or "", "", "_No extraction yet._", ""]

        lines += ["## Links", ""]
        lines += [f"- cites [[{slugs[d]}]]" for d in cited_by_me.get(pid, [])]
        lines += [f"- shares {w} references with [[{slugs[o]}]]"
                  for o, w in sorted(coupled.get(pid, []), key=lambda t: -t[1])]
        if not cited_by_me.get(pid) and not coupled.get(pid):
            lines.append("- no links within this set")
        (config.NOTES_DIR / f"{slugs[pid]}.md").write_text("\n".join(lines) + "\n",
                                                          encoding="utf-8")


def bullets(items, empty):
    items = [x for x in items if x and x.strip()]
    return [f"- {x}" for x in items] or [f"- {empty}"]


def extraction_markdown(d):
    sd, m, r, ev = d["study_design"], d["methods"], d["results"], d["evidence"]
    return [
        "## Study design",
        f"- Type: {sd['type']}", f"- Population: {sd['population']}",
        f"- Sample size: {sd['sample_size']}", f"- Data source: {sd['data_source']}",
        f"- Validation: {sd['validation']}", f"- Evidence: {ev['study_design']}", "",
        "## Methods",
        f"- Approach: {m['approach']}", f"- Baselines: {m['baselines']}",
        f"- Metrics: {m['metrics']}", f"- Evidence: {ev['methods']}", "",
        "## Results",
        f"- Headline: {r['headline']}",
        f"- Uncertainty reported: {'yes' if r['uncertainty_reported'] else 'no'}",
        f"- Evidence: {ev['results']}", "",
        "## Limitations (stated by authors)",
        *bullets(d["limitations_stated"], "none stated"),
        f"- Evidence: {ev['limitations_stated']}", "",
        "## Limitations (inferred, not from the authors)",
        *bullets(d["limitations_inferred"], "none"), "",
        "## Discussion", d["discussion"], "",
    ]


def main(topic):
    conn = db.connect()
    papers = relevant_papers(conn, topic)
    if not papers:
        raise SystemExit("No relevant papers yet. Run score.py (and/or label.py) first.")
    g, cites, coupling = build_graph(conn, papers)
    comm = communities(g)

    scores = {sc: db.scores_for_topic(conn, topic, sc) for sc in ("minilm", "bge", "llm")}
    labels = db.labels_for_topic(conn, topic)
    extractions = db.extractions_for_topic(conn, topic)
    used = set()
    slugs = {p["paper_id"]: slugify(p["title"], used)
             for p in sorted(papers, key=lambda p: p["paper_id"])}

    nodes, edges, axis = build_elements(papers, cites, coupling, comm, scores, labels,
                                        extractions, slugs)
    payload = {"topic": topic, "nodes": nodes, "edges": edges, "axis": axis,
               "n_communities": len(set(comm.values()))}

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (config.OUTPUT_DIR / "graph.json").write_text(json.dumps(payload, indent=1),
                                                  encoding="utf-8")
    # Inline the data. Escaping "</" keeps a title like "</script>" from ending the tag.
    inline = json.dumps(payload).replace("</", "<\\/")
    html = config.GRAPH_TEMPLATE.read_text(encoding="utf-8")
    html = html.replace("/*__GRAPH_DATA__*/null", inline)
    (config.OUTPUT_DIR / "graph.html").write_text(html, encoding="utf-8")

    write_notes(papers, cites, coupling, slugs, extractions, scores, topic)
    print(f"[graph] {len(papers)} relevant papers, {len(cites)} cites edges, "
          f"{len(coupling)} coupling edges, {payload['n_communities']} communities. "
          f"Wrote outputs/graph.html and {len(papers)} notes.")
    return payload


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--topic", required=True)
    args = ap.parse_args()
    main(args.topic)
