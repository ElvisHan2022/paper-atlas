"""Milestone 7: a running map of which research fields are crowded and which are open.

    python landscape.py --rescore-only        # recompute scores from the evidence log
    python landscape.py --check-seed          # the refactor still matches landscape_score.py
    python landscape.py --import-private <unzipped package folder or .zip>

The tracked seed lives in landscape/ (no personal data). The first run copies it to
data/landscape/, and every topic run after that adds one observation, nudges the field's
quality metrics, rescores, and logs what changed. Priors are human judgments: runs only
ever *suggest* changes to them, in data/landscape/review_queue.md.
"""
import argparse
import csv
import io
import json
import math
import re
import shutil
import statistics
import zipfile
from datetime import date

import pandas as pd

import config
import db

QUALITY_KEYS = ["external_validation_rate", "uncertainty_reported_rate", "open_data_rate",
                "transport_limitation_rate", "review_share", "top_journal_share"]
COUNT_SOURCES = ["openalex", "pubmed_connector"]   # bibliometric sources, preferred first
STOPWORDS = {"a", "an", "and", "or", "not", "the", "of", "in", "on", "for", "to", "with", "by",
             "from", "at", "as", "is", "are", "its", "into", "vs", "versus", "mesh", "terms"}


# ---- files ----

def live_path(name):
    return config.LANDSCAPE_DIR / name


def ensure_live():
    """Copy the tracked seed to data/landscape/ on first use, scored with the private fit."""
    if live_path("fields.json").exists():
        return
    config.LANDSCAPE_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("fields.json", "observations.jsonl"):
        shutil.copyfile(config.LANDSCAPE_SEED_DIR / name, live_path(name))
    doc = load_fields()
    rescore(doc)
    save_fields(doc)


def load_fields(path=None):
    """The whole document: {"meta": ..., "fields": [...]}."""
    path = path or live_path("fields.json")
    return json.loads(path.read_text(encoding="utf-8"))


def save_fields(doc, path=None):
    """Write to a temp file, then replace, so a crash mid-write can't corrupt the map."""
    path = path or live_path("fields.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def load_observations(path=None):
    path = path or live_path("observations.jsonl")
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_observation(field_id, obs, topic=None, path=None, source="atlas_run",
                       kind="quality", **extra):
    """Append-only: past observations are never edited or deleted."""
    path = path or live_path("observations.jsonl")
    record = {"date": date.today().isoformat(), "field": field_id, "source": source,
              "kind": kind, **({"topic": topic} if topic else {}), **extra, "data": obs}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return record


def private_fit():
    """{field id: fit 0-3} from the gitignored landscape/private/fit.csv, or {}."""
    path = config.LANDSCAPE_PRIVATE_DIR / "fit.csv"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8", newline="") as f:
        return {row["id"]: float(row["fit"]) for row in csv.DictReader(f) if row.get("fit")}


def context_text():
    """CONTEXT.md plus the owner's private lens when present, for any LLM-written summary."""
    text = (config.LANDSCAPE_SEED_DIR / "CONTEXT.md").read_text(encoding="utf-8")
    owner = config.LANDSCAPE_PRIVATE_DIR / "owner.md"
    if owner.exists():
        text += "\n\n" + owner.read_text(encoding="utf-8")
    return text


# ---- matching a topic to a field ----

def words(text):
    """Lowercase content words, plurals folded, so "LLMs" matches "LLM"."""
    text = re.sub(r"\[[^\]]*\]", " ", (text or "").lower()).replace("'s", "")
    out = []
    for w in re.split(r"[^a-z0-9]+", text):
        if len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "is", "us")):
            w = w[:-1]
        if w and w not in STOPWORDS:
            out.append(w)
    return out


def key_phrases(query):
    """The OR'd terms of a PubMed query, each as a set of words, plus their acronyms."""
    if not query or " AND " in query:
        return []   # AND queries need several terms together; leave them to word overlap
    phrases = []
    for term in re.split(r"\s+OR\s+", query):
        ws = words(term)
        content = {w for w in ws if w not in config.LANDSCAPE_GENERIC_WORDS}
        if content:
            phrases.append(content)
        acronym = "".join(w[0] for w in ws)
        if len(ws) >= 2 and len(acronym) >= 3:
            phrases.append({acronym})   # "large language model" -> llm
    return phrases


def vocabulary(field):
    queries = field.get("queries") or {}
    vocab = set(words(queries.get("pubmed"))) | set(words(field.get("label")))
    vocab |= set(words(queries.get("topic")))
    for phrase in key_phrases(queries.get("pubmed")):
        vocab |= phrase
    return vocab - config.LANDSCAPE_GENERIC_WORDS


def match_score(topic_words, field):
    """1.0 when the topic contains one of the field's search terms whole; otherwise the
    share of the topic's words that the field's query or label uses."""
    if not topic_words:
        return 0.0
    queries = field.get("queries") or {}
    phrases = key_phrases(queries.get("pubmed"))
    if field.get("status") == "provisional":
        phrases.append(set(words(field.get("label"))) - config.LANDSCAPE_GENERIC_WORDS)
    if any(p and p <= topic_words for p in phrases):
        return 1.0
    return len(topic_words & vocabulary(field)) / len(topic_words)


def match_topic(topic, fields):
    """Field ids that cover the topic, best first. If none do, a new provisional id."""
    topic_words = set(words(topic)) - config.LANDSCAPE_GENERIC_WORDS
    scored = [(match_score(topic_words, f), f["id"]) for f in fields]
    hits = [fid for s, fid in sorted(scored, key=lambda x: -x[0])
            if s >= config.LANDSCAPE_MATCH_THRESHOLD]
    if hits:
        return hits
    base = "_".join([w for w in words(topic) if w not in config.LANDSCAPE_GENERIC_WORDS][:4]) or "topic"
    taken = {f["id"] for f in fields}
    new_id, i = base, 2
    while new_id in taken:
        new_id, i = f"{base}_{i}", i + 1
    return [new_id]


def new_provisional(field_id, topic):
    return {"id": field_id, "label": " ".join(words(topic)), "status": "provisional",
            "queries": {"topic": topic}, "priors": dict(config.LANDSCAPE_DEFAULT_PRIORS),
            "signal": "WAIT",
            "note": "Created by an atlas run; add a PubMed query and priors before scoring.",
            "history": [{"date": date.today().isoformat(), "event": "created_provisional",
                         "topic": topic}]}


# ---- what one run says about a field ----

def normalize_venue(venue):
    v = re.sub(r"[^a-z0-9 ]+", " ", (venue or "").lower())
    return re.sub(r"\s+", " ", v).strip()


TOP_VENUES = {normalize_venue(s) for spellings in config.TOP_JOURNALS.values() for s in spellings}


def share(flags):
    flags = list(flags)
    return round(sum(flags) / len(flags), 3) if flags else None


def mentions(text, terms):
    text = (text or "").lower()
    return any(t in text for t in terms)


def community_count(conn, ids):
    """Louvain communities in the citation and coupling graph of these papers, or None
    when there are no edges (web searches don't fetch references)."""
    import networkx as nx
    from networkx.algorithms.community import louvain_communities
    rows = conn.execute("SELECT src, dst, weight FROM edges").fetchall()
    g = nx.Graph()
    g.add_weighted_edges_from((r["src"], r["dst"], r["weight"] or 1.0) for r in rows
                              if r["src"] in ids and r["dst"] in ids and r["src"] != r["dst"])
    if g.number_of_edges() == 0:
        return None
    return len(louvain_communities(g, weight="weight", seed=config.RANDOM_SEED))


def observe_run(topic, db_path=None):
    """Summarize the papers and extractions stored for one topic."""
    conn = db.connect(db_path)
    try:
        papers = db.papers_for_topic(conn, topic)
        extractions = [d for d, _ in db.extractions_for_topic(conn, topic).values()]
        by_year = {}
        for p in papers:
            if p["year"]:
                by_year[str(p["year"])] = by_year.get(str(p["year"]), 0) + 1
        typed = [json.loads(p["publication_types"]) for p in papers if p["publication_types"]]
        obs = {
            "n_papers": len(papers),
            "papers_by_year": dict(sorted(by_year.items())),
            "ai_share": share(mentions(f"{p['title']} {p['abstract']}", config.LANDSCAPE_AI_TERMS)
                              for p in papers),
            "review_share": share(any(re.search(r"review|meta", t, re.I) for t in types)
                                  for types in typed),
            "median_citations": (statistics.median(p["citation_count"] or 0 for p in papers)
                                 if papers else None),
            "top_journal_share": share(normalize_venue(p["venue"]) in TOP_VENUES for p in papers),
            "communities": community_count(conn, {p["paper_id"] for p in papers}),
            "n_extracted": len(extractions),
        }
    finally:
        conn.close()
    enough = len(extractions) >= config.LANDSCAPE_MIN_EXTRACTED
    design = [e["study_design"] for e in extractions]
    obs.update({
        # "unclear" counts as not external: a reader can't rely on validation nobody reported.
        "external_validation_rate": share(d["validation"] == "external" for d in design) if enough else None,
        "uncertainty_reported_rate": (share(e["results"]["uncertainty_reported"] is True
                                            for e in extractions) if enough else None),
        "open_data_rate": (share(mentions(d["data_source"], config.OPEN_DATA_NAMES) for d in design)
                           if enough else None),
        "transport_limitation_rate": (share(mentions(" ".join(e["limitations_stated"]),
                                                     config.TRANSPORT_TERMS) for e in extractions)
                                      if enough else None),
    })
    return obs


def update_metrics(field, obs, alpha=config.LANDSCAPE_ALPHA):
    """Exponentially weighted average: each run moves a metric `alpha` of the way toward
    what it saw, so one narrow query can't swing the map."""
    quality = field.setdefault("quality", {})
    for key in QUALITY_KEYS:
        new = obs.get(key)
        if new is None:
            quality.setdefault(key, None)
            continue
        old = quality.get(key)
        quality[key] = round(new if old is None else alpha * new + (1 - alpha) * old, 4)
    evr = quality.get("external_validation_rate")
    quality["warrant_gap"] = None if evr is None else round(1 - evr, 4)
    quality["observations"] = quality.get("observations", 0) + 1
    return field


# ---- scoring (the logic of landscape/landscape_score.py) ----

def pct_rank(series):
    """Percentile rank in [0, 100]; ranks are robust to the heavy-tailed counts."""
    return series.rank(pct=True) * 100


def safe_ratio(num, den, floor=1):
    """Growth ratio that tolerates fields that did not exist in 2021."""
    return num / max(den, floor)


def classify(momentum, crowding, whitespace):
    if crowding >= 50 and momentum < 50:
        return "Crowded, decelerating"
    if crowding >= 50 and momentum >= 50:
        return "Hot and contested"
    if momentum >= 50:
        return "Blooming"
    if whitespace >= 60:
        return "Dormant whitespace (ML gap)"
    return "Quiet niche"


def latest_counts(observations):
    """Per field: market counts from one source for every field (percentiles only compare
    like with like), the RCT base, and the arXiv share-adjusted growth."""
    by_source = {}
    rct, arxiv = {}, {}
    for o in observations:   # file order is date order, so later lines win
        data = o.get("data") or {}
        if o.get("source") in COUNT_SOURCES and "n2025" in data:
            by_source.setdefault(o["source"], {})[o["field"]] = data
        if data.get("rct") is not None:
            rct[o["field"]] = (data["rct"], data["n2025"])
        if data.get("share_adj_2025_vs_2021") is not None:
            arxiv[o["field"]] = data["share_adj_2025_vs_2021"]
    return by_source, rct, arxiv


def rescore(doc, observations=None, fit=None):
    """Recompute metrics, scores and quadrant for every active field with counts and priors.
    Provisional fields stay listed but unranked."""
    observations = load_observations() if observations is None else observations
    fit = private_fit() if fit is None else fit
    by_source, rct, arxiv = latest_counts(observations)
    active = [f for f in doc["fields"] if f.get("status") == "active" and f.get("priors")]
    # The first source that has counts for every active field; otherwise the one covering most.
    source = max(COUNT_SOURCES, key=lambda s: (len(set(by_source.get(s, {}))
                                                   & {f["id"] for f in active}),
                                               -COUNT_SOURCES.index(s)))
    counts = by_source.get(source, {})
    ranked = [f for f in active if f["id"] in counts]
    if not ranked:
        return doc
    df = pd.DataFrame([{"id": f["id"], **counts[f["id"]], **f["priors"],
                        "fit_value": fit.get(f["id"], config.LANDSCAPE_DEFAULT_FIT)}
                       for f in ranked])
    df["growth"] = [safe_ratio(b, a) for a, b in zip(df.n2021, df.n2025)]
    df["ai_growth"] = [safe_ratio(b, a) for a, b in zip(df.ai2021, df.ai2025)]
    df["arxiv_growth"] = df.id.map(arxiv)
    momentum_parts = [pct_rank(df.growth.map(math.log)), pct_rank(df.ai_growth.map(math.log))]
    arxiv_pct = pct_rank(df.arxiv_growth.dropna().map(math.log))
    momentum_parts.append(arxiv_pct.reindex(df.index))
    df["momentum"] = pd.concat(momentum_parts, axis=1).mean(axis=1, skipna=True)
    df["crowding"] = pct_rank(df.ai2025.map(lambda x: math.log(x + 1)))
    df["ai_share_2025"] = df.ai2025 / df.n2025
    df["whitespace"] = pct_rank((df.n2025 / (df.ai2025 + 1)).map(math.log))
    df["rct_index"] = [rct[i][0] / rct[i][1] if i in rct else float("nan") for i in df.id]
    df["evidence_maturity"] = pct_rank(df.rct_index)

    w = doc["meta"]["weights"]
    wo, we = w["opportunity"], w["entry"]
    df["opportunity"] = (wo["momentum"] * df.momentum + wo["whitespace"] * df.whitespace
                         + wo["inverse_crowding"] * (100 - df.crowding))
    df["feasibility"] = (df.data_access + df.mvp_simplicity + df.domain_access) / 9 * 100
    df["fit"] = df.fit_value / 3 * 100
    df["entry_score"] = (we["opportunity"] * df.opportunity + we["feasibility"] * df.feasibility
                         + we["fit"] * df.fit)

    def num(x):
        return None if pd.isna(x) else round(float(x), 2)

    rows = {r["id"]: r for r in df.to_dict(orient="records")}
    for f in ranked:
        r = rows[f["id"]]
        f["metrics"] = {k: num(r[k]) for k in
                        ["momentum", "crowding", "whitespace", "evidence_maturity", "growth",
                         "ai_growth", "arxiv_growth", "ai_share_2025", "rct_index"]}
        f["scores"] = {k: num(r[k]) for k in ["entry_score", "opportunity", "feasibility", "fit"]}
        f["quadrant"] = classify(r["momentum"], r["crowding"], r["whitespace"])
    doc["meta"]["counts_source"] = source
    return doc


# ---- what changed, and what a human should look at ----

def log_changes(old, new, cause=None, path=None):
    """For each field whose quadrant or signal changed: a history entry and a changes line."""
    before = {f["id"]: f for f in old["fields"]}
    lines = []
    today = date.today().isoformat()
    for f in new["fields"]:
        prev = before.get(f["id"])
        if prev is None:
            continue
        changed = {k: [prev.get(k), f.get(k)] for k in ("quadrant", "signal")
                   if prev.get(k) != f.get(k)}
        if not changed:
            continue
        entry = {"date": today, "event": "changed", **changed,
                 "entry_score": (f.get("scores") or {}).get("entry_score"), "cause": cause}
        f.setdefault("history", []).append(entry)
        what = "; ".join(f"{k}: {a} -> {b}" for k, (a, b) in changed.items())
        lines.append(f"- {today} **{f['id']}**: {what} (cause: {cause or 'rescore'})")
    if lines:
        path = path or config.OUTPUT_DIR / "landscape_changes.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not path.exists()
        with open(path, "a", encoding="utf-8") as f:
            if new_file:
                f.write("# Landscape changes\n\n")
            f.write("\n".join(lines) + "\n")
    return lines


def propose_priors(field, obs, path=None):
    """Never edits priors. Writes suggestions, with their evidence, for the owner to review."""
    path = path or live_path("review_queue.md")
    today = date.today().isoformat()
    items = []
    if field.get("status") == "provisional" and len(field.get("history", [])) == 1:
        items.append((f"{field['id']}:new",
                      f"New provisional field **{field['id']}** from the topic "
                      f"\"{(field.get('queries') or {}).get('topic')}\". Add a PubMed query, "
                      f"priors (data_access, mvp_simplicity, domain_access) and status "
                      f"\"active\" to rank it, or merge it into an existing field."))
    priors = field.get("priors") or {}
    rate = (field.get("quality") or {}).get("open_data_rate")
    access = priors.get("data_access")
    if rate is not None and access is not None:
        if rate < config.LANDSCAPE_OPEN_DATA_MIN and access > 1:
            items.append((f"{field['id']}:data_access:{access - 1}",
                          f"Lower **{field['id']}** data_access {access} -> {access - 1}: only "
                          f"{rate:.0%} of extracted papers name an open dataset (this run: "
                          f"{obs.get('open_data_rate')} of {obs.get('n_extracted')})."))
        elif rate >= 3 * config.LANDSCAPE_OPEN_DATA_MIN and access < 3:
            items.append((f"{field['id']}:data_access:{access + 1}",
                          f"Raise **{field['id']}** data_access {access} -> {access + 1}: "
                          f"{rate:.0%} of extracted papers name an open dataset (this run: "
                          f"{obs.get('open_data_rate')} of {obs.get('n_extracted')})."))
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    fresh = [(key, text) for key, text in items if f"<!-- {key} -->" not in existing]
    if fresh:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            if not existing:
                f.write("# Review queue\n\nSuggested changes to human judgments. Nothing here "
                        "is applied automatically; edit fields.json yourself if you agree.\n\n")
            for key, text in fresh:
                f.write(f"- [ ] {today} {text} <!-- {key} -->\n")
    return [text for _, text in fresh]


# ---- one topic run, end to end ----

def update_from_run(topic, db_path=None):
    """Everything a finished topic run does to the map. Returns a short summary."""
    import copy
    ensure_live()
    doc = load_fields()
    old = copy.deepcopy(doc)
    field_id = match_topic(topic, doc["fields"])[0]
    field = next((f for f in doc["fields"] if f["id"] == field_id), None)
    if field is None:
        field = new_provisional(field_id, topic)
        doc["fields"].append(field)
    obs = observe_run(topic, db_path)
    append_observation(field_id, obs, topic)
    update_metrics(field, obs)
    rescore(doc)
    changes = log_changes(old, doc, cause=f"atlas run: {topic}")
    suggestions = propose_priors(field, obs)
    save_fields(doc)
    import landscape_report
    landscape_report.write_reports(doc)
    return {"field": field_id, "label": field.get("label"), "status": field.get("status"),
            "quadrant": field.get("quadrant"), "signal": field.get("signal"),
            "entry_score": (field.get("scores") or {}).get("entry_score"),
            "warrant_gap": field["quality"].get("warrant_gap"),
            "changes": changes, "suggestions": suggestions}


# ---- what the app shows on its "How it works" page ----

def current_doc():
    """The live map if a run has made one, else the tracked seed. Reading never writes."""
    live = live_path("fields.json")
    if live.exists():
        return load_fields(live)
    doc = load_fields(config.LANDSCAPE_SEED_DIR / "fields.json")
    if private_fit():   # the seed's scores use the neutral fit; show the owner's
        rescore(doc, load_observations(config.LANDSCAPE_SEED_DIR / "observations.jsonl"))
    return doc


def summary():
    """The landscape's current state, for people: ranking, evidence so far, what changed,
    and what is waiting for a human decision."""
    live = live_path("fields.json").exists()
    doc = current_doc()
    observations = load_observations(live_path("observations.jsonl") if live
                                     else config.LANDSCAPE_SEED_DIR / "observations.jsonl")
    runs = [o for o in observations if o.get("source") == "atlas_run"]
    changes_file = config.OUTPUT_DIR / "landscape_changes.md"
    changes = ([l[2:] for l in changes_file.read_text(encoding="utf-8").splitlines()
                if l.startswith("- ")][-5:] if changes_file.exists() else [])
    queue_file = live_path("review_queue.md")
    queue = ([re.sub(r"\s*<!--.*?-->", "", l[6:]) for l in
              queue_file.read_text(encoding="utf-8").splitlines() if l.startswith("- [ ] ")]
             if queue_file.exists() else [])
    ranked = sorted((f for f in doc["fields"] if f.get("scores") and f.get("status") == "active"),
                    key=lambda f: -f["scores"]["entry_score"])
    return {
        "live": live,
        "private_fit": bool(private_fit()),
        "counts_source": doc["meta"].get("counts_source", "pubmed_connector"),
        "weights": doc["meta"]["weights"],
        "n_observations": len(observations),
        "n_runs": len(runs),
        "recent_runs": [{"date": o["date"], "topic": o.get("topic"), "field": o["field"]}
                        for o in runs[-5:]][::-1],
        "fields": [{"id": f["id"], "label": f["label"], "signal": f.get("signal"),
                    "regime": f.get("quadrant"), "entry": f["scores"]["entry_score"],
                    "momentum": f["metrics"]["momentum"], "crowding": f["metrics"]["crowding"],
                    "warrant_gap": (f.get("quality") or {}).get("warrant_gap"),
                    "runs": (f.get("quality") or {}).get("observations", 0)} for f in ranked],
        "provisional": [{"id": f["id"], "label": f["label"]} for f in doc["fields"]
                        if f.get("status") == "provisional"],
        "changes": changes,
        "review_queue": queue,
    }


# ---- one-time import of the owner's private lens ----

def import_private(package):
    """Pull fit_elvis and the owner's lens out of the original package (folder or .zip)
    into the gitignored landscape/private/."""
    from pathlib import Path
    package = Path(package)

    def read(name):
        if package.suffix == ".zip":
            with zipfile.ZipFile(package) as z:
                match = next(n for n in z.namelist() if n.endswith(f"landscape/{name}"))
                return z.read(match).decode("utf-8")
        found = next(package.rglob(name))
        return found.read_text(encoding="utf-8")

    out = config.LANDSCAPE_PRIVATE_DIR
    out.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(io.StringIO(read("field_priors.csv"))))
    with open(out / "fit.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "fit"])
        writer.writerows([r["id"], r["fit_elvis"]] for r in rows if r.get("fit_elvis"))
    context = read("CONTEXT.md")
    sections = re.split(r"(?m)^(?=## )", context)
    keep = [s for s in sections if s.startswith(("## 4.", "## 5."))]
    (out / "owner.md").write_text("# Owner's lens (private, never committed)\n\n"
                                  + "\n".join(s.strip() + "\n" for s in keep), encoding="utf-8")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rescore-only", action="store_true", help="rescore the live map")
    ap.add_argument("--check-seed", action="store_true",
                    help="rescore the tracked seed and compare with its stored entry scores")
    ap.add_argument("--import-private", metavar="PACKAGE",
                    help="original landscape package (folder or .zip) to take fit and lens from")
    args = ap.parse_args()
    if args.import_private:
        print(f"Wrote {import_private(args.import_private)} (gitignored).")
        if live_path("fields.json").exists():
            doc = load_fields()
            save_fields(rescore(doc))
            import landscape_report
            landscape_report.write_reports(doc)
            print("Rescored the live map with your fit.")
    elif args.check_seed:
        seed = load_fields(config.LANDSCAPE_SEED_DIR / "fields.json")
        stored = {f["id"]: f["scores"]["entry_score"] for f in seed["fields"] if f.get("scores")}
        fresh = rescore(json.loads(json.dumps(seed)),
                        load_observations(config.LANDSCAPE_SEED_DIR / "observations.jsonl"), fit={})
        worst = max(abs(f["scores"]["entry_score"] - stored[f["id"]])
                    for f in fresh["fields"] if f["id"] in stored)
        print(f"{len(stored)} fields; largest entry_score difference {worst:.4f}")
        raise SystemExit(0 if worst <= 0.01 else 1)
    elif args.rescore_only:
        ensure_live()
        doc = load_fields()
        old = json.loads(json.dumps(doc))
        rescore(doc)
        log_changes(old, doc, cause="rescore")
        save_fields(doc)
        import landscape_report
        md, html = landscape_report.write_reports(doc)
        print(f"Rescored {sum(1 for f in doc['fields'] if f.get('scores'))} fields. "
              f"Wrote {md} and {html}")
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
