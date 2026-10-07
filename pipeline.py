"""One search, start to finish, for the web app: query -> candidates -> scores -> review.

app.py runs `run(query, report)` in a background thread. `report(stage, fraction, message)`
is how this file tells the browser how far along it is.
"""
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import numpy as np

import config
import db
import extract
import fetch
import llm
import score

STAGES = [
    "Processing your query",
    "Identifying candidate papers",
    "Scoring the papers",
    "Finishing scoring",
]
# Share of the overall progress bar each stage owns (start %, end %).
STAGE_SPAN = [(0, 8), (8, 22), (22, 70), (70, 100)]

RUBRIC = [
    (5, "Core", "Directly answers the topic; would anchor a review of it."),
    (4, "Relevant", "Addresses the topic with its own data or method."),
    (3, "Related background", "Same area, but the topic is not its main question."),
    (2, "Tangential", "Shares keywords or methods, different question."),
    (1, "Unrelated", "Off topic."),
]

REVIEW_SECTIONS = [
    ("study_design", "Study design"),
    ("methods", "Methods"),
    ("results", "Results"),
    ("limitations", "Limitations"),
    ("discussion", "Discussion"),
]


class PipelineError(Exception):
    """A problem the person searching can act on (shown in the UI as-is)."""


def overall_percent(stage, fraction):
    start, end = STAGE_SPAN[stage]
    return start + (end - start) * max(0.0, min(1.0, fraction))


# ---------- stage 1: query ----------

def understand_query(query):
    """Turn loose keywords into a topic sentence plus relevance criteria for the scorers."""
    system = "You help researchers turn a search into precise inclusion criteria."
    user = (f'Search: "{query}"\n\nReturn JSON only: {{"topic": "<the search restated as '
            'one clear research topic, under 15 words>", "criteria": "<2 sentences on what '
            'kinds of papers count as relevant>"}}')
    data, stats = llm.ask_json(
        system, user, 300,
        lambda d: isinstance(d.get("topic"), str) and isinstance(d.get("criteria"), str))
    if not data:  # keep going with a plain fallback rather than failing the search
        return query, f"Papers whose main contribution addresses: {query}.", stats
    return data["topic"].strip() or query, data["criteria"].strip(), stats


# ---------- stage 3: scoring ----------

def judge_many(topic, criteria, papers, on_done):
    """LLM-judge papers in parallel. Returns {paper_id: (score, rationale, stats)}."""
    llm.client()  # create the shared client once, before the threads start
    out = {}
    with ThreadPoolExecutor(max_workers=config.WEB_WORKERS) as pool:
        futures = {pool.submit(score.judge, topic, criteria, p): p["paper_id"] for p in papers}
        for f in as_completed(futures):
            out[futures[f]] = f.result()
            on_done(len(out))
    return out


def rank_key(p):
    """LLM rubric first; the cross-encoder average breaks ties and ranks unjudged papers."""
    llm_score = p["scores"]["llm"]
    return (llm_score if llm_score is not None else -1.0, p["ce_mean"])


# ---------- stage 4: review + clusters ----------

def extract_many(papers, on_done):
    out = {}
    with ThreadPoolExecutor(max_workers=config.WEB_WORKERS) as pool:
        futures = {pool.submit(extract.extract_one, p): p["paper_id"] for p in papers}
        for f in as_completed(futures):
            out[futures[f]] = f.result()  # (data or None, source_text_kind, stats)
            on_done(len(out))
    return out


def valid_review(d):
    for key, _ in REVIEW_SECTIONS:
        sec = d.get(key)
        if not isinstance(sec, dict) or not isinstance(sec.get("summary"), str):
            return False
        if not isinstance(sec.get("key_points"), list):
            return False
    return isinstance(d.get("overview"), str)


def write_review(topic, papers):
    """Synthesize the extractions into five sections, citing papers as [1]..[10]."""
    blocks = []
    for p in papers:
        facts = p.get("extraction") or {"abstract": p["abstract"]}
        blocks.append(f"[{p['rank']}] {p['title']} ({p['year']})\n{json.dumps(facts)}")
    keys = ", ".join(f'"{k}"' for k, _ in REVIEW_SECTIONS)
    user = f"""Topic: {topic}

Below are {len(papers)} papers, each with structured facts extracted from it.

{chr(10).join(blocks)}

Write a short systematic review of these papers. Return JSON only, with keys "overview" and {keys}.
- "overview": one sentence on what this body of work shows as a whole.
- Each section key maps to {{"summary": "<3-5 plain sentences>", "key_points": ["<2-4 short points>"]}}.
- Cite papers by their number in square brackets, e.g. [3] or [2, 7]. Every claim needs a citation.
- Compare across papers (agreement, disagreement, gaps). Do not just list them one by one.
- limitations: combine what authors stated with gaps you see across the set, and say which is which.
- If the facts don't support something, say so plainly instead of guessing."""
    system = ("You write concise, evidence-based systematic reviews for busy researchers. "
              "You only state what the provided facts support.")
    data, stats = llm.ask_json(system, user, config.LLM_MAX_TOKENS_REVIEW, valid_review,
                               model=config.REVIEW_MODEL)
    if not data:
        raise PipelineError("The review could not be written (the model returned invalid "
                            "JSON twice). Try the search again.")
    review = {"overview": data["overview"], "sections": [
        {"key": key, "title": title, "summary": data[key]["summary"],
         "key_points": [str(x) for x in data[key]["key_points"]]}
        for key, title in REVIEW_SECTIONS]}
    return review, stats


_embedder = None


def embed(texts):
    """Bi-encoder embeddings (one vector per paper), so similar topics sit close together.

    Falls back to TF-IDF + SVD when the embedding model can't be loaded (e.g. offline).
    """
    global _embedder
    try:
        if _embedder is None:
            from sentence_transformers import SentenceTransformer
            _embedder = SentenceTransformer(config.EMBED_MODEL, device="cpu")
        return np.asarray(_embedder.encode(texts, normalize_embeddings=True,
                                           show_progress_bar=False))
    except Exception:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        tfidf = TfidfVectorizer(stop_words="english", max_features=5000).fit_transform(texts)
        dims = max(2, min(50, tfidf.shape[1] - 1, len(texts) - 1))
        vecs = TruncatedSVD(dims, random_state=config.RANDOM_SEED).fit_transform(tfidf)
        return vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)


def choose_clusters(vectors):
    """KMeans for each k in CLUSTER_K_RANGE; keep the k with the best silhouette score."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    lo, hi = config.CLUSTER_K_RANGE
    best = (None, -2.0)
    for k in range(lo, min(hi, len(vectors) - 1) + 1):
        labels = KMeans(k, n_init=10, random_state=config.RANDOM_SEED).fit_predict(vectors)
        s = silhouette_score(vectors, labels)
        if s > best[1]:
            best = (labels, s)
    return best[0] if best[0] is not None else np.zeros(len(vectors), dtype=int)


def keyword_names(texts, labels):
    """Fallback cluster names: the top TF-IDF words that set each cluster apart."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(stop_words="english", max_features=3000, ngram_range=(1, 2))
    m = vec.fit_transform(texts)
    words = np.array(vec.get_feature_names_out())
    names = []
    for c in range(labels.max() + 1):
        weights = np.asarray(m[labels == c].mean(axis=0)).ravel()
        names.append(" / ".join(words[np.argsort(-weights)[:2]]).title())
    return names


def name_clusters(topic, titles_by_cluster, texts, labels):
    listing = "\n\n".join(f"Cluster {i + 1}:\n" + "\n".join(f"- {t}" for t in titles[:8])
                          for i, titles in enumerate(titles_by_cluster))
    user = (f"Topic: {topic}\n\nPapers were grouped by meaning:\n\n{listing}\n\n"
            f'Give each cluster a 2-4 word name that tells them apart. Return JSON only: '
            f'{{"names": [<{len(titles_by_cluster)} strings, in order>]}}')
    data, stats = llm.ask_json(
        "You name groups of research papers.", user, 300,
        lambda d: isinstance(d.get("names"), list) and len(d["names"]) == len(titles_by_cluster))
    if data:
        return [str(n) for n in data["names"]], stats
    return keyword_names(texts, labels), stats


def topic_clusters(topic, candidates, top_ids):
    """Cluster every candidate by meaning; chart = cumulative papers per cluster by year."""
    texts = [score.paper_text(p) for p in candidates]
    labels = choose_clusters(embed(texts))
    k = int(labels.max()) + 1
    titles = [[p["title"] for p, c in zip(candidates, labels) if c == i] for i in range(k)]
    names, stats = name_clusters(topic, titles, texts, labels)

    years = [p["year"] for p in candidates if p["year"]]
    span = list(range(min(years), max(years) + 1)) if years else []
    series = []
    for i in range(k):
        members = [p for p, c in zip(candidates, labels) if c == i]
        per_year = [sum(1 for p in members if p["year"] == y) for y in span]
        series.append({
            "cluster": i, "name": names[i], "size": len(members),
            "points": [{"year": y, "count": int(c)}
                       for y, c in zip(span, np.cumsum(per_year).tolist())],
            "top_papers": [p["paper_id"] for p in members if p["paper_id"] in top_ids],
        })
    series.sort(key=lambda s: -s["size"])  # biggest topic gets the first color
    order = {s["cluster"]: rank for rank, s in enumerate(series)}
    for s in series:
        s["cluster"] = order[s["cluster"]]
    by_paper = {p["paper_id"]: order[int(c)] for p, c in zip(candidates, labels)}
    return {"years": span, "series": series}, by_paper, stats


# ---------- the whole run ----------

def paper_card(row):
    return {
        "paper_id": row["paper_id"], "title": row["title"], "abstract": row["abstract"],
        "year": row["year"], "venue": row["venue"], "citations": row["citation_count"],
        "authors": json.loads(row["authors_json"] or "[]"),
        "pdf_url": row["pdf_url"], "doi": row["doi"],
        "url": f"https://www.semanticscholar.org/paper/{row['paper_id']}",
    }


def run(query, report=lambda stage, fraction, message: None):
    query = re.sub(r"\s+", " ", query or "").strip()
    if len(query) < 3:
        raise PipelineError("Type a topic or a few keywords (at least 3 characters).")
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise PipelineError("ANTHROPIC_API_KEY is not set. Add it to the .env file and "
                            "restart the app.")
    cost = 0.0

    # 1) Processing your query
    report(0, 0.1, "Reading your query")
    topic, criteria, stats = understand_query(query)
    cost += stats["cost_usd"]
    report(0, 1.0, f"Searching for: {topic}")

    # 2) Identifying candidate papers
    report(1, 0.1, "Searching Semantic Scholar")
    try:
        raw = fetch.search_papers(query, config.WEB_CANDIDATES)
    except Exception as e:
        raise PipelineError(f"Could not reach Semantic Scholar ({e}). Check your connection "
                            "and try again in a minute.") from e
    if len(raw) < config.WEB_TOP_N:
        raise PipelineError(f"Only {len(raw)} papers with abstracts matched. Try broader "
                            "keywords.")
    conn = db.connect()
    for p in raw:
        db.upsert_paper(conn, p, query)
    conn.commit()
    ids = {p["paperId"] for p in raw}
    candidates = [paper_card(r) for r in db.papers_for_topic(conn, query)
                  if r["paper_id"] in ids]
    report(1, 1.0, f"Found {len(candidates)} candidate papers")

    # 3) Scoring the papers
    q = score.query_text(topic, criteria)
    n = len(candidates)
    for i, scorer in enumerate(("minilm", "bge")):
        report(2, 0.05 + 0.2 * i, f"Ranking with {scorer.upper()} (local model)")
        results = score.cross_encode(
            scorer, q, candidates,
            on_batch=lambda done, i=i: report(2, 0.05 + 0.2 * i + 0.2 * done / n,
                                              f"{scorer.upper()}: {done}/{n} papers"))
        for p, (s, ms) in zip(candidates, results):
            p.setdefault("scores", {})[scorer] = s
            db.save_score(conn, p["paper_id"], query, scorer, s, latency_ms=ms, cost_usd=0.0)
    conn.commit()
    for p in candidates:
        p["ce_mean"] = (p["scores"]["minilm"] + p["scores"]["bge"]) / 2
        p["scores"]["llm"] = None
        p["rationale"] = None

    shortlist = sorted(candidates, key=lambda p: -p["ce_mean"])[:config.WEB_LLM_SHORTLIST]
    m = len(shortlist)
    report(2, 0.45, f"LLM judge reading the top {m} abstracts")
    judged = judge_many(topic, criteria, shortlist,
                        lambda done: report(2, 0.45 + 0.55 * done / m,
                                            f"LLM judge: {done}/{m} papers"))
    for p in shortlist:
        s, rationale, st = judged[p["paper_id"]]
        p["scores"]["llm"], p["rationale"] = s, rationale
        cost += st["cost_usd"]
        db.save_score(conn, p["paper_id"], query, "llm", s, rationale, **st)
    conn.commit()

    # 4) Finishing scoring
    ranked = sorted(candidates, key=rank_key, reverse=True)
    top = ranked[:config.WEB_TOP_N]
    for rank, p in enumerate(top, start=1):
        p["rank"] = rank
        p["level"] = None if p["scores"]["llm"] is None else round(p["scores"]["llm"] * 4) + 1
    report(3, 0.05, f"Extracting study details from the top {len(top)} papers")
    extracted = extract_many(top, lambda done: report(
        3, 0.05 + 0.45 * done / len(top), f"Extracted {done}/{len(top)} papers"))
    for p in top:
        data, kind, st = extracted[p["paper_id"]]
        cost += st["cost_usd"]
        p["extraction"], p["source_text_kind"] = data, kind
        if data:
            db.save_extraction(conn, p["paper_id"], query, data, kind)
    conn.commit()

    report(3, 0.55, "Writing the systematic review")
    review, st = write_review(topic, top)
    cost += st["cost_usd"]

    report(3, 0.85, "Grouping papers into topics")
    clusters, cluster_of, st = topic_clusters(topic, candidates, {p["paper_id"] for p in top})
    cost += st["cost_usd"]
    for p in top:
        p["cluster"] = cluster_of[p["paper_id"]]

    levels = [p["scores"]["llm"] for p in shortlist if p["scores"]["llm"] is not None]
    counts = {lv: sum(1 for s in levels if round(s * 4) + 1 == lv) for lv, _, _ in RUBRIC}
    result = {
        "query": query, "topic": topic, "criteria": criteria,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_candidates": n, "n_shortlisted": m, "cost_usd": round(cost, 4),
        "models": {"minilm": config.CROSS_ENCODERS["minilm"],
                   "bge": config.CROSS_ENCODERS["bge"], "llm": config.LLM_MODEL},
        "papers": [{k: v for k, v in p.items() if k != "ce_mean"} for p in top],
        "review": review,
        "clusters": clusters,
        "rubric": [{"level": lv, "label": label, "description": desc, "count": counts[lv]}
                   for lv, label, desc in RUBRIC],
    }
    save_run(result)
    report(3, 1.0, "Done")
    return result


def save_run(result):
    """Keep every result as JSON so a search can be reopened without re-running it."""
    config.RUNS_DIR.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", result["query"].lower()).strip("-")[:50] or "search"
    stamp = result["created_at"].replace(":", "").replace("-", "")[:15]
    path = config.RUNS_DIR / f"{stamp}-{slug}.json"
    path.write_text(json.dumps(result, indent=1), encoding="utf-8")
    return path
