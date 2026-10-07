"""One search, start to finish, for the web app: query -> candidates -> scores -> review.

app.py runs `run(query, report)` in a background thread. `report(stage, fraction, message)`
is how this file tells the browser how far along it is.
"""
import json
import sqlite3
import threading
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import numpy as np

import config
import db
import extract
import llm
import score
import sources

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

LENS_RUBRIC = [(5, "Exemplary fit"), (4, "Good fit"), (3, "Partial fit"), (2, "Weak fit"),
               (1, "Does not fit")]

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
        lambda d: isinstance(d.get("topic"), str) and isinstance(d.get("criteria"), str),
        cache=True)
    if not data:  # keep going with a plain fallback rather than failing the search
        return query, f"Papers whose main contribution addresses: {query}.", stats
    return data["topic"].strip() or query, data["criteria"].strip(), stats


# ---------- stage 3: scoring ----------

def judge_many(topic, criteria, papers, on_done, lens=None):
    """LLM-judge papers in parallel. Returns {paper_id: (score, rationale, stats, lens_score)}."""
    llm.client()  # create the shared client once, before the threads start
    out = {}
    with ThreadPoolExecutor(max_workers=config.WEB_WORKERS) as pool:
        futures = {pool.submit(score.judge, topic, criteria, p, lens, True): p["paper_id"]
                   for p in papers}
        for f in as_completed(futures):
            out[futures[f]] = f.result()
            on_done(len(out))
    return out


def final_llm_score(relevance, lens_score):
    """Relevance, blended with the evidence-lens score when a lens is active."""
    if relevance is None:
        return None
    if lens_score is None:
        return relevance
    return (1 - config.LENS_WEIGHT) * relevance + config.LENS_WEIGHT * lens_score


def rank_key(p):
    """LLM score first; the cross-encoder average breaks ties and ranks unjudged papers."""
    llm_score = p.get("final", p["scores"]["llm"])
    return (llm_score if llm_score is not None else -1.0, p["ce_mean"])


# ---------- stage 4: review + clusters ----------

def extract_many(papers, on_done):
    out = {}
    with ThreadPoolExecutor(max_workers=config.WEB_WORKERS) as pool:
        futures = {pool.submit(extract.extract_one, p, True): p["paper_id"] for p in papers}
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


def write_review(topic, papers, lens=None):
    """Synthesize the extractions into five sections, citing papers as [1]..[10]."""
    blocks = []
    for p in papers:
        facts = p.get("extraction") or {"abstract": p["abstract"]}
        kind = "; ".join([p.get("venue_type") or "", *(p.get("evidence") or [])]).strip("; ")
        blocks.append(f"[{p['rank']}] {p['title']} ({p['year']}, {p.get('venue') or 'venue unknown'}"
                      f"{' - ' + kind if kind else ''})\n{json.dumps(facts)}")
    keys = ", ".join(f'"{k}"' for k, _ in REVIEW_SECTIONS)
    focus = (f"\nThe reader is weighing evidence through a '{lens['label']}' lens: "
             f"{lens['criteria']} Say in the discussion how well this set meets that bar.\n"
             if lens and lens.get("criteria") else "")
    user = f"""Topic: {topic}
{focus}
Below are {len(papers)} papers, each with structured facts extracted from it.

{chr(10).join(blocks)}

Write a short systematic review of these papers. Return JSON only, with keys "overview" and {keys}.
- "overview": one sentence on what this body of work shows as a whole.
- Each section key maps to {{"summary": "<3-5 plain sentences>", "key_points": ["<2-4 short points>"]}}.
- Cite papers by their number in square brackets, e.g. [3] or [2, 7]. Every claim needs a citation.
- Compare across papers (agreement, disagreement, gaps). Do not just list them one by one.
- limitations: combine what authors stated with gaps you see across the set, and say which is which.
- Note the strength of evidence (e.g. RCT vs retrospective vs preprint) where it matters.
- If the facts don't support something, say so plainly instead of guessing."""
    system = ("You write concise, evidence-based systematic reviews for busy researchers. "
              "You only state what the provided facts support.")
    data, stats = llm.ask_json(system, user, config.LLM_MAX_TOKENS_REVIEW, valid_review,
                               model=config.REVIEW_MODEL, cache=True)
    if not data:
        raise PipelineError("The review could not be written (the model returned invalid "
                            "JSON twice). Try the search again.")
    review = {"overview": data["overview"], "sections": [
        {"key": key, "title": title, "summary": data[key]["summary"],
         "key_points": [str(x) for x in data[key]["key_points"]]}
        for key, title in REVIEW_SECTIONS]}
    return review, stats


# ---------- independent verification of the review ----------

VERIFY_SYSTEM = (
    "You are an independent fact-checker for systematic reviews. You did not write the "
    "review you are checking, and you have no stake in it being right. You judge claims only "
    "against the source excerpts you are given, never against outside knowledge.")

VERIFY_PROMPT = """Section of a systematic review: {title}

{text}

Sources for the papers this section cites (abstracts and evidence quotes taken directly from
the papers):

{sources}

Audit every factual claim in the section that cites a paper. For each claim, check it against
the cited papers' sources above and nothing else. A claim is:
- "supported" if the cited sources clearly state it,
- "partial" if they support part of it or a weaker version,
- "unsupported" if they don't say it, contradict it, or the citation points to the wrong paper.
Report outcomes faithfully: do not round "partial" up to "supported". If a cited number has no
source above, the claim is unsupported.

Return JSON only:
{{"claims": [{{"claim": "<the claim, at most 25 words>", "cites": [<paper numbers>],
  "verdict": "supported" | "partial" | "unsupported", "reason": "<one sentence pointing to the source text>"}}]}}"""

VERDICTS = ("supported", "partial", "unsupported")


def cited_numbers(text):
    """Paper numbers cited as [3] or [2, 7] in a piece of review text."""
    found = set()
    for group in re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", text):
        found.update(int(n) for n in re.findall(r"\d+", group))
    return sorted(found)


def valid_check(d):
    claims = d.get("claims")
    return isinstance(claims, list) and all(
        isinstance(c, dict) and c.get("verdict") in VERDICTS and isinstance(c.get("claim"), str)
        for c in claims)


def source_excerpt(p):
    """What the verifier sees for one paper: the raw abstract plus the extraction's quotes."""
    quotes = (p.get("extraction") or {}).get("evidence") or {}
    lines = [f"[{p['rank']}] {p['title']} ({p.get('year')}, {p.get('venue') or 'venue unknown'})",
             f"Abstract: {p.get('abstract') or 'not available'}"]
    lines += [f"Evidence quote ({k}): {v}" for k, v in quotes.items() if v and v != "not found"]
    return "\n".join(lines)


def verify_section(section, papers_by_rank):
    """Check one section's cited claims in a fresh context. Returns (check, stats)."""
    text = section["summary"] + "\n" + "\n".join(f"- {k}" for k in section["key_points"])
    cites = [n for n in cited_numbers(text) if n in papers_by_rank]
    if not cites:
        return {"claims": [], "note": "No citations to check."}, {"cost_usd": 0.0}
    user = VERIFY_PROMPT.format(title=section["title"], text=text,
                                sources="\n\n".join(source_excerpt(papers_by_rank[n]) for n in cites))
    data, stats = llm.ask_json(VERIFY_SYSTEM, user, 2000, valid_check,
                               model=config.VERIFY_MODEL, cache=True)
    if not data:
        return {"claims": [], "note": "The checker could not complete this section."}, stats
    return {"claims": [{"claim": c["claim"], "cites": [n for n in c.get("cites") or [] if isinstance(n, int)],
                        "verdict": c["verdict"], "reason": str(c.get("reason") or "")}
                       for c in data["claims"]]}, stats


def verify_review(review, papers):
    """Fan out: the five sections are independent, so they're checked in parallel."""
    by_rank = {p["rank"]: p for p in papers}
    cost = 0.0
    with ThreadPoolExecutor(max_workers=len(review["sections"])) as pool:
        futures = [pool.submit(verify_section, sec, by_rank) for sec in review["sections"]]
        for sec, future in zip(review["sections"], futures):
            check, stats = future.result()
            sec["check"] = check
            cost += stats.get("cost_usd", 0.0)
    claims = [c for sec in review["sections"] for c in sec["check"]["claims"]]
    review["verification"] = {v: sum(1 for c in claims if c["verdict"] == v) for v in VERDICTS}
    review["verification"].update(total=len(claims), model=config.VERIFY_MODEL)
    return review, cost


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


def keyword_names(texts, labels, skip=()):
    """Fallback cluster names: the two words that most set each cluster apart.

    Words in `skip` (usually the search terms, which every cluster shares) are left out.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(stop_words="english", max_features=3000, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]{2,}\b")
    m = vec.fit_transform(texts)
    words = vec.get_feature_names_out()
    skip = {w.lower() for w in skip} | {"study", "studies", "paper", "results", "using", "based"}
    names = []
    for c in range(labels.max() + 1):
        weights = np.asarray(m[labels == c].mean(axis=0)).ravel()
        picked = [words[i] for i in np.argsort(-weights) if words[i] not in skip][:2]
        names.append(" & ".join(w.capitalize() for w in picked) or f"Topic {c + 1}")
    return names


def name_clusters(topic, titles_by_cluster, texts, labels):
    listing = "\n\n".join(f"Cluster {i + 1}:\n" + "\n".join(f"- {t}" for t in titles[:8])
                          for i, titles in enumerate(titles_by_cluster))
    user = (f"Topic: {topic}\n\nPapers were grouped by meaning:\n\n{listing}\n\n"
            f'Give each cluster a 2-4 word name that tells them apart. Return JSON only: '
            f'{{"names": [<{len(titles_by_cluster)} strings, in order>]}}')
    data, stats = llm.ask_json(
        "You name groups of research papers.", user, 300,
        lambda d: isinstance(d.get("names"), list) and len(d["names"]) == len(titles_by_cluster),
        cache=True)
    if data:
        return [str(n) for n in data["names"]], stats
    return keyword_names(texts, labels, re.findall(r"[a-z]+", topic.lower())), stats


def topic_clusters(topic, candidates, top_ids, use_llm=True):
    """Cluster every candidate by meaning; chart = cumulative papers per cluster by year."""
    texts = [score.paper_text(p) for p in candidates]
    labels = choose_clusters(embed(texts))
    k = int(labels.max()) + 1
    titles = [[p["title"] for p, c in zip(candidates, labels) if c == i] for i in range(k)]
    if use_llm:
        names, stats = name_clusters(topic, titles, texts, labels)
    else:
        names = keyword_names(texts, labels, re.findall(r"[a-z]+", topic.lower()))
        stats = {"cost_usd": 0.0}

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

def paper_card(p):
    """The fields the UI needs, from a merged search result (see sources.py)."""
    ext = p.get("externalIds") or {}
    types = p.get("publicationTypes") or []
    return {
        "paper_id": p["paperId"], "title": p["title"], "abstract": p["abstract"],
        "year": p.get("year"), "venue": p.get("venue") or "",
        "citations": p.get("citationCount"),
        "authors": [a.get("name") for a in (p.get("authors") or []) if a.get("name")],
        "pdf_url": (p.get("openAccessPdf") or {}).get("url"), "doi": ext.get("DOI"),
        "url": p.get("url") or f"https://www.semanticscholar.org/paper/{p['paperId']}",
        "sources": p.get("sources") or [p.get("source", "semantic_scholar")],
        "venue_type": sources.venue_type(p.get("venue"), p.get("sources") or [], types),
        "evidence": sources.evidence_tags(types),
        "publicationTypes": types,
    }


def api_error_message(e):
    """Plain-English version of an Anthropic API error."""
    status = getattr(e, "status_code", None)
    text = str(getattr(e, "message", e))
    if status == 401:
        return ("Your Anthropic API key was rejected. Check the ANTHROPIC_API_KEY line in .env "
                "(no quotes or spaces), then restart app.py.")
    if status == 400 and "credit" in text.lower():
        return "Your Anthropic account is out of credit. Add credit at console.anthropic.com."
    if status == 429:
        return "The Anthropic API is rate-limiting this key. Wait a minute and try again."
    return f"The Anthropic API returned an error ({status}): {text}"


def run(query, report=lambda stage, fraction, message: None, source_names=None,
        lens_key="balanced"):
    try:
        return _run(query, report, source_names, lens_key)
    except Exception as e:
        log = getattr(_current, "log", None)
        if log:  # the step log records how far the search got and why it stopped
            log("error", kind=type(e).__name__, message=str(e)[:500])
        raise_explained(e)
    finally:
        _current.log = None


def raise_explained(e):
    """Re-raise an exception as something the person searching can act on."""
    if isinstance(e, PipelineError):
        raise e
    if isinstance(e, sqlite3.OperationalError) and "locked" in str(e):
        raise PipelineError(
            f"The results database ({config.DB_PATH}) is busy. Close any program that has it "
            "open (such as DB Browser for SQLite), wait for other searches to finish, and "
            "try again.") from e
    if type(e).__module__.startswith("anthropic") and hasattr(e, "status_code"):
        raise PipelineError(api_error_message(e)) from e
    raise e


_current = threading.local()   # the running search's step logger, for error reporting


def make_run_log(run_id):
    """Append-only step log: one JSON line per step, written as it happens.

    If a search crashes, the log shows exactly how far it got and why. Together with the
    HTTP and LLM caches, re-running the search replays the finished steps for free.
    """
    config.RUNS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.RUNS_DIR / f"{run_id}.log.jsonl"
    events, start = [], time.perf_counter()

    def log(step, **detail):
        event = {"t": round(time.perf_counter() - start, 2), "step": step, **detail}
        events.append(event)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")
    return log, events


def run_id_for(query):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"{stamp}-{re.sub(r'[^a-z0-9]+', '-', query.lower()).strip('-')[:50] or 'search'}"


def _run(query, report, source_names, lens_key):
    query = re.sub(r"\s+", " ", query or "").strip()
    if len(query) < 3:
        raise PipelineError("Type a topic or a few keywords (at least 3 characters).")
    source_names = [s for s in (source_names or list(config.SOURCES)) if s in config.SOURCES]
    if not source_names:
        raise PipelineError("Pick at least one source to search.")
    lens = config.LENSES.get(lens_key) or config.LENSES["balanced"]
    has_llm = llm.has_key()
    cost = 0.0
    run_id = run_id_for(query)
    log, events = make_run_log(run_id)
    _current.log = log
    log("start", query=query, lens=lens["label"], sources=source_names, llm=has_llm)

    # 1) Processing your query
    report(0, 0.1, "Reading your query")
    if has_llm:
        topic, criteria, stats = understand_query(query)
        cost += stats["cost_usd"]
    else:
        topic, criteria = query, f"Papers whose main contribution addresses: {query}."
    log("query", topic=topic, criteria=criteria)
    report(0, 1.0, f"Searching for: {topic}")

    # 2) Identifying candidate papers
    names = ", ".join(config.SOURCES[s] for s in source_names)
    report(1, 0.1, f"Searching {names}")
    raw, counts, errors = sources.search_all(query, source_names)
    log("search", counts=counts, errors=errors, merged=len(raw))
    if errors and not counts:
        detail = "; ".join(f"{config.SOURCES[s]}: {e}" for s, e in errors.items())
        raise PipelineError(f"Could not reach any source ({detail}). Check your connection and "
                            "try again in a minute.")
    if len(raw) < config.WEB_TOP_N:
        raise PipelineError(f"Only {len(raw)} papers with abstracts matched. Try broader "
                            "keywords or more sources.")
    conn = db.connect()
    for p in raw:
        db.upsert_paper(conn, p, query)
    conn.commit()
    candidates = [paper_card(p) for p in raw]
    n = len(candidates)
    merged = sum(counts.values()) - n
    report(1, 1.0, f"Found {n} candidate papers" + (f" ({merged} duplicates merged)" if merged > 0 else ""))

    # 3) Scoring the papers
    q = score.query_text(topic, criteria)
    span = 0.45 if has_llm else 0.95   # without the LLM judge, local models are the whole stage
    for i, scorer in enumerate(("minilm", "bge")):
        report(2, 0.05 + span / 2 * i, f"Ranking with {scorer.upper()} (local model)")
        results = score.cross_encode(
            scorer, q, candidates,
            on_batch=lambda done, i=i: report(2, 0.05 + span / 2 * (i + done / n),
                                              f"{scorer.upper()}: {done}/{n} papers"))
        for p, (s, ms) in zip(candidates, results):
            p.setdefault("scores", {})[scorer] = s
            db.save_score(conn, p["paper_id"], query, scorer, s, latency_ms=ms, cost_usd=0.0)
        conn.commit()  # commit before the next model runs, so the database is never held open
        log("rank", model=scorer, papers=n)
    for p in candidates:
        p["ce_mean"] = (p["scores"]["minilm"] + p["scores"]["bge"]) / 2
        p["scores"]["llm"] = p["scores"]["lens"] = None
        p["rationale"] = None

    shortlist = sorted(candidates, key=lambda p: -p["ce_mean"])[:config.WEB_LLM_SHORTLIST]
    m = len(shortlist) if has_llm else 0
    if has_llm:
        report(2, 0.5, f"LLM judge reading the top {m} abstracts")
        judged = judge_many(topic, criteria, shortlist,
                            lambda done: report(2, 0.5 + 0.5 * done / m,
                                                f"LLM judge: {done}/{m} papers"), lens)
        for p in shortlist:
            s, rationale, st, lens_score = judged[p["paper_id"]]
            p["scores"]["llm"], p["scores"]["lens"], p["rationale"] = s, lens_score, rationale
            p["final"] = final_llm_score(s, lens_score)
            cost += st["cost_usd"]
            db.save_score(conn, p["paper_id"], query, "llm", s, rationale,
                          **{k: v for k, v in st.items() if k != "cached"})
        conn.commit()
        log("judge", judged=m, failed=sum(1 for p in shortlist if p["scores"]["llm"] is None),
            reused_from_cache=sum(1 for v in judged.values() if v[2].get("cached")))

    # 4) Finishing scoring
    ranked = sorted(candidates, key=rank_key, reverse=True)
    top = ranked[:config.WEB_TOP_N]
    for rank, p in enumerate(top, start=1):
        p["rank"] = rank
        p["level"] = None if p["scores"]["llm"] is None else round(p["scores"]["llm"] * 4) + 1
        p["lens_level"] = None if p["scores"]["lens"] is None else round(p["scores"]["lens"] * 4) + 1
        p["extraction"], p["source_text_kind"] = None, "abstract_only"
    review = None
    if has_llm:
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
        log("extract", ok=sum(1 for p in top if p["extraction"]),
            full_text=sum(1 for p in top if p["source_text_kind"] == "full_text"))
        report(3, 0.55, "Writing the systematic review")
        review, st = write_review(topic, top, lens)
        cost += st["cost_usd"]
        log("review", sections=len(review["sections"]))
        report(3, 0.72, "Independent check of every cited claim")
        review, verify_cost = verify_review(review, top)
        cost += verify_cost
        log("verify", **{k: v for k, v in review["verification"].items() if k != "model"})

    report(3, 0.85, "Grouping papers into topics")
    clusters, cluster_of, st = topic_clusters(topic, candidates, {p["paper_id"] for p in top},
                                              use_llm=has_llm)
    cost += st["cost_usd"]
    for p in top:
        p["cluster"] = cluster_of[p["paper_id"]]
    log("clusters", topics=len(clusters["series"]))

    def level_counts(key):
        vals = [p["scores"][key] for p in shortlist if p["scores"][key] is not None]
        return {lv: sum(1 for v in vals if round(v * 4) + 1 == lv) for lv in range(1, 6)}

    rel_counts, lens_counts = level_counts("llm"), level_counts("lens")
    log("done", cost_usd=round(cost, 4))
    result = {
        "run_id": run_id, "log": events,
        "query": query, "topic": topic, "criteria": criteria,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_candidates": n, "n_shortlisted": m, "cost_usd": round(cost, 4),
        "has_llm": has_llm,
        "sources": {"searched": source_names, "counts": counts, "errors": errors,
                    "labels": config.SOURCES},
        "lens": {"key": lens_key if lens_key in config.LENSES else "balanced", **lens},
        "lens_weight": config.LENS_WEIGHT,
        "models": {"minilm": config.CROSS_ENCODERS["minilm"],
                   "bge": config.CROSS_ENCODERS["bge"], "llm": config.LLM_MODEL},
        "papers": [{k: v for k, v in p.items() if k not in ("ce_mean", "final")} | {"final": p.get("final")}
                   for p in top],
        "review": review,
        "clusters": clusters,
        "rubric": [{"level": lv, "label": label, "description": desc, "count": rel_counts[lv]}
                   for lv, label, desc in RUBRIC],
        "lens_rubric": ([{"level": lv, "label": label, "count": lens_counts[lv]}
                         for lv, label in LENS_RUBRIC] if lens.get("criteria") else None),
    }
    save_run(result)
    report(3, 1.0, "Done")
    return result


def save_run(result):
    """Keep every result as JSON so a search can be reopened without re-running it."""
    config.RUNS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.RUNS_DIR / f"{result['run_id']}.json"
    path.write_text(json.dumps(result, indent=1), encoding="utf-8")
    return path
