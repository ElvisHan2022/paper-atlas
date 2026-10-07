"""Milestone 2a: score every paper for relevance with up to three scorers.

    python score.py --topic "..." --scorers minilm,bge,llm [--criteria "..."]

Scores already in the database are skipped, so an interrupted run resumes for free.
"""
import argparse
import time

import numpy as np
from tqdm import tqdm

import config
import db
import llm

LLM_SYSTEM = (
    "You are screening research papers for a literature review. "
    "Judge only from the title and abstract you are given."
)

LLM_PROMPT = """Topic: {topic}

Relevance criteria: {criteria}

Rubric:
1 = unrelated
2 = tangential
3 = related background
4 = relevant
5 = core

Paper title: {title}
Abstract: {abstract}

Reply with JSON only: {{"score": <integer 1-5>, "rationale": "<one sentence>"}}"""

# Used only by the web app's evidence lens. The plain prompt above stays unchanged so
# evaluate.py results remain comparable.
LENS_PROMPT = """Topic: {topic}

Relevance criteria: {criteria}

Rubric for "score" (relevance to the topic):
1 = unrelated
2 = tangential
3 = related background
4 = relevant
5 = core

Evidence lens: {lens_label}. {lens_criteria}
Rubric for "lens_score" (how well the paper's evidence fits this lens, judged only from what
the title, abstract, venue and publication types show):
1 = does not fit
2 = weak fit
3 = partial fit
4 = good fit
5 = exemplary fit

Paper title: {title}
Venue: {venue} ({year})
Publication types: {types}
Abstract: {abstract}

Reply with JSON only: {{"score": <integer 1-5>, "lens_score": <integer 1-5>, "rationale": "<one sentence covering both>"}}"""


def query_text(topic, criteria):
    """Every scorer sees the same query, so the comparison is fair."""
    return f"{topic}. {criteria}"


def paper_text(p):
    return f"{p['title']}. {p['abstract']}"


def sigmoid(x):
    """Squash raw logits into (0, 1) so cross-encoder scores share a scale with the LLM."""
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


_models = {}


def load_cross_encoder(scorer):
    """Load each model once per process; the web app reuses it across searches."""
    if scorer not in _models:
        from sentence_transformers import CrossEncoder  # heavy import, only when needed
        _models[scorer] = CrossEncoder(config.CROSS_ENCODERS[scorer], device="cpu")
    return _models[scorer]


def cross_encode(scorer, query, papers, batch_size=32, on_batch=None):
    """Cross-encoder: reads query and paper together, outputs one relevance logit.

    Returns [(score in (0, 1), latency_ms per paper), ...] in the same order as papers.
    `on_batch(n_done)` lets a caller show progress.
    """
    model = load_cross_encoder(scorer)
    out = []
    for i in range(0, len(papers), batch_size):
        batch = papers[i:i + batch_size]
        pairs = [(query, paper_text(p)) for p in batch]
        start = time.perf_counter()
        # Identity activation = raw logits, so we always apply our own sigmoid once.
        logits = model.predict(pairs, activation_fn=_identity(), show_progress_bar=False)
        per_paper_ms = (time.perf_counter() - start) * 1000 / len(batch)
        out += [(float(s), per_paper_ms) for s in sigmoid(logits)]
        if on_batch:
            on_batch(len(out))
    return out


def score_cross_encoder(conn, topic, criteria, scorer, papers):
    results = cross_encode(scorer, query_text(topic, criteria), papers)
    for p, (s, ms) in tqdm(zip(papers, results), total=len(papers), desc=scorer):
        db.save_score(conn, p["paper_id"], topic, scorer, s, latency_ms=ms, cost_usd=0.0)
    conn.commit()


def _identity():
    import torch
    return torch.nn.Identity()


def valid_llm_score(data):
    return isinstance(data.get("score"), int) and 1 <= data["score"] <= 5


def valid_lens_score(data):
    return valid_llm_score(data) and isinstance(data.get("lens_score"), int) \
        and 1 <= data["lens_score"] <= 5


def judge(topic, criteria, p, lens=None):
    """LLM judge for one paper: (score, rationale, stats, lens_score).

    Rubric scores 1-5 are normalized as (score - 1) / 4. Scores are None after two bad
    replies; lens_score is None unless an evidence lens (a config.LENSES entry) is given.
    """
    if lens and lens.get("criteria"):
        prompt = LENS_PROMPT.format(
            topic=topic, criteria=criteria, lens_label=lens["label"],
            lens_criteria=lens["criteria"], title=p["title"], abstract=p["abstract"],
            venue=p.get("venue") or "unknown venue", year=p.get("year") or "year unknown",
            types=", ".join(p.get("publicationTypes") or []) or "not listed")
        validate = valid_lens_score
    else:
        prompt = LLM_PROMPT.format(topic=topic, criteria=criteria,
                                   title=p["title"], abstract=p["abstract"])
        validate = valid_llm_score
    data, stats = llm.ask_json(LLM_SYSTEM, prompt, config.LLM_MAX_TOKENS_SCORE, validate)
    if not data:
        return None, None, stats, None
    lens_score = (data["lens_score"] - 1) / 4 if "lens_score" in data and lens else None
    return (data["score"] - 1) / 4, data.get("rationale"), stats, lens_score


def score_llm(conn, topic, criteria, papers):
    for p in tqdm(papers, desc="llm"):
        score, rationale, stats, _ = judge(topic, criteria, p)
        db.save_score(conn, p["paper_id"], topic, "llm", score, rationale, **stats)
        conn.commit()


def main(topic, scorers, criteria, rescore=False):
    conn = db.connect()
    papers = db.papers_for_topic(conn, topic)
    if not papers:
        raise SystemExit(f"No papers for topic {topic!r}. Run fetch.py first.")

    for scorer in scorers:
        done = {pid for pid, s in db.scores_for_topic(conn, topic, scorer).items()
                if s is not None}
        todo = papers if rescore else [p for p in papers if p["paper_id"] not in done]
        if scorer == "llm":
            score_llm(conn, topic, criteria, todo)
        else:
            score_cross_encoder(conn, topic, criteria, scorer, todo)

    # Acceptance: each scorer covers >= 95% of papers.
    all_ok = True
    for scorer in scorers:
        scored = [s for s in db.scores_for_topic(conn, topic, scorer).values() if s is not None]
        coverage = len(scored) / len(papers)
        all_ok &= coverage >= 0.95
        print(f"[score] {scorer}: {len(scored)}/{len(papers)} scored ({coverage:.0%})")
    print(f"[score] Acceptance (>= 95% each): {'PASS' if all_ok else 'FAIL'}")
    return all_ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--topic", required=True)
    ap.add_argument("--scorers", default="minilm,bge,llm")
    ap.add_argument("--criteria", default=config.DEFAULT_CRITERIA)
    ap.add_argument("--rescore", action="store_true", help="ignore existing scores")
    args = ap.parse_args()
    main(args.topic, args.scorers.split(","), args.criteria, args.rescore)
