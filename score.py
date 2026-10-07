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


def query_text(topic, criteria):
    """Every scorer sees the same query, so the comparison is fair."""
    return f"{topic}. {criteria}"


def paper_text(p):
    return f"{p['title']}. {p['abstract']}"


def sigmoid(x):
    """Squash raw logits into (0, 1) so cross-encoder scores share a scale with the LLM."""
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def score_cross_encoder(conn, topic, criteria, scorer, papers, batch_size=32):
    """Cross-encoder: reads query and paper together, outputs one relevance logit."""
    from sentence_transformers import CrossEncoder  # heavy import, only when needed

    model = CrossEncoder(config.CROSS_ENCODERS[scorer], device="cpu")
    query = query_text(topic, criteria)
    for i in tqdm(range(0, len(papers), batch_size), desc=scorer):
        batch = papers[i:i + batch_size]
        pairs = [(query, paper_text(p)) for p in batch]
        start = time.perf_counter()
        # Identity activation = raw logits, so we always apply our own sigmoid once.
        logits = model.predict(pairs, activation_fn=_identity(), show_progress_bar=False)
        per_paper_ms = (time.perf_counter() - start) * 1000 / len(batch)
        for p, s in zip(batch, sigmoid(logits)):
            db.save_score(conn, p["paper_id"], topic, scorer, float(s),
                          latency_ms=per_paper_ms, cost_usd=0.0)
        conn.commit()


def _identity():
    import torch
    return torch.nn.Identity()


def valid_llm_score(data):
    return isinstance(data.get("score"), int) and 1 <= data["score"] <= 5


def score_llm(conn, topic, criteria, papers):
    """LLM judge: rubric score 1-5, normalized to [0, 1] as (score - 1) / 4."""
    for p in tqdm(papers, desc="llm"):
        prompt = LLM_PROMPT.format(topic=topic, criteria=criteria,
                                   title=p["title"], abstract=p["abstract"])
        data, stats = llm.ask_json(LLM_SYSTEM, prompt, config.LLM_MAX_TOKENS_SCORE,
                                   valid_llm_score)
        score = (data["score"] - 1) / 4 if data else None   # NULL after two bad replies
        rationale = data.get("rationale") if data else None
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
