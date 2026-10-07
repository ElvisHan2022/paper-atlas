"""Milestone 2b: hand-label papers in the terminal. These labels are the ground truth.

    python label.py --topic "..." --k 150

Keys: y = relevant, n = not relevant, s = skip, q = quit (progress is saved after every key).
Rerun the same command to pick up where you stopped.
"""
import argparse
import random
import textwrap
from datetime import datetime, timezone

import config
import db


def stratified_order(paper_ids, llm_scores, k, seed=config.RANDOM_SEED):
    """Pick k papers with equal draws from each LLM rubric level, interleaved.

    Why: most search hits are easy negatives or easy positives. Equal draws per level
    put the hard middle cases in front of the labeler. Interleaving (level 1, 2, 3, 4, 5,
    1, 2, ...) keeps the sample balanced even if you quit early.
    Levels with too few papers give their unused slots to the others.
    """
    rng = random.Random(seed)
    buckets = {}
    for pid in sorted(paper_ids):
        s = llm_scores.get(pid)
        level = "none" if s is None else round(s * 4) + 1  # back to rubric 1-5
        buckets.setdefault(level, []).append(pid)
    for ids in buckets.values():
        rng.shuffle(ids)

    order = []
    levels = sorted(buckets, key=str)
    while len(order) < k and any(buckets[lv] for lv in levels):
        for lv in levels:
            if buckets[lv] and len(order) < k:
                order.append(buckets[lv].pop())
    return order


def show(p, i, k):
    print("\n" + "=" * 80)
    print(f"[{i}/{k}]  {p['title']}  ({p['year']})")
    print("-" * 80)
    print(textwrap.fill(p["abstract"] or "(no abstract)", width=80))
    print("=" * 80)


def main(topic, k):
    conn = db.connect()
    papers = {p["paper_id"]: p for p in db.papers_for_topic(conn, topic)}
    llm_scores = db.scores_for_topic(conn, topic, "llm")
    if not llm_scores:
        print("No LLM scores yet, so sampling is plain random. Run score.py first for strata.")

    order = stratified_order(list(papers), llm_scores, k)
    seen = {r["paper_id"] for r in conn.execute(
        "SELECT paper_id FROM labels WHERE topic=?", (topic,))}
    todo = [pid for pid in order if pid not in seen]
    print(f"{len(order) - len(todo)} of {len(order)} already done. "
          f"Criteria: {config.DEFAULT_CRITERIA}")

    for i, pid in enumerate(todo, start=len(order) - len(todo) + 1):
        show(papers[pid], i, len(order))
        answer = ""
        while answer not in ("y", "n", "s", "q"):
            answer = input("relevant? [y/n/s/q] ").strip().lower()
        if answer == "q":
            break
        relevant = {"y": 1, "n": 0, "s": None}[answer]   # skip is stored as NULL
        db.save_label(conn, pid, topic, relevant, datetime.now(timezone.utc).isoformat())
        conn.commit()
        seen.add(pid)

    labels = db.labels_for_topic(conn, topic)
    print(f"[label] {len(labels)} labels saved "
          f"({sum(labels.values())} relevant, {len(labels) - sum(labels.values())} not).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--topic", required=True)
    ap.add_argument("--k", type=int, default=150)
    args = ap.parse_args()
    main(args.topic, args.k)
