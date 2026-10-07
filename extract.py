"""Milestone 4: pull a fixed, evidence-anchored summary out of each relevant paper.

    python extract.py --topic "..." --top 30
"""
import argparse
import json
import random

import requests
from pypdf import PdfReader
from tqdm import tqdm

import config
import db
import llm

SCHEMA = {
    "study_design": {"type": "", "population": "", "sample_size": "", "data_source": "",
                     "validation": "internal | external | none | unclear"},
    "methods": {"approach": "", "baselines": "", "metrics": ""},
    "results": {"headline": "", "uncertainty_reported": True},
    "limitations_stated": [""],
    "limitations_inferred": [""],
    "discussion": "",
    "evidence": {"study_design": "", "methods": "", "results": "", "limitations_stated": ""},
}
VALIDATION_VALUES = {"internal", "external", "none", "unclear"}

SYSTEM = (
    "You extract structured facts from research papers for a systematic review. "
    "You never guess: if the text does not say something, you write \"not found\"."
)

PROMPT = """Read the paper text below and fill in this JSON exactly (same keys, no extras):

{schema}

Rules:
- study_design.validation must be one of: internal, external, none, unclear.
- results.uncertainty_reported is true only if the paper reports confidence intervals, standard deviations, p-values, or similar.
- limitations_stated: only limitations the authors themselves wrote.
- limitations_inferred: limitations YOU notice that the authors did not state. Keep these separate.
- evidence: for each key, give a section or table anchor plus a direct quote of at most 15 words, e.g. "Sec 3.2: 'we recruited 120 clinicians from two hospitals'". If you cannot point to the text, write "not found".
- Any field the text does not support: "not found" (or [] for lists). Never guess.
{note}
Paper title: {title}

Paper text:
<<<
{text}
>>>

Reply with the JSON object only."""


def validate_extraction(data):
    """True when `data` has exactly the schema's keys and sensible value types."""
    if not isinstance(data, dict) or set(data) != set(SCHEMA):
        return False
    for key, template in SCHEMA.items():
        value = data[key]
        if isinstance(template, dict):
            if not isinstance(value, dict) or set(value) != set(template):
                return False
        elif isinstance(template, list):
            if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
                return False
        elif not isinstance(value, str):
            return False
    if not isinstance(data["results"]["uncertainty_reported"], bool):
        return False
    if data["study_design"]["validation"] not in VALIDATION_VALUES:
        return False
    return all(isinstance(v, str) for v in data["evidence"].values())


def pdf_text(paper_id, url):
    """Download (once) and read the open-access PDF. Returns '' on any failure."""
    path = config.PDF_DIR / f"{paper_id}.pdf"
    try:
        if not path.exists():
            resp = requests.get(url, timeout=60, headers={"User-Agent": "paper-atlas/0.1"})
            if resp.status_code != 200 or not resp.content.startswith(b"%PDF"):
                return ""
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(resp.content)
        reader = PdfReader(path)
        chunks, total = [], 0
        for page in reader.pages:
            t = page.extract_text() or ""
            chunks.append(t)
            total += len(t)
            if total >= config.PDF_MAX_CHARS:
                break
        return "\n".join(chunks)[:config.PDF_MAX_CHARS]
    except Exception as e:  # broken PDFs are common; fall back to the abstract
        print(f"  PDF failed for {paper_id}: {e}")
        return ""


def source_text(p):
    """Full text when we can get a real PDF, otherwise the abstract (and say so)."""
    if p["pdf_url"]:
        text = pdf_text(p["paper_id"], p["pdf_url"])
        if len(text) > 2000:  # a few hundred chars usually means a cover page or a paywall
            return text, "full_text"
    return p["abstract"] or "", "abstract_only"


def pick_papers(conn, topic, top):
    """Labeled-relevant papers first, then by LLM score. Labeled-irrelevant are skipped."""
    labels = db.labels_for_topic(conn, topic)
    llm_scores = db.scores_for_topic(conn, topic, "llm")
    papers = [p for p in db.papers_for_topic(conn, topic) if labels.get(p["paper_id"]) != 0]
    papers.sort(key=lambda p: (labels.get(p["paper_id"], 0),
                               llm_scores.get(p["paper_id"]) or 0.0), reverse=True)
    return papers[:top]


def extract_one(p, cache=False):
    text, kind = source_text(p)
    note = ("\nNote: only the abstract is available, so most evidence will be "
            "'not found'.\n" if kind == "abstract_only" else "")
    prompt = PROMPT.format(schema=json.dumps(SCHEMA, indent=2), note=note,
                           title=p["title"], text=text)
    data, stats = llm.ask_json(SYSTEM, prompt, config.LLM_MAX_TOKENS_EXTRACT,
                               validate_extraction, cache=cache)
    return data, kind, stats


def main(topic, top, redo=False):
    conn = db.connect()
    done = db.extractions_for_topic(conn, topic)
    papers = pick_papers(conn, topic, top)
    total_cost = 0.0
    for p in tqdm(papers, desc="extract"):
        if p["paper_id"] in done and not redo:
            continue
        data, kind, stats = extract_one(p)
        total_cost += stats["cost_usd"]
        if data is None:
            print(f"  invalid JSON twice, skipped: {p['title'][:70]}")
            continue
        db.save_extraction(conn, p["paper_id"], topic, data, kind)
        conn.commit()

    done = db.extractions_for_topic(conn, topic)
    n_full = sum(1 for _, k in done.values() if k == "full_text")
    ok = "PASS" if len(done) >= top else f"FAIL (need {top})"
    print(f"[extract] {len(done)} extractions stored ({n_full} full text, "
          f"{len(done) - n_full} abstract only), ${total_cost:.3f} this run. Acceptance: {ok}")

    titles = {p["paper_id"]: p["title"] for p in db.papers_for_topic(conn, topic)}
    sample = random.Random(config.RANDOM_SEED).sample(sorted(done), min(3, len(done)))
    for pid in sample:
        data, kind = done[pid]
        print("\n" + "#" * 80 + f"\nSPOT-CHECK ({kind}): {titles.get(pid)}\n" + "#" * 80)
        print(json.dumps(data, indent=2))
    return len(done)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--topic", required=True)
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--redo", action="store_true", help="re-extract papers already done")
    args = ap.parse_args()
    main(args.topic, args.top, args.redo)
