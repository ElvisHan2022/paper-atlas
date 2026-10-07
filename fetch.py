"""Milestone 1: pull candidate papers and their citation links from Semantic Scholar.

    python fetch.py --topic "LLM evaluation for clinical text" --n 300
"""
import argparse
import hashlib
import json
import os
import time

import requests
from tqdm import tqdm

import config
import db

_last_request_at = 0.0


def cached_get_json(url, params=None):
    """GET a URL, caching the JSON body on disk so reruns never hit the API twice."""
    global _last_request_at
    full_url = requests.Request("GET", url, params=params).prepare().url
    key = hashlib.sha256(full_url.encode()).hexdigest()[:32]
    path = config.CACHE_DIR / "http" / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))

    api_key = os.getenv("S2_API_KEY")
    headers = {"x-api-key": api_key} if api_key else {}
    wait = config.S2_SECONDS_WITH_KEY if api_key else config.S2_SECONDS_PER_REQUEST

    for attempt in range(8):
        # Rate limit: only real network calls count, cache hits are free.
        sleep_for = wait - (time.time() - _last_request_at)
        if sleep_for > 0:
            time.sleep(sleep_for)
        _last_request_at = time.time()
        resp = requests.get(full_url, headers=headers, timeout=60)
        if resp.status_code == 429 or resp.status_code >= 500:
            time.sleep(2 ** attempt)  # back off: 1, 2, 4, 8... seconds
            continue
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        data = resp.json()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")
        return data
    raise RuntimeError(f"Gave up after repeated 429/5xx: {full_url}")


def search_papers(topic, n):
    """Page through /paper/search until we hold n papers that have an abstract.

    Many results lack abstracts, so we keep paging past n raw hits (the API stops at 1,000).
    """
    kept, offset = [], 0
    while len(kept) < n and offset < 1000:
        limit = min(config.S2_PAGE_SIZE, 1000 - offset)
        page = cached_get_json(
            f"{config.S2_BASE}/paper/search",
            {"query": topic, "fields": config.S2_FIELDS, "offset": offset, "limit": limit},
        )
        batch = (page or {}).get("data") or []
        if not batch:
            break
        kept += [p for p in batch if p.get("abstract") and p.get("paperId")]
        offset += len(batch)
        if page.get("next") is None:
            break
    return kept[:n]


def references_path(paper_id):
    return config.CACHE_DIR / "refs" / f"{paper_id}.json"


def fetch_references(paper_id):
    """List of paperIds this paper cites. Saved to cache/refs/ for coupling in graph.py."""
    path = references_path(paper_id)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    data = cached_get_json(
        f"{config.S2_BASE}/paper/{paper_id}/references",
        {"fields": "paperId", "limit": 1000},
    )
    refs = []
    for item in (data or {}).get("data") or []:
        cited = item.get("citedPaper") or {}
        if cited.get("paperId"):
            refs.append(cited["paperId"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(refs), encoding="utf-8")
    return refs


def load_references(paper_id):
    """Cached reference list, or [] if fetch.py never saw this paper."""
    path = references_path(paper_id)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def main(topic, n):
    conn = db.connect()
    papers = search_papers(topic, n)
    for p in papers:
        db.upsert_paper(conn, p, topic)
    conn.commit()

    in_set = {p["paperId"] for p in papers}
    n_edges = 0
    for p in tqdm(papers, desc="references"):
        for ref in fetch_references(p["paperId"]):
            if ref in in_set and ref != p["paperId"]:
                db.add_edge(conn, p["paperId"], ref, "cites")
                n_edges += 1
    conn.commit()

    ok = "PASS" if len(papers) >= 200 else "FAIL (need >= 200)"
    print(f"[fetch] {len(papers)} papers with abstracts stored, "
          f"{n_edges} within-set cites edges. Acceptance: {ok}")
    return len(papers), n_edges


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--topic", required=True)
    ap.add_argument("--n", type=int, default=300)
    args = ap.parse_args()
    main(args.topic, args.n)
