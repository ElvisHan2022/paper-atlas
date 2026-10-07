"""Fast checks that need no network, no API key, and no model downloads.

    pytest
"""
import copy
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate import expected_calibration_error, precision_at_k  # noqa: E402
from extract import SCHEMA, validate_extraction  # noqa: E402
from graph import coupling_weight  # noqa: E402
from label import stratified_order  # noqa: E402
from llm import parse_json  # noqa: E402
from score import valid_llm_score  # noqa: E402


def good_extraction():
    d = copy.deepcopy(SCHEMA)
    d["study_design"]["validation"] = "external"
    d["limitations_stated"] = ["small sample"]
    d["limitations_inferred"] = []
    return d


# ---- JSON validators ----

def test_extraction_validator_accepts_schema_shape():
    assert validate_extraction(good_extraction())


@pytest.mark.parametrize("break_it", [
    lambda d: d.pop("discussion"),                                   # missing key
    lambda d: d.update(extra="x"),                                   # extra key
    lambda d: d["study_design"].update(validation="maybe"),          # bad enum
    lambda d: d["results"].update(uncertainty_reported="yes"),       # bool as string
    lambda d: d.update(limitations_stated="small sample"),           # str instead of list
    lambda d: d["evidence"].pop("methods"),                          # nested key missing
])
def test_extraction_validator_rejects_bad_shapes(break_it):
    d = good_extraction()
    break_it(d)
    assert not validate_extraction(d)


def test_llm_score_validator():
    assert valid_llm_score({"score": 4, "rationale": "x"})
    assert not valid_llm_score({"score": 6})
    assert not valid_llm_score({"score": "4"})
    assert not valid_llm_score({"rationale": "no score"})


def test_parse_json_handles_fences_and_garbage():
    assert parse_json('```json\n{"score": 3}\n```') == {"score": 3}
    assert parse_json("no json here") is None
    assert parse_json("{broken") is None


# ---- coupling ----

def test_coupling_weight_toy_example():
    a = ["r1", "r2", "r3", "r4"]
    b = ["r3", "r4", "r5"]
    c = ["r9"]
    assert coupling_weight(a, b) == 2          # share r3 and r4
    assert coupling_weight(a, c) == 0
    assert coupling_weight(a, a + a) == 4      # duplicates don't double count


# ---- ECE ----

def test_ece_perfectly_calibrated_is_zero():
    # 4 papers at 0.25 with 1 relevant; 4 at 0.75 with 3 relevant.
    s = [0.25] * 4 + [0.75] * 4
    y = [1, 0, 0, 0, 1, 1, 1, 0]
    assert expected_calibration_error(y, s) == pytest.approx(0.0)


def test_ece_known_case():
    # Everything scored 0.9 but only half relevant: gap 0.4 in a single bin.
    assert expected_calibration_error([1, 0, 1, 0], [0.9] * 4) == pytest.approx(0.4)
    # Two bins, half the papers each: |0.1 - 0| and |0.9 - 1| -> 0.1 on average.
    assert expected_calibration_error([0, 0, 1, 1], [0.1, 0.1, 0.9, 0.9]) == pytest.approx(0.1)


def test_ece_score_of_one_lands_in_last_bin():
    assert expected_calibration_error([1], [1.0]) == pytest.approx(0.0)


# ---- small helpers ----

def test_precision_at_k():
    y = np.array([1, 0, 1, 0, 0])
    s = np.array([0.9, 0.8, 0.7, 0.2, 0.1])
    assert precision_at_k(y, s, 2) == 0.5
    assert precision_at_k(y, s, 50) == pytest.approx(0.4)   # k larger than n


def test_stratified_order_draws_evenly():
    ids = [f"p{i}" for i in range(50)]
    # 40 papers at rubric 1, 10 at rubric 5.
    scores = {pid: (0.0 if i < 40 else 1.0) for i, pid in enumerate(ids)}
    order = stratified_order(ids, scores, 10)
    top = [scores[p] for p in order]
    assert top.count(1.0) == 5 and top.count(0.0) == 5
    assert stratified_order(ids, scores, 10) == order        # deterministic -> resumable


# ---- web pipeline helpers ----

def test_progress_percent_is_monotonic_across_stages():
    from pipeline import STAGE_SPAN, overall_percent
    points = [overall_percent(s, f) for s in range(len(STAGE_SPAN)) for f in (0, 0.5, 1)]
    assert points == sorted(points)
    assert points[0] == 0 and points[-1] == 100
    assert overall_percent(2, 5.0) == overall_percent(2, 1.0)   # fractions are clamped


def test_rank_key_puts_llm_score_first():
    from pipeline import rank_key
    judged = {"scores": {"llm": 0.75}, "ce_mean": 0.1}
    unjudged = {"scores": {"llm": None}, "ce_mean": 0.99}
    tie = {"scores": {"llm": 0.75}, "ce_mean": 0.5}
    assert sorted([unjudged, judged, tie], key=rank_key, reverse=True) == [tie, judged, unjudged]


def test_review_validator():
    from pipeline import REVIEW_SECTIONS, valid_review
    good = {"overview": "x", **{k: {"summary": "s [1]", "key_points": ["a"]} for k, _ in REVIEW_SECTIONS}}
    assert valid_review(good)
    bad = dict(good, methods={"summary": "s"})            # key_points missing
    assert not valid_review(bad)
    assert not valid_review({k: v for k, v in good.items() if k != "overview"})


def test_clusters_find_obvious_groups():
    from pipeline import choose_clusters, keyword_names
    rng = np.random.default_rng(0)
    vecs = np.vstack([rng.normal(0, .05, (10, 4)) + [1, 0, 0, 0],
                      rng.normal(0, .05, (10, 4)) + [0, 1, 0, 0]])
    labels = choose_clusters(vecs)
    assert len(set(labels[:10])) == 1 and len(set(labels[10:])) == 1
    assert labels[0] != labels[10]
    texts = ["sepsis prediction icu"] * 10 + ["radiology report generation"] * 10
    names = keyword_names(texts, labels)
    assert len(names) == 2 and names[0] != names[1]


# ---- sources (fixtures copy the documented PubMed and arXiv response formats) ----

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def test_parse_pubmed():
    from sources import parse_pubmed
    a, b = parse_pubmed((FIXTURES / "pubmed_efetch.xml").read_text(encoding="utf-8"))
    assert a["paperId"] == "PMID:11111111" and a["year"] == 2024
    assert a["title"] == "A randomized trial of an AI sepsis alert in emergency departments"
    assert a["abstract"].startswith("Background: Sepsis alerts are common. Methods:")
    assert a["venue"] == "N Engl J Med"
    assert a["externalIds"]["DOI"] == "10.1056/NEJMoa0000001"
    assert a["openAccessPdf"]["url"].endswith("PMC9999999?pdf=render")
    assert [x["name"] for x in a["authors"]] == ["Ana Rivera", "SEPSIS-AI Investigators"]
    assert "Randomized Controlled Trial" in a["publicationTypes"]
    assert b["year"] == 2023 and b["openAccessPdf"] is None   # MedlineDate, no PMC copy


def test_parse_arxiv():
    from sources import parse_arxiv
    a, b = parse_arxiv((FIXTURES / "arxiv.xml").read_text(encoding="utf-8"))
    assert a["paperId"] == "ARXIV:2401.01234"                   # version suffix dropped
    assert a["title"] == "A Benchmark for Clinical Reasoning in LLMs"
    assert a["year"] == 2024 and a["venue"] == "arXiv"
    assert a["openAccessPdf"]["url"] == "https://arxiv.org/pdf/2401.01234"
    assert b["venue"] == "ML4H 2023" and b["externalIds"]["DOI"] is None


def test_merge_dedupes_by_doi_and_fills_gaps():
    from sources import merge, parse_arxiv, parse_pubmed
    pubmed = parse_pubmed((FIXTURES / "pubmed_efetch.xml").read_text(encoding="utf-8"))
    arxiv = parse_arxiv((FIXTURES / "arxiv.xml").read_text(encoding="utf-8"))
    s2 = [{"paperId": "abc", "title": "Evaluating Large Language Models on Discharge Summaries",
           "abstract": None, "year": 2023, "venue": "", "citationCount": 12,
           "externalIds": {}, "publicationTypes": ["JournalArticle"], "source": "semantic_scholar"},
          {"paperId": "def", "title": "Evaluating large language models on discharge summaries!",
           "abstract": "Has an abstract.", "year": 2023, "venue": "JAMIA", "citationCount": 12,
           "externalIds": {}, "publicationTypes": [], "source": "semantic_scholar"}]
    merged = merge({"semantic_scholar": s2, "pubmed": pubmed, "arxiv": arxiv})
    # The NEJM trial appears on PubMed and arXiv with the same DOI -> one record.
    trial = [p for p in merged if (p["externalIds"].get("DOI") or "").startswith("10.1056")]
    assert len(trial) == 1 and trial[0]["sources"] == ["pubmed", "arxiv"]
    # Same title (case and punctuation aside) -> one record that keeps S2's citation count.
    summaries = [p for p in merged if "discharge" in p["title"].lower()]
    assert len(summaries) == 1 and summaries[0]["citationCount"] == 12
    assert summaries[0]["sources"] == ["semantic_scholar", "pubmed"]
    assert len(merged) == 3


def test_venue_and_evidence_tags():
    from sources import evidence_tags, venue_type
    assert venue_type("N Engl J Med") == "Clinical journal"
    assert venue_type("J Am Med Inform Assoc") == "Informatics journal"
    assert venue_type("NeurIPS 2023 Datasets and Benchmarks") == "ML / AI venue"
    assert venue_type("arXiv", ["arxiv"]) == "Preprint"
    assert venue_type("Manchester Medical Review") == "Journal"   # "chest" is not a whole word
    assert venue_type("") == "Unknown venue"
    assert venue_type("Nature") == "General science journal"
    assert venue_type("Nature Medicine") == "Clinical journal"
    assert venue_type("Nature Reviews Cardiology") == "Journal"
    assert venue_type("Lecture Notes in Computer Science") == "Journal"
    assert evidence_tags(["Journal Article", "Multicenter Study", "Randomized Controlled Trial"]) \
        == ["RCT", "Multicenter"]
    assert evidence_tags(["MetaAnalysis", "Review"]) == ["Meta-analysis", "Review"]


# ---- API key handling ----

@pytest.mark.parametrize("raw, expected", [
    ("sk-ant-api03-abcdefghijklmnopqrstuvwxyz", "sk-ant-api03-abcdefghijklmnopqrstuvwxyz"),
    ('  "sk-ant-api03-abcdefghijklmnopqrstuvwxyz"  ', "sk-ant-api03-abcdefghijklmnopqrstuvwxyz"),
    ("'sk-ant-api03-abcdefghijklmnopqrstuvwxyz'", "sk-ant-api03-abcdefghijklmnopqrstuvwxyz"),
    ("sk-ant-...", None),          # the placeholder from .env.example
    ("", None),
])
def test_api_key_is_cleaned(monkeypatch, raw, expected):
    import llm
    monkeypatch.setenv("ANTHROPIC_API_KEY", raw)
    assert llm.api_key() == expected
    if expected:
        assert llm.masked_key().startswith("sk-ant-api03") and llm.masked_key().endswith("wxyz")


def test_rejected_key_falls_back_to_local_mode(monkeypatch):
    import types
    import llm

    class Rejected(Exception):
        status_code = 401

    def refuse(**kwargs):
        raise Rejected("invalid x-api-key")

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-abcdefghijklmnopqrstuvwxyz")
    monkeypatch.setattr(llm, "_client", types.SimpleNamespace(models=types.SimpleNamespace(list=refuse)))
    monkeypatch.setattr(llm, "KEY_STATUS", "unchecked")
    assert llm.check_key() == "rejected"
    assert not llm.has_key()          # searches run with the local models instead of failing


def test_parse_openalex():
    import json as _json
    from sources import merge, parse_openalex, parse_pubmed, rebuild_abstract
    a, b = parse_openalex(_json.loads((FIXTURES / "openalex.json").read_text(encoding="utf-8")))
    assert a["paperId"] == "OPENALEX:W4300000001" and a["venue"] == "Nature Medicine"
    assert a["abstract"] == "We validated a sepsis model in a trial."
    assert a["externalIds"]["DOI"] == "10.1038/s41591-024-00001-1"
    assert a["openAccessPdf"]["url"].endswith(".pdf") and a["citationCount"] == 87
    assert [x["name"] for x in a["authors"]] == ["Mei Lin", "Omar Haddad"]
    assert b["abstract"] is None and b["publicationTypes"] == ["Review"] and b["venue"] == ""
    assert rebuild_abstract({}) is None
    # Papers without an abstract are dropped at merge time.
    merged = merge({"openalex": [a, b],
                    "pubmed": parse_pubmed((FIXTURES / "pubmed_efetch.xml").read_text(encoding="utf-8"))})
    assert [p["source"] for p in merged] == ["openalex", "pubmed", "pubmed"]


def test_search_all_runs_sources_in_parallel(monkeypatch):
    import time
    import sources

    def slow(name):
        def search(query, n):
            time.sleep(0.4)
            return [{"paperId": name, "title": f"{name} paper", "abstract": "x", "externalIds": {}}]
        return search

    def broken(query, n):
        raise ConnectionError("403 Forbidden")

    monkeypatch.setattr(sources, "SEARCHERS", {"a": slow("a"), "b": slow("b"), "c": broken})
    start = time.perf_counter()
    merged, counts, errors = sources.search_all("q", ["a", "b", "c"], {"a": 5, "b": 5, "c": 5})
    assert time.perf_counter() - start < 0.7          # ~0.4 s in parallel, not 0.8 s in a row
    assert counts == {"a": 1, "b": 1} and "403" in errors["c"] and len(merged) == 2
