"""The Opportunities tab: prompt, JSON checks, gap memory and the gap map's recency. Offline."""
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
import landscape  # noqa: E402
import opportunities  # noqa: E402
from pipeline import recency  # noqa: E402
from test_landscape import sandbox  # noqa: E402,F401  (pytest fixture)

GAP = {"title": "Label-free performance estimation", "cluster": 2,
       "why_open": "Methods are validated on one site [1].", "defense": "Paper 2 stays in-site [2].",
       "evidence_strength": "moderate",
       "angles": [{"angle": "Confidence estimators", "how": "Benchmark them."}],
       "mvp": {"title": "Failing before labels", "question": "q", "data": "MIMIC-IV", "method": "LR",
               "validation": "eICU", "signal": "s", "effort_weeks": 6, "difficulty": "low"}}
REPLY = {"bottom_line": "Open.", "field_read": "Blooming.", "gaps": [GAP]}
CLUSTERS = {"series": [
    {"cluster": 0, "name": "Drift detection", "size": 60, "recent": 6, "recent_share": 0.1, "top_papers": ["a"]},
    {"cluster": 1, "name": "Label shift", "size": 15, "recent": 12, "recent_share": 0.8, "top_papers": []}]}
PAPERS = [{"rank": 1, "title": "A study", "year": 2024, "venue": "JAMIA", "cluster": 0, "abstract": "x",
           "extraction": {"study_design": {"type": "cohort", "data_source": "MIMIC-IV", "sample_size": "100",
                                           "validation": "internal"},
                          "results": {"headline": "AUROC 0.8", "uncertainty_reported": True},
                          "limitations_stated": ["single site"], "limitations_inferred": []}}]


def test_valid_opportunities_accepts_the_shape_and_rejects_broken_gaps():
    assert opportunities.valid_opportunities(REPLY)
    broken = copy.deepcopy(REPLY)
    del broken["gaps"][0]["mvp"]["data"]
    assert not opportunities.valid_opportunities(broken)
    assert not opportunities.valid_opportunities({**REPLY, "gaps": ["not a dict"]})
    assert not opportunities.valid_opportunities({**REPLY, "gaps": []})


def test_clean_makes_clusters_zero_based_and_drops_bad_values():
    out = opportunities.clean(REPLY, n_clusters=2)
    assert out["gaps"][0]["cluster"] == 1
    bad = copy.deepcopy(REPLY)
    bad["gaps"][0].update(cluster=9, evidence_strength="huge")
    bad["gaps"][0]["mvp"].update(effort_weeks="soon", difficulty="??")
    g = opportunities.clean(bad, n_clusters=2)["gaps"][0]
    assert g["cluster"] is None and g["evidence_strength"] is None
    assert g["mvp"]["effort_weeks"] is None and g["mvp"]["difficulty"] is None


def test_prompt_carries_the_evidence_and_past_gaps():
    field = {"label": "Dataset shift", "status": "active", "signal": "GO", "quadrant": "Blooming",
             "metrics": {"momentum": 63}, "scores": {"entry_score": 81}, "quality": {},
             "priors": {"data_access": 3, "fit_elvis": 3},
             "previous_opportunities": [{"title": "Temporal validation standards", "date": "2026-10-01",
                                         "topic": "older search"}]}
    text = opportunities.build_prompt("shift", "criteria", PAPERS, CLUSTERS, field, [],
                                      "MENTAL MODEL", n_candidates=75)
    for expected in ("[1] A study", "validation: internal", "2. Label shift: 15 papers, recent 80%",
                     "Temporal validation standards", "MENTAL MODEL", "Regime: Blooming",
                     "No matching registered trials", "all candidates: 24%"):
        assert expected in text, expected
    assert "fit_elvis" not in text      # personal fit never goes into prompts


def test_recency_counts_this_year_and_last():
    assert recency([2026, 2025, 2024, 2019, None], 2026) == {"recent": 2, "recent_share": 0.4}
    assert recency([], 2026)["recent_share"] == 0.0


def test_gaps_are_remembered_per_field_without_duplicates(sandbox):  # noqa: F811
    landscape.ensure_live()
    gap = {"title": "Label-free estimation", "mvp": {"title": "MVP"}}
    assert len(landscape.record_opportunities("dataset_shift", "t1", [gap])) == 1
    assert landscape.record_opportunities("dataset_shift", "t2", [{**gap, "title": "label-free ESTIMATION"}]) == []
    for i in range(config.LANDSCAPE_MAX_OPPORTUNITIES + 3):
        landscape.record_opportunities("dataset_shift", "t", [{"title": f"gap {i}", "mvp": {}}])
    field = next(f for f in landscape.load_fields()["fields"] if f["id"] == "dataset_shift")
    assert len(field["opportunities"]) == config.LANDSCAPE_MAX_OPPORTUNITIES
    assert landscape.record_opportunities("no_such_field", "t", [gap]) == []
