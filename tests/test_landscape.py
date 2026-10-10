"""Landscape memory (Milestone 7). Offline: a temp folder stands in for data/ and outputs/."""
import copy
import json
import runpy
import shutil
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
import db  # noqa: E402
import landscape  # noqa: E402
from extract import SCHEMA  # noqa: E402

SEED = config.LANDSCAPE_SEED_DIR
SHIFT_TOPIC = "dataset shift and performance monitoring of clinical prediction models"


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Point every landscape path at a temp folder, with no private fit."""
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(config, "LANDSCAPE_DIR", tmp_path / "data" / "landscape")
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path / "outputs")
    monkeypatch.setattr(config, "LANDSCAPE_PRIVATE_DIR", tmp_path / "private")
    return tmp_path


def seed_doc():
    return landscape.load_fields(SEED / "fields.json")


def seed_obs():
    return landscape.load_observations(SEED / "observations.jsonl")


# ---- EWMA ----

def test_update_metrics_is_an_exponentially_weighted_average():
    field = {"quality": {"external_validation_rate": 0.5, "open_data_rate": None}}
    obs = {"external_validation_rate": 1.0, "open_data_rate": 0.2, "uncertainty_reported_rate": None}
    landscape.update_metrics(field, obs, alpha=0.3)
    q = field["quality"]
    assert q["external_validation_rate"] == pytest.approx(0.65)   # 0.3 * 1.0 + 0.7 * 0.5
    assert q["warrant_gap"] == pytest.approx(0.35)
    assert q["open_data_rate"] == pytest.approx(0.2)              # first value is taken as is
    assert q["uncertainty_reported_rate"] is None                 # no evidence, no change
    assert q["observations"] == 1


# ---- matching ----

@pytest.mark.parametrize("topic, field_id", [
    (SHIFT_TOPIC, "dataset_shift"),
    ("LLM evaluation and reliability for clinical and health text", "llm_med"),
    ("pharmacokinetics pharmacodynamics", "pkpd"),
    ("continuous glucose monitoring forecasting", "cgm"),
])
def test_match_topic_finds_the_field(topic, field_id):
    assert landscape.match_topic(topic, seed_doc()["fields"])[0] == field_id


def test_match_topic_unmatched_gives_a_new_provisional_id():
    fields = seed_doc()["fields"]
    ids = landscape.match_topic("retinal photography for kidney disease screening", fields)
    assert len(ids) == 1 and ids[0] not in {f["id"] for f in fields}


# ---- files ----

def test_save_fields_round_trips_the_seed_unchanged(tmp_path):
    out = tmp_path / "fields.json"
    landscape.save_fields(landscape.load_fields(SEED / "fields.json"), out)
    assert out.read_bytes() == (SEED / "fields.json").read_bytes()


def test_tracked_seed_has_no_personal_fit():
    text = "".join((SEED / n).read_text(encoding="utf-8")
                   for n in ("fields.json", "field_priors.csv", "CONTEXT.md"))
    assert "fit_elvis" not in text


# ---- scoring ----

def test_rescore_reproduces_the_seed_and_the_original_script(tmp_path):
    doc = seed_doc()
    stored = {f["id"]: (f["scores"]["entry_score"], f["quadrant"])
              for f in doc["fields"] if f.get("scores")}
    landscape.rescore(doc, seed_obs(), fit={})
    fresh = {f["id"]: (f["scores"]["entry_score"], f["quadrant"])
             for f in doc["fields"] if f.get("scores")}
    for fid, (score, quadrant) in stored.items():           # acceptance check 1
        assert fresh[fid][0] == pytest.approx(score, abs=0.01)
        assert fresh[fid][1] == quadrant

    for name in ("landscape_score.py", "pubmed_counts_oct9.csv", "field_priors.csv"):
        shutil.copy(SEED / name, tmp_path / name)            # no private/ here, so fit = 2
    runpy.run_path(str(tmp_path / "landscape_score.py"), run_name="__main__")
    original = json.loads((tmp_path / "field_scores.json").read_text())
    assert len(original) == len(fresh)
    for row in original:
        assert fresh[row["id"]][0] == pytest.approx(row["entry_score"], abs=0.01)
        assert fresh[row["id"]][1] == row["quadrant"]


def test_provisional_fields_are_listed_but_not_ranked():
    doc = landscape.rescore(seed_doc(), seed_obs(), fit={})
    provisional = [f for f in doc["fields"] if f["status"] == "provisional"]
    assert provisional and all(not f.get("scores") for f in provisional)


def test_private_fit_changes_entry_scores(sandbox):
    config.LANDSCAPE_PRIVATE_DIR.mkdir()
    (config.LANDSCAPE_PRIVATE_DIR / "fit.csv").write_text("id,fit\ndataset_shift,3\n")
    doc = landscape.rescore(seed_doc(), seed_obs())
    shift = next(f for f in doc["fields"] if f["id"] == "dataset_shift")
    assert shift["scores"]["fit"] == 100.0


# ---- a whole topic run ----

def extraction(validation, data_source, limitation):
    d = copy.deepcopy(SCHEMA)
    d["study_design"].update(validation=validation, data_source=data_source)
    d["results"]["uncertainty_reported"] = validation == "external"
    d["limitations_stated"] = [limitation]
    d["limitations_inferred"] = []
    return d


def toy_db(path, topic):
    conn = db.connect(path)
    rows = [("a", "Nature Medicine", ["JournalArticle"], "external", "MIMIC-IV and eICU",
             "Single-center data may not generalize."),
            ("b", "Some Journal", ["Review"], "internal", "private hospital EHR", "small sample"),
            ("c", "npj Digital Medicine", ["JournalArticle"], "none", "hospital registry",
             "retrospective design"),
            ("d", "arXiv", None, "unclear", "not found", "none stated")]
    for i, (pid, venue, types, validation, source, limitation) in enumerate(rows):
        db.upsert_paper(conn, {"paperId": pid, "title": f"Deep learning paper {pid}",
                               "abstract": "We use machine learning." if i % 2 else "A cohort study.",
                               "year": 2021 + i, "venue": venue, "citationCount": i * 10,
                               "publicationTypes": types}, topic)
        db.save_extraction(conn, pid, topic, extraction(validation, source, limitation), "abstract_only")
    db.add_edge(conn, "a", "b", "cites")
    db.add_edge(conn, "c", "d", "cites")
    conn.commit()
    conn.close()


def test_observe_run_reads_the_database(sandbox):
    path = sandbox / "atlas.db"
    toy_db(path, SHIFT_TOPIC)
    obs = landscape.observe_run(SHIFT_TOPIC, path)
    assert obs["n_papers"] == 4 and obs["n_extracted"] == 4
    assert obs["papers_by_year"] == {"2021": 1, "2022": 1, "2023": 1, "2024": 1}
    assert obs["ai_share"] == 1.0                     # every title says "deep learning"
    assert obs["review_share"] == pytest.approx(1 / 3, abs=1e-3)  # paper d has no known type
    assert obs["top_journal_share"] == 0.5
    assert obs["external_validation_rate"] == 0.25
    assert obs["open_data_rate"] == 0.25
    assert obs["transport_limitation_rate"] == 0.25
    assert obs["communities"] == 2


def test_topic_run_updates_only_its_field(sandbox):
    path = sandbox / "atlas.db"
    toy_db(path, SHIFT_TOPIC)
    landscape.ensure_live()
    before = landscape.load_fields()
    n_obs = len(landscape.load_observations())
    summary = landscape.update_from_run(SHIFT_TOPIC, path)   # acceptance check 2
    after = landscape.load_fields()
    assert summary["field"] == "dataset_shift"
    assert len(landscape.load_observations()) == n_obs + 1
    shift = next(f for f in after["fields"] if f["id"] == "dataset_shift")
    assert shift["quality"]["external_validation_rate"] == 0.25
    assert shift["quality"]["warrant_gap"] == 0.75
    assert [f["priors"] for f in after["fields"]] == [f["priors"] for f in before["fields"]]
    md = (config.OUTPUT_DIR / "landscape.md").read_text(encoding="utf-8")   # acceptance check 4
    html = (config.OUTPUT_DIR / "landscape.html").read_text(encoding="utf-8")
    active = [f for f in after["fields"] if f["status"] == "active"]
    assert all(f["label"] in md and f["label"] in html for f in active)


def test_unmatched_topic_creates_a_provisional_field_and_a_queue_entry(sandbox):
    topic = "retinal photography for kidney disease screening"   # acceptance check 3
    path = sandbox / "atlas.db"
    toy_db(path, topic)
    summary = landscape.update_from_run(topic, path)
    field = next(f for f in landscape.load_fields()["fields"] if f["id"] == summary["field"])
    assert field["status"] == "provisional" and not field.get("scores")
    queue = landscape.live_path("review_queue.md").read_text(encoding="utf-8")
    assert summary["field"] in queue
    # The same topic again lands in the same field and doesn't repeat the queue entry.
    assert landscape.update_from_run(topic, path)["field"] == summary["field"]
    assert landscape.live_path("review_queue.md").read_text(encoding="utf-8") == queue


def test_propose_priors_suggests_but_never_edits(sandbox):
    field = {"id": "x", "status": "active", "priors": {"data_access": 3},
             "quality": {"open_data_rate": 0.1}, "history": []}
    path = sandbox / "queue.md"
    first = landscape.propose_priors(field, {"open_data_rate": 0.1, "n_extracted": 10}, path)
    assert len(first) == 1 and "data_access 3 -> 2" in first[0]
    assert field["priors"] == {"data_access": 3}
    assert landscape.propose_priors(field, {}, path) == []      # no duplicate suggestions


def test_log_changes_records_a_quadrant_change(sandbox):
    old = {"fields": [{"id": "x", "quadrant": "Blooming", "signal": "GO", "history": []}]}
    new = copy.deepcopy(old)
    new["fields"][0]["quadrant"] = "Hot and contested"
    lines = landscape.log_changes(old, new, cause="test")
    assert len(lines) == 1 and new["fields"][0]["history"][-1]["quadrant"] == ["Blooming",
                                                                               "Hot and contested"]
    assert "Hot and contested" in (config.OUTPUT_DIR / "landscape_changes.md").read_text()


# ---- private split ----

def test_import_private_reads_the_original_package(sandbox):
    package = sandbox / "pkg.zip"
    with zipfile.ZipFile(package, "w") as z:
        z.writestr("landscape/field_priors.csv",
                   "id,label,data_access,mvp_simplicity,domain_access,fit_elvis\nt2d,T2D,3,3,3,1\n")
        z.writestr("landscape/CONTEXT.md",
                   "# C\n\n## 4. Current signals\n\n- GO: x\n\n## 5. Owner's lens\n\n- Targets: y\n\n## 6. M\n")
    out = landscape.import_private(package)
    assert (out / "fit.csv").read_text().splitlines() == ["id,fit", "t2d,1"]
    owner = (out / "owner.md").read_text()
    assert "Targets: y" in owner and "## 6." not in owner
    assert landscape.private_fit() == {"t2d": 1.0}
