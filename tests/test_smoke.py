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
