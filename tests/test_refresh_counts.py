"""OpenAlex counts for the landscape, parsed against a fixture in the documented format."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import refresh_counts  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"


def test_openalex_group_by_parses_and_queries_translate():
    data = json.loads((FIXTURES / "openalex_group_by.json").read_text())
    by_year = refresh_counts.parse_group_by(data)
    assert by_year[2021] == 26 and by_year[2025] == 140 and len(by_year) == 6
    obs = refresh_counts.observation(by_year, {2021: 2, 2025: 30})
    assert (obs["n2021"], obs["n2025"], obs["ai2021"], obs["ai2025"]) == (26, 140, 2, 30)
    assert refresh_counts.openalex_terms('"diabetes mellitus, type 2"[MeSH Terms]') == \
        '"diabetes mellitus type 2"'
