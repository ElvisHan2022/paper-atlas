"""Refresh each field's paper counts from OpenAlex, then rescore the landscape.

    python refresh_counts.py --source openalex

For every active field it counts works per year whose title or abstract matches the field's
query, with and without the AI terms, and appends the counts to the evidence log. Once
every active field has OpenAlex counts, scoring switches to them (percentiles only compare
counts from the same source). RCT counts still come from the PubMed seed.
"""
import argparse
import json
import os
import re

import config
import fetch
import landscape
import sources


def openalex_terms(pubmed_query):
    """PubMed syntax to OpenAlex search text: drop [MeSH Terms] tags, and commas, which
    OpenAlex reads as the start of the next filter."""
    text = re.sub(r"\[[^\]]*\]", "", pubmed_query or "").replace(",", " ")
    return re.sub(r"\s+", " ", text).strip()


def ai_terms():
    return " OR ".join(f'"{t}"' for t in config.LANDSCAPE_AI_TERMS)


def count_params(terms, years=config.LANDSCAPE_YEARS):
    params = {"filter": f"title_and_abstract.search:{terms},"
                        f"publication_year:{years[0]}-{years[1]}",
              "group_by": "publication_year"}
    if os.getenv("OPENALEX_EMAIL"):      # OpenAlex's faster "polite pool"
        params["mailto"] = os.getenv("OPENALEX_EMAIL")
    return params


def parse_group_by(data):
    """{year: count} from an OpenAlex group_by=publication_year response."""
    return {int(g["key"]): int(g["count"]) for g in (data or {}).get("group_by") or []
            if str(g.get("key", "")).isdigit()}


def counts_for(terms):
    data = fetch.cached_get(sources.OPENALEX, count_params(terms), "json", 0.2,
                            max_age_days=config.LANDSCAPE_COUNTS_MAX_AGE_DAYS)
    return parse_group_by(data)


def observation(by_year, ai_by_year):
    return {"n2021": by_year.get(2021, 0), "n2025": by_year.get(2025, 0),
            "ai2021": ai_by_year.get(2021, 0), "ai2025": ai_by_year.get(2025, 0),
            "by_year": {str(k): v for k, v in sorted(by_year.items())},
            "ai_by_year": {str(k): v for k, v in sorted(ai_by_year.items())}}


def refresh_openalex(doc):
    done = []
    for f in doc["fields"]:
        pubmed = (f.get("queries") or {}).get("pubmed")
        if f.get("status") != "active" or not pubmed:
            continue
        terms = f["queries"].setdefault("openalex", openalex_terms(pubmed))
        ai_query = f"({terms}) AND ({ai_terms()})"
        try:
            obs = observation(counts_for(terms), counts_for(ai_query))
        except Exception as e:   # one bad query shouldn't lose the rest
            print(f"  {f['id']}: failed ({e})")
            continue
        landscape.append_observation(f["id"], obs, source="openalex", kind="bibliometric",
                                     query=terms, ai_query=ai_query)
        done.append(f["id"])
        print(f"  {f['id']}: {obs['n2021']} -> {obs['n2025']} papers, "
              f"AI {obs['ai2021']} -> {obs['ai2025']}")
    return done


def main(source):
    if source != "openalex":
        raise SystemExit("Only --source openalex is supported.")
    landscape.ensure_live()
    doc = landscape.load_fields()
    old = json.loads(json.dumps(doc))
    done = refresh_openalex(doc)
    landscape.rescore(doc)
    landscape.log_changes(old, doc, cause="refresh_counts openalex")
    landscape.save_fields(doc)
    import landscape_report
    landscape_report.write_reports(doc)
    print(f"Refreshed {len(done)} fields; scoring now uses {doc['meta'].get('counts_source')} counts.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", default="openalex")
    main(ap.parse_args().source)
