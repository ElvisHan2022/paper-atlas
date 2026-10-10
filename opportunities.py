"""Where to contribute: research gaps, each argued from the evidence, with angles and an MVP.

The search's papers, extractions, topic clusters, registered trials and the field's place in
the landscape all go into one call, which must cite papers as [n] for every claim about the
literature. The independent checker in pipeline.py then audits those claims, so the gaps are
held to the same standard as the review.
"""
import json
from datetime import date

import config
import llm

DIFFICULTY = ("low", "medium", "high")
STRENGTH = ("strong", "moderate", "weak")

SYSTEM = (
    "You are a research strategist advising a master's student in clinical machine learning. "
    "You think like a management consultant sizing a market: where is demand unmet, how hard "
    "is it to enter, and what is the smallest credible first product. You only claim what the "
    "evidence you are given supports, and you say when it is thin.")

PROMPT = """Search topic: {topic}
Relevance criteria: {criteria}

## The field this search belongs to (from the running research landscape)
{field}

## Standing mental model of healthcare AI research
{context}

## Gaps proposed by earlier searches in this field
{previous}

## Topic clusters among all {n_candidates} candidate papers
Size = papers in the cluster. Recent = share published in the last {recent_years} years
(all candidates: {overall_recent:.0%}). A small cluster growing faster than average is often a gap.
{clusters}

## The top {n_papers} papers, with facts extracted from each
{papers}

## Registered trials (ClinicalTrials.gov)
{trials}

Find the {n} most promising gaps where a newcomer could contribute. Favor gaps in evaluation,
validation, monitoring and transport across sites and populations when the evidence shows
them, since those are where demand outruns supply. A good MVP is a simple, well-validated
study on data a student can get (for example a calibrated classifier with external validation
on MIMIC-IV); sophistication is optional.

Return JSON only:
{{"bottom_line": "<2-3 sentences: how open this space is and the single best way in>",
  "field_read": "<1-2 sentences reading the landscape numbers above in plain words>",
  "gaps": [{{
    "title": "<the gap, at most 10 words>",
    "cluster": <cluster number from the list above, or null>,
    "why_open": "<2-4 sentences on why this is unsolved, citing papers as [n]>",
    "defense": "<the strongest objection (e.g. 'someone already did this') and why the gap still holds, citing [n]>",
    "evidence_strength": "strong" | "moderate" | "weak",
    "angles": [{{"angle": "<a way in, at most 8 words>", "how": "<one sentence>"}}],
    "mvp": {{"title": "<project name>", "question": "<one research question>",
            "data": "<named dataset(s) and how to get access>",
            "method": "<the simplest credible method>",
            "validation": "<how it is validated: external site, temporal split, subgroups, calibration>",
            "signal": "<why this reads as quality to a lab or employer>",
            "effort_weeks": <integer>, "difficulty": "low" | "medium" | "high"}}
  }}]}}
Rules:
- Exactly {n} gaps, 2-3 angles each, most promising first.
- Every claim about what papers did or did not do cites them as [n]. Claims about the field
  that come from the landscape numbers or the mental model say so instead of citing a paper.
- If an earlier gap still holds, keep its title so it accumulates evidence; if this search
  contradicts it, say so in its defense.
- Plain, formal prose. No em-dashes."""


def field_block(field):
    if not field:
        return "Not matched to a field (the landscape update failed)."
    m, s, q = field.get("metrics") or {}, field.get("scores") or {}, field.get("quality") or {}
    priors = {k: v for k, v in (field.get("priors") or {}).items() if k != "fit_elvis"}
    lines = [f"Field: {field.get('label')} ({field.get('status')}); owner's signal: {field.get('signal')}"]
    if field.get("quadrant"):
        lines.append(f"Regime: {field['quadrant']}. Percentiles among the mapped fields: momentum "
                     f"{m.get('momentum')}, crowding {m.get('crowding')}, ML whitespace "
                     f"{m.get('whitespace')}, evidence maturity {m.get('evidence_maturity')}.")
        lines.append(f"Growth 2021 to 2025: all papers x{m.get('growth')}, AI papers x{m.get('ai_growth')}; "
                     f"AI share of 2025 papers {m.get('ai_share_2025')}.")
        lines.append(f"Entry score {s.get('entry_score')} (opportunity {s.get('opportunity')}, "
                     f"feasibility {s.get('feasibility')}).")
    else:
        lines.append("Provisional field: not yet ranked against the others.")
    if priors:
        lines.append("Owner's priors (0-3): " + ", ".join(f"{k} {v}" for k, v in priors.items()))
    if q.get("observations"):
        lines.append(f"Across {q['observations']} searches: external validation rate "
                     f"{q.get('external_validation_rate')}, warrant gap {q.get('warrant_gap')}, "
                     f"open data rate {q.get('open_data_rate')}, transport limitation rate "
                     f"{q.get('transport_limitation_rate')}.")
    if field.get("mvp_example"):
        lines.append(f"Seed MVP idea: {field['mvp_example']} (open data: {field.get('open_data')}).")
    return "\n".join(lines)


def previous_block(field):
    past = (field or {}).get("previous_opportunities") or []
    if not past:
        return "None yet; this is the first search in this field that proposes gaps."
    return "\n".join(f"- {o['title']} (proposed {o['date']} from \"{o['topic']}\")" for o in past)


def cluster_block(series):
    return "\n".join(f"{s['cluster'] + 1}. {s['name']}: {s['size']} papers, recent "
                     f"{s.get('recent_share', 0):.0%}, {len(s['top_papers'])} of the top papers"
                     for s in series)


def paper_block(p):
    e = p.get("extraction") or {}
    d, r = e.get("study_design") or {}, e.get("results") or {}
    facts = (f"design: {d.get('type')}; data: {d.get('data_source')}; n: {d.get('sample_size')}; "
             f"validation: {d.get('validation')}; headline: {r.get('headline')}; "
             f"uncertainty reported: {r.get('uncertainty_reported')}; "
             f"limitations stated: {'; '.join(e.get('limitations_stated') or [])}; "
             f"limitations noticed: {'; '.join(e.get('limitations_inferred') or [])}"
             if e else f"abstract: {(p.get('abstract') or '')[:900]}")
    cluster = f", cluster {p['cluster'] + 1}" if p.get("cluster") is not None else ""
    return (f"[{p['rank']}] {p['title']} ({p.get('year')}, {p.get('venue') or 'venue unknown'}"
            f"{cluster})\n{facts}")


def trial_block(trials):
    if trials is None:
        return "Not searched."
    if not trials:
        return "No matching registered trials."
    with_results = sum(1 for t in trials if t.get("has_results"))
    lines = [f"{len(trials)} matching studies, {with_results} with results posted."]
    lines += [f"- {t['title']} ({t.get('status')}, {', '.join(t.get('phases') or []) or 'no phase'})"
              for t in trials[:8]]
    return "\n".join(lines)


def build_prompt(topic, criteria, papers, clusters, field, trials, context, n_candidates):
    series = clusters["series"]
    overall = (sum(s.get("recent", 0) for s in series) / max(1, sum(s["size"] for s in series)))
    return PROMPT.format(
        topic=topic, criteria=criteria, field=field_block(field), context=context.strip(),
        previous=previous_block(field), n_candidates=n_candidates,
        recent_years=config.RECENT_YEARS, overall_recent=overall, clusters=cluster_block(series),
        n_papers=len(papers), papers="\n\n".join(paper_block(p) for p in papers),
        trials=trial_block(trials), n=config.N_OPPORTUNITIES)


def valid_opportunities(d):
    gaps = d.get("gaps")
    if not (isinstance(d.get("bottom_line"), str) and isinstance(d.get("field_read"), str)
            and isinstance(gaps, list) and gaps):
        return False
    for g in gaps:
        if not isinstance(g, dict):
            return False
        mvp = g.get("mvp")
        if not (all(isinstance(g.get(k), str) for k in ("title", "why_open", "defense"))
                and isinstance(g.get("angles"), list) and isinstance(mvp, dict)
                and all(isinstance(mvp.get(k), str) for k in ("title", "question", "data", "method", "validation", "signal"))):
            return False
    return True


def clean(d, n_clusters):
    """Normalize the model's JSON into what the UI renders (clusters become 0-based)."""
    gaps = []
    for g in d["gaps"][:config.N_OPPORTUNITIES]:
        c = g.get("cluster")
        mvp = g["mvp"]
        weeks = mvp.get("effort_weeks")
        gaps.append({
            "title": g["title"].strip(), "why_open": g["why_open"], "defense": g["defense"],
            "cluster": c - 1 if isinstance(c, int) and 1 <= c <= n_clusters else None,
            "evidence_strength": g.get("evidence_strength") if g.get("evidence_strength") in STRENGTH else None,
            "angles": [{"angle": str(a.get("angle", "")), "how": str(a.get("how", ""))}
                       for a in g["angles"] if isinstance(a, dict)][:3],
            "mvp": {**{k: mvp[k] for k in ("title", "question", "data", "method", "validation", "signal")},
                    "effort_weeks": weeks if isinstance(weeks, int) and weeks > 0 else None,
                    "difficulty": mvp.get("difficulty") if mvp.get("difficulty") in DIFFICULTY else None},
        })
    return {"bottom_line": d["bottom_line"], "field_read": d["field_read"], "gaps": gaps,
            "model": config.OPPORTUNITY_MODEL, "date": date.today().isoformat()}


def write_opportunities(topic, criteria, papers, clusters, field, trials, context, n_candidates):
    """One call. Returns (opportunities or None, stats)."""
    user = build_prompt(topic, criteria, papers, clusters, field, trials, context, n_candidates)
    data, stats = llm.ask_json(SYSTEM, user, config.LLM_MAX_TOKENS_OPPORTUNITIES,
                               valid_opportunities, model=config.OPPORTUNITY_MODEL, cache=True)
    if not data:
        return None, stats
    return clean(data, len(clusters["series"])), stats


def check_sections(opps):
    """What the independent checker audits: each gap's argument, in the review's section shape."""
    return [{"title": g["title"], "summary": g["why_open"] + " " + g["defense"], "key_points": []}
            for g in opps["gaps"]]


if __name__ == "__main__":   # print the prompt for the most recent saved run, for inspection
    runs = sorted(config.RUNS_DIR.glob("*.json"))
    if runs:
        r = json.loads(runs[-1].read_text(encoding="utf-8"))
        print(build_prompt(r["topic"], r["criteria"], r["papers"], r["clusters"], r.get("landscape"),
                           r.get("trials"), "(context)", r["n_candidates"]))
