"""Score research fields for saturation and entry feasibility.

Reads pubmed_counts_oct9.csv (bibliometric evidence) and field_priors.csv
(analyst judgments that paper-atlas should revise over time), and writes
field_scores.csv and field_scores.json. Personal fit is read from the gitignored
private/fit.csv; without it every field gets fit 2. landscape.rescore() is the live
version of this logic, and tests check that the two agree.

Why two inputs: counts measure the market (volume, growth, crowding), but
whether a newcomer can ship a credible MVP depends on data access and domain
barriers, which no bibliometric count captures.
"""
import json
import math
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
DEFAULT_FIT = 2   # same as config.LANDSCAPE_DEFAULT_FIT

# arXiv share-adjusted growth (2025 vs 2021) from the DataCite proxy, keyed to PubMed ids.
ARXIV_SHARE_ADJ = {
    "wearables": 1.70, "cgm": 2.34, "digital_phenotyping": 2.34, "alzheimers": 1.28,
    "foundation_models": 8.62, "algo_fairness": 5.30, "radiology": 1.45, "ecg": 1.50,
    "digital_twin": 14.68, "sepsis": 1.24,
}


def pct_rank(series):
    """Percentile rank in [0, 100]; ranks are robust to the heavy-tailed counts."""
    return series.rank(pct=True) * 100


def safe_ratio(num, den, floor=1):
    """Growth ratio that tolerates fields that did not exist in 2021."""
    return num / max(den, floor)


def classify(momentum, crowding, whitespace):
    if crowding >= 50 and momentum < 50:
        return "Crowded, decelerating"
    if crowding >= 50 and momentum >= 50:
        return "Hot and contested"
    if momentum >= 50:
        return "Blooming"
    if whitespace >= 60:
        return "Dormant whitespace (ML gap)"
    return "Quiet niche"


def main():
    counts = pd.read_csv(HERE / "pubmed_counts_oct9.csv")
    priors = pd.read_csv(HERE / "field_priors.csv")
    df = counts.merge(priors, on="id")
    fit_file = HERE / "private" / "fit.csv"
    fit = pd.read_csv(fit_file).set_index("id")["fit"] if fit_file.exists() else pd.Series(dtype=float)
    df["fit_elvis"] = df.id.map(fit).fillna(DEFAULT_FIT)

    # Market momentum: total growth, AI growth, and arXiv growth where measured.
    df["growth"] = [safe_ratio(b, a) for a, b in zip(df.n2021, df.n2025)]
    df["ai_growth"] = [safe_ratio(b, a) for a, b in zip(df.ai2021, df.ai2025)]
    df["arxiv_growth"] = df.id.map(ARXIV_SHARE_ADJ)
    momentum_parts = [pct_rank(df.growth.map(math.log)), pct_rank(df.ai_growth.map(math.log))]
    arxiv_pct = pct_rank(df.arxiv_growth.dropna().map(math.log))
    momentum_parts.append(arxiv_pct.reindex(df.index))
    df["momentum"] = pd.concat(momentum_parts, axis=1).mean(axis=1, skipna=True)

    # Crowding: absolute AI paper volume in 2025 (how many competitors publish ML here).
    df["crowding"] = pct_rank(df.ai2025.map(lambda x: math.log(x + 1)))

    # ML whitespace: clinical papers per AI paper. A big field with little ML is open ground.
    df["ai_share_2025"] = df.ai2025 / df.n2025
    df["whitespace"] = pct_rank((df.n2025 / (df.ai2025 + 1)).map(math.log))

    # Evidence maturity: settled ground truths (RCT base) make labels cheaper for an MVP.
    df["rct_index"] = df.rct / df.n2025
    df["evidence_maturity"] = pct_rank(df.rct_index)

    df["opportunity"] = 0.45 * df.momentum + 0.35 * df.whitespace + 0.20 * (100 - df.crowding)
    df["feasibility"] = (df.data_access + df.mvp_simplicity + df.domain_access) / 9 * 100
    df["fit"] = df.fit_elvis / 3 * 100
    df["entry_score"] = 0.40 * df.opportunity + 0.35 * df.feasibility + 0.25 * df.fit
    df["quadrant"] = [classify(m, c, w) for m, c, w in zip(df.momentum, df.crowding, df.whitespace)]

    cols = ["id", "label", "quadrant", "entry_score", "opportunity", "feasibility", "fit",
            "momentum", "crowding", "whitespace", "evidence_maturity", "growth", "ai_growth",
            "arxiv_growth", "ai_share_2025", "rct_index", "n2025", "ai2025", "skill_axis",
            "mvp_example", "open_data"]
    out = df[cols].sort_values("entry_score", ascending=False).round(2)
    out.to_csv(HERE / "field_scores.csv", index=False)
    (HERE / "field_scores.json").write_text(json.dumps(out.to_dict(orient="records"), indent=2))
    print(out[["id", "quadrant", "entry_score", "opportunity", "feasibility", "fit",
               "momentum", "crowding", "whitespace"]].to_string(index=False))


if __name__ == "__main__":
    main()
