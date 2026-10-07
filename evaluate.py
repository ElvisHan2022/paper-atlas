"""Milestone 3: compare the scorers against your hand labels. This is the headline result.

    python evaluate.py --topic "..."

Writes outputs/scorer_comparison.md and outputs/figures/*.png. Everything is seeded,
so rerunning on the same database gives the same numbers.
"""
import argparse

import matplotlib
matplotlib.use("Agg")  # write PNGs without needing a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, cohen_kappa_score, f1_score,
                             precision_recall_curve, roc_auc_score)

import config
import db

SCORERS = ["minilm", "bge", "llm"]
# Fixed order; marker + line style give a second cue so identity is never color-only.
STYLE = {
    "minilm": {"color": "#2a78d6", "marker": "o", "linestyle": "-"},
    "bge": {"color": "#eb6834", "marker": "s", "linestyle": "--"},
    "llm": {"color": "#1baf7a", "marker": "^", "linestyle": "-."},
}
MIN_PER_CLASS = 30  # rough floor for AUROC intervals narrower than about +/- 0.1


# ---------- metrics (each one small enough to test by hand) ----------

def precision_at_k(y, s, k):
    """Of the k highest-scored papers, what fraction did you label relevant?"""
    k = min(k, len(y))
    top = np.argsort(-np.asarray(s), kind="stable")[:k]
    return float(np.mean(np.asarray(y)[top]))


def best_f1_threshold(y, s):
    """Try every observed score as a cutoff; return the one with the highest F1."""
    best_t, best_f1 = 0.5, -1.0
    for t in np.unique(s):
        f1 = f1_score(y, (np.asarray(s) >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = float(t), f1
    return best_t, best_f1


def expected_calibration_error(y, s, n_bins=config.ECE_BINS):
    """Average gap between 'score says X% likely' and 'X% actually relevant'.

    Equal-width bins on [0, 1]; each bin's gap is weighted by how many papers fall in it.
    """
    y, s = np.asarray(y, dtype=float), np.asarray(s, dtype=float)
    bins = np.minimum((s * n_bins).astype(int), n_bins - 1)  # score 1.0 goes in last bin
    ece = 0.0
    for b in range(n_bins):
        in_bin = bins == b
        if in_bin.any():
            ece += in_bin.mean() * abs(s[in_bin].mean() - y[in_bin].mean())
    return float(ece)


def reliability_bins(y, s, n_bins=config.ECE_BINS):
    """(mean score, fraction relevant, count) per non-empty bin, for the diagram."""
    y, s = np.asarray(y, dtype=float), np.asarray(s, dtype=float)
    bins = np.minimum((s * n_bins).astype(int), n_bins - 1)
    out = []
    for b in range(n_bins):
        in_bin = bins == b
        if in_bin.any():
            out.append((s[in_bin].mean(), y[in_bin].mean(), int(in_bin.sum())))
    return out


def bootstrap_ci(y, s, metric, n=config.BOOTSTRAP_RESAMPLES, seed=config.RANDOM_SEED):
    """95% interval by resampling papers with replacement.

    Resamples with only one class are skipped because AUROC/AP are undefined there.
    """
    rng = np.random.default_rng(seed)
    y, s = np.asarray(y), np.asarray(s)
    values = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        values.append(metric(y[idx], s[idx]))
    if not values:
        return float("nan"), float("nan")
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


# ---------- loading ----------

def load_frame(conn, topic):
    """One row per paper: label (may be NaN) and a column per scorer."""
    rows = conn.execute("SELECT paper_id FROM paper_topics WHERE topic=?", (topic,))
    df = pd.DataFrame({"paper_id": [r["paper_id"] for r in rows]}).set_index("paper_id")
    labels = db.labels_for_topic(conn, topic)
    df["label"] = pd.Series(labels, dtype=float)
    for scorer in SCORERS:
        df[scorer] = pd.Series(db.scores_for_topic(conn, topic, scorer), dtype=float)
    return df.sort_index()


def run_stats(conn, topic, scorer):
    """Median latency and USD per 1,000 papers, over every paper the scorer touched."""
    r = conn.execute(
        "SELECT latency_ms, cost_usd FROM scores WHERE topic=? AND scorer=? "
        "AND score IS NOT NULL", (topic, scorer)).fetchall()
    lat = [x["latency_ms"] for x in r if x["latency_ms"] is not None]
    cost = [x["cost_usd"] or 0.0 for x in r]
    median_ms = float(np.median(lat)) if lat else float("nan")
    return {
        "median_ms": median_ms,
        "sec_per_1k": median_ms,  # ms per paper * 1,000 papers / 1,000 ms per s
        "usd_per_1k": float(np.mean(cost)) * 1000 if cost else float("nan"),
    }


# ---------- report ----------

def evaluate(df, conn, topic):
    """Compute every metric for every scorer on the papers all scorers + a label cover."""
    scorers = [c for c in SCORERS if df[c].notna().any()]
    lab = df.dropna(subset=["label"] + scorers)
    y = lab["label"].astype(int).to_numpy()
    results = {}
    for sc in scorers:
        s = lab[sc].to_numpy()
        t, f1 = best_f1_threshold(y, s)
        row = {
            "auroc": roc_auc_score(y, s),
            "auroc_ci": bootstrap_ci(y, s, roc_auc_score),
            "ap": average_precision_score(y, s),
            "ap_ci": bootstrap_ci(y, s, average_precision_score),
            "p_at_k": precision_at_k(y, s, config.EVAL_TOP_K),
            "threshold": t,
            "f1": f1,
            "kappa": cohen_kappa_score(y, (s >= t).astype(int)),
            "ece": expected_calibration_error(y, s),
            **run_stats(conn, topic, sc),
        }
        if sc == "llm":
            row["kappa_rubric"] = cohen_kappa_score(
                y, (s >= config.LLM_RELEVANT_CUTOFF).astype(int))
        results[sc] = row
    return lab, y, scorers, results


def agreement(df, scorers, results):
    """Spearman on raw scores (all papers both scored), kappa on thresholded calls."""
    out = []
    for i, a in enumerate(scorers):
        for b in scorers[i + 1:]:
            both = df.dropna(subset=[a, b])
            rho = both[a].corr(both[b], method="spearman")
            ka = cohen_kappa_score((both[a] >= results[a]["threshold"]).astype(int),
                                   (both[b] >= results[b]["threshold"]).astype(int))
            out.append((a, b, len(both), rho, ka))
    return out


def plot_reliability(lab, y, scorers, path):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], color="#9a9893", linewidth=1, label="perfect calibration")
    for sc in scorers:
        pts = reliability_bins(y, lab[sc].to_numpy())
        ax.plot([p[0] for p in pts], [p[1] for p in pts], linewidth=2, markersize=8,
                label=sc, **STYLE[sc])
    ax.set_xlabel("mean predicted score in bin")
    ax.set_ylabel("fraction labeled relevant")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Reliability diagram (10 equal-width bins)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_pr(lab, y, scorers, results, path):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for sc in scorers:
        prec, rec, _ = precision_recall_curve(y, lab[sc].to_numpy())
        ax.plot(rec, prec, linewidth=2, label=f"{sc} (AP {results[sc]['ap']:.2f})",
                color=STYLE[sc]["color"], linestyle=STYLE[sc]["linestyle"])
    ax.axhline(y.mean(), color="#9a9893", linewidth=1,
               label=f"base rate {y.mean():.2f}")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title("Precision-recall on labeled papers")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def md_table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def headline_table(scorers, results):
    rows = []
    for sc in scorers:
        r = results[sc]
        kappa = f"{r['kappa']:.2f}"
        if "kappa_rubric" in r:
            kappa += f" (rubric >=4: {r['kappa_rubric']:.2f})"
        rows.append([
            sc,
            f"{r['auroc']:.3f} [{r['auroc_ci'][0]:.2f}, {r['auroc_ci'][1]:.2f}]",
            f"{r['ap']:.3f} [{r['ap_ci'][0]:.2f}, {r['ap_ci'][1]:.2f}]",
            f"{r['p_at_k']:.2f}",
            kappa,
            f"{r['ece']:.3f}",
            f"{r['median_ms']:.1f}",
            f"{r['sec_per_1k']:.1f}",
            f"${r['usd_per_1k']:.2f}",
            config.LICENSES[sc],
        ])
    header = ["Scorer", "AUROC [95% CI]", "AP [95% CI]", f"P@{config.EVAL_TOP_K}",
              "Kappa @ best-F1", "ECE", "Median ms/paper", "Sec / 1k papers",
              "USD / 1k papers", "License"]
    return md_table(header, rows)


def conclusion(scorers, results, n, n_pos):
    """Five plain sentences, written from the numbers so they stay true on rerun."""
    best = max(scorers, key=lambda s: results[s]["auroc"])
    local = [s for s in scorers if s != "llm"]
    best_local = max(local, key=lambda s: results[s]["auroc"]) if local else None
    lines = [f"On {n} hand-labeled papers ({n_pos} relevant), **{best}** ranked relevant "
             f"papers best (AUROC {results[best]['auroc']:.3f})."]
    if best_local:
        lines.append(
            f"The best accuracy per dollar is **{best_local}**: AUROC "
            f"{results[best_local]['auroc']:.3f} at $0 marginal cost, running locally in "
            f"about {results[best_local]['sec_per_1k']:.1f} seconds per 1,000 papers.")
    if "llm" in scorers and best_local:
        llm, loc = results["llm"], results[best_local]
        gap = llm["auroc"] - loc["auroc"]
        overlap = llm["auroc_ci"][0] <= loc["auroc_ci"][1] and loc["auroc_ci"][0] <= llm["auroc_ci"][1]
        lines.append(
            f"The LLM judge costs ${llm['usd_per_1k']:.2f} per 1,000 papers and its AUROC is "
            f"{abs(gap):.3f} {'higher' if gap >= 0 else 'lower'} than {best_local}'s; "
            f"the 95% intervals {'overlap, so this gap is not clearly real at this sample size' if overlap else 'do not overlap'}.")
        lines.append(
            "The LLM is worth its cost when its accuracy gain survives more labels, when a "
            "one-sentence rationale per paper is useful to you, and when sending abstracts "
            f"to an external provider is acceptable; otherwise screen with {best_local}.")
    best_cal = min(scorers, key=lambda s: results[s]["ece"])
    lines.append(
        f"Cross-encoder sigmoid outputs are ranking scores, not probabilities, so use them "
        f"to sort rather than to threshold; **{best_cal}** is the best calibrated "
        f"(ECE {results[best_cal]['ece']:.3f}).")
    return " ".join(lines)


def labels_needed(n, n_pos):
    """How many more labels until both classes reach MIN_PER_CLASS, at the current rate."""
    n_neg = n - n_pos
    if n == 0:
        return 2 * MIN_PER_CLASS
    rate = n_pos / n
    more_pos = (MIN_PER_CLASS - n_pos) / rate if n_pos < MIN_PER_CLASS and rate > 0 else 0
    more_neg = (MIN_PER_CLASS - n_neg) / (1 - rate) if n_neg < MIN_PER_CLASS and rate < 1 else 0
    return int(np.ceil(max(more_pos, more_neg, 0)))


def update_readme(table, n, n_pos):
    """Copy the headline table into README.md between its results markers."""
    readme = config.ROOT / "README.md"
    start, end = "<!-- results:start -->", "<!-- results:end -->"
    text = readme.read_text(encoding="utf-8") if readme.exists() else ""
    if start not in text or end not in text:
        return
    block = (f"{start}\n{table}\n\n_{n} hand-labeled papers ({n_pos} relevant). "
             f"Full report: [outputs/scorer_comparison.md](outputs/scorer_comparison.md)._\n{end}")
    before, rest = text.split(start, 1)
    readme.write_text(before + block + rest.split(end, 1)[1], encoding="utf-8")


def main(topic):
    conn = db.connect()
    df = load_frame(conn, topic)
    lab, y, scorers, results = evaluate(df, conn, topic)
    n, n_pos = len(y), int(y.sum())
    if n == 0 or n_pos in (0, n):
        raise SystemExit("Need labels from both classes. Run label.py first.")

    config.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plot_reliability(lab, y, scorers, config.FIGURE_DIR / "reliability.png")
    plot_pr(lab, y, scorers, results, config.FIGURE_DIR / "precision_recall.png")

    agree = agreement(df, scorers, results)
    agree_tbl = md_table(
        ["Scorer A", "Scorer B", "Papers", "Spearman rho", "Kappa (best-F1 cutoffs)"],
        [[a, b, m, f"{rho:.2f}", f"{ka:.2f}"] for a, b, m, rho, ka in agree])
    thresholds = ", ".join(f"{s} >= {results[s]['threshold']:.3f} (F1 {results[s]['f1']:.2f})"
                           for s in scorers)
    more = labels_needed(n, n_pos)
    stability = (f"Intervals are stable enough to compare (>= {MIN_PER_CLASS} per class)."
                 if more == 0 else
                 f"Label about **{more} more** papers to reach {MIN_PER_CLASS} per class; "
                 "until then treat the intervals as wide.")

    report = f"""# Scorer comparison: {topic}

{conclusion(scorers, results, n, n_pos)}

## Results

<!-- headline-table:start -->
{headline_table(scorers, results)}
<!-- headline-table:end -->

- Labeled papers scored by every scorer: **{n}** ({n_pos} relevant, base rate {n_pos / n:.2f}). {stability}
- Kappa uses each scorer's best-F1 cutoff on these labels: {thresholds}. The LLM also shows its fixed rubric cutoff (score >= 4).
- Latency is median wall-clock per paper on this machine. Cross-encoders are batched on CPU (batch time / batch size); the LLM is one API call per paper.
- Cost: cross-encoders are $0 marginal because they run locally. LLM cost uses ${config.PRICE_INPUT_PER_M:.2f} / ${config.PRICE_OUTPUT_PER_M:.2f} per million input / output tokens for `{config.LLM_MODEL}` (set in `config.py`).
- Bootstrap: {config.BOOTSTRAP_RESAMPLES:,} resamples, seed {config.RANDOM_SEED}.

## Calibration

![Reliability diagram](figures/reliability.png)

Points on the diagonal mean "a score of 0.7 is right 70% of the time". ECE is the
count-weighted average distance from the diagonal over 10 equal-width bins.

## Precision and recall

![Precision-recall curves](figures/precision_recall.png)

With few relevant papers, average precision is the stricter measure: AUROC can look good
while the top of the list is still noisy.

## Agreement between scorers

Computed on all fetched papers each pair scored (labels not needed).

{agree_tbl}

_Regenerate with `python evaluate.py --topic "{topic}"`._
"""
    out = config.OUTPUT_DIR / "scorer_comparison.md"
    out.write_text(report, encoding="utf-8")
    update_readme(headline_table(scorers, results), n, n_pos)

    print(f"[evaluate] {n} labeled papers ({n_pos} relevant). Wrote {out}")
    for sc in scorers:
        r = results[sc]
        print(f"  {sc:7s} AUROC {r['auroc']:.3f}  AP {r['ap']:.3f}  "
              f"ECE {r['ece']:.3f}  ${r['usd_per_1k']:.2f}/1k papers")
    if more:
        print(f"[evaluate] Label about {more} more papers for stable intervals.")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--topic", required=True)
    args = ap.parse_args()
    main(args.topic)
