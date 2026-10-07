# BUILD_SPEC.md: paper-atlas

Instructions for Claude Code. Read this whole file before writing any code, then build milestones in order. Optimize for a working end-to-end tool today. The owner (Elvis) will study the code afterward, so keep it readable and write `LEARN.md` at the end (see Milestone 6).

## 1. What we are building

A local Python tool that takes a research topic and produces:

1. A ranked set of candidate papers, scored for relevance by **two scorers**: an open-weight cross-encoder and an LLM judge.
2. A **scorer comparison report** that validates both scorers against hand labels on accuracy, calibration, latency, cost, and license. **This is the most important output.** It is the portfolio piece.
3. A **structured extraction** of each relevant paper into fixed sections (study design, methods, results, limitations, discussion), each field backed by an evidence anchor.
4. A **Litmaps-style interactive graph** (x = year, y = citations, edges = citations and bibliographic coupling) plus a folder of linked markdown notes.

Inspiration: Chris Oh's research-landscape agent (topic, retrieve, rerank, extract, synthesize) and Litmaps (citation-based discovery and a timeline layout).

## 2. Hard constraints

- **Python 3.11.** Windows 10/11 machine, VS Code, conda. Use `pathlib` for every path. No shell-specific commands in the code.
- **Minimal, intuitive code.** Plain functions, no classes unless unavoidable, no abstract base classes, no factories, no dependency injection. One script per pipeline step. Prefer 30 obvious lines over 10 clever ones. Short docstrings that say *why*, not *what*.
- **Few dependencies.** Allowed: `requests`, `sentence-transformers`, `torch` (CPU is fine), `anthropic`, `pypdf`, `networkx`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `python-dotenv`, `tqdm`. Use `sqlite3` from the standard library. Ask before adding anything else.
- **Git:** do **not** create a remote, push, or add any co-author or "Generated with" lines to commits. Elvis makes all commits himself. You may suggest commit messages in chat at the end of each milestone.
- **Secrets:** read `ANTHROPIC_API_KEY` (and optional `S2_API_KEY`) from a `.env` file. Create `.env.example`. Add `.env`, `data/`, and `cache/` to `.gitignore`.
- **Be polite to APIs:** cache every HTTP response to `cache/` keyed by URL hash, and rate-limit Semantic Scholar to 1 request per second without a key.
- **All configuration in one place:** `config.py` holds model names, thresholds, prices per million tokens, and file paths.

## 3. Repository layout

```
paper-atlas/
  README.md          # what it is, setup, one-command demo, headline results
  LEARN.md           # written last: study guide mapped to files
  BUILD_SPEC.md      # this file
  environment.yml    # conda env "atlas"
  .env.example
  .gitignore
  config.py
  db.py              # sqlite helpers: connect(), init_schema(), upsert_paper(), etc.
  fetch.py           # Milestone 1
  score.py           # Milestone 2
  label.py           # Milestone 2
  evaluate.py        # Milestone 3
  extract.py         # Milestone 4
  graph.py           # Milestone 5
  run_all.py         # runs fetch -> score -> extract -> graph for a topic
  templates/graph.html
  data/atlas.db      # gitignored
  outputs/           # reports, figures, graph HTML, notes/ (committed)
  tests/test_smoke.py
```

## 4. Data model (SQLite, `data/atlas.db`)

- `papers(paper_id PK, title, abstract, year, venue, citation_count, doi, arxiv_id, pdf_url, authors_json, source, topic)`
- `edges(src, dst, kind)`: `kind` is one of `cites`, `coupling` (with `weight` column), `llm_typed`.
- `scores(paper_id, topic, scorer, score, rationale, latency_ms, input_tokens, output_tokens, cost_usd)`: `scorer` is one of `minilm`, `bge`, `llm`.
- `labels(paper_id, topic, relevant INTEGER, labeled_at)`
- `extractions(paper_id, topic, json, source_text_kind)`: `source_text_kind` is `full_text` or `abstract_only`.

## 5. Milestones

Build in order. After each milestone, run its acceptance check, print a one-line summary, and continue unless the check fails.

### Milestone 1: Fetch (`fetch.py`)

`python fetch.py --topic "LLM evaluation for clinical text" --n 300`

- Query the **Semantic Scholar Graph API** (`/graph/v1/paper/search`) for the topic, paging to `--n` results. Fields: `paperId,title,abstract,year,venue,citationCount,externalIds,openAccessPdf,authors`.
- Drop papers with no abstract. Store everything in `papers`.
- For every stored paper, fetch its references (`/paper/{id}/references`) and store `cites` edges **only between papers that are both in the set**. Keep the raw reference lists in cache for coupling later.
- **Acceptance:** at least 200 papers with abstracts stored, and a printed count of within-set `cites` edges.

### Milestone 2: Score and label (`score.py`, `label.py`)

`python score.py --topic "..." --scorers minilm,bge,llm`

- **Query text** for all scorers: the topic string plus an optional `--criteria` paragraph describing what counts as relevant (default criteria in `config.py`).
- `minilm`: `cross-encoder/ms-marco-MiniLM-L-6-v2` via `sentence_transformers.CrossEncoder`, scoring (query, title + abstract). Apply a sigmoid so scores fall in [0, 1].
- `bge`: `BAAI/bge-reranker-base`, same interface.
- `llm`: Anthropic Messages API (model name in `config.py`; default to a fast, inexpensive Claude model). Prompt with the criteria and a fixed rubric: 1 = unrelated, 2 = tangential, 3 = related background, 4 = relevant, 5 = core. Force JSON output `{"score": int, "rationale": str}`. Retry once on malformed JSON, then record `NULL`. Normalize to [0, 1] as `(score - 1) / 4`.
- Log `latency_ms` per paper for every scorer, plus tokens and `cost_usd` for the LLM using prices in `config.py`.
- `python label.py --topic "..." --k 150` opens a terminal loop. It shows title, year, and abstract, and records `y` / `n` / `s` (skip) / `q` (quit and save). **Sample stratified across the LLM score range** (equal draws from each rubric level where possible) so the labels are not all easy negatives. Show no scores to the labeler.
- **Acceptance:** all three scorers produce a score for at least 95% of papers. The labeling loop resumes where it left off.

### Milestone 3: Scorer comparison (`evaluate.py`) (priority deliverable)

`python evaluate.py --topic "..."`

Using only labeled papers, compute for each scorer:

| Metric | Definition |
|---|---|
| AUROC | `roc_auc_score(labels, score)` |
| Average precision | `average_precision_score` |
| Precision@20 | precision among the top 20 by score |
| Cohen's kappa vs. labels | threshold each scorer at its best F1 cutoff; for the LLM also report the fixed rubric cutoff (score >= 4) |
| Expected calibration error | 10 equal-width bins |
| Bootstrap 95% CI | 1,000 resamples for AUROC and AP |
| Latency | median ms per paper, and projected seconds per 1,000 papers |
| Cost | USD per 1,000 papers (cross-encoders = $0 marginal; note that they run locally) |
| License | MiniLM: Apache-2.0; bge-reranker-base: MIT; LLM: provider terms, data leaves the machine |

Also compute agreement between scorers (Spearman correlation on scores, kappa on thresholded labels).

Outputs:
- `outputs/scorer_comparison.md`: a results table, a reliability diagram, a precision-recall plot, and a five-sentence plain-language conclusion naming the best accuracy-per-dollar option and the conditions under which the LLM is worth its cost.
- `outputs/figures/*.png`.
- **Acceptance:** the report renders with real numbers, and every number in it is reproducible by rerunning the script.

### Milestone 4: Structured extraction (`extract.py`)

`python extract.py --topic "..." --top 30` (top 30 by LLM score, or by labels where available)

- **Text source:** if `openAccessPdf` exists, download it to `cache/pdfs/`, extract text with `pypdf`, and keep the first ~40,000 characters. Otherwise use the abstract and mark `abstract_only`.
- Ask the LLM for this exact JSON (validate keys; retry once on failure):

```json
{
  "study_design": {"type": "", "population": "", "sample_size": "", "data_source": "", "validation": "internal | external | none | unclear"},
  "methods": {"approach": "", "baselines": "", "metrics": ""},
  "results": {"headline": "", "uncertainty_reported": true},
  "limitations_stated": [""],
  "limitations_inferred": [""],
  "discussion": "",
  "evidence": {"study_design": "", "methods": "", "results": "", "limitations_stated": ""}
}
```

- `evidence` values are section or table anchors plus a quote of at most 15 words. Write `"not found"` and never guess. `limitations_inferred` must be clearly separated from what the authors wrote.
- **Acceptance:** 30 extractions stored. Print 3 at random for Elvis to spot-check against the PDFs.

### Milestone 5: Graph and knowledge base (`graph.py`)

`python graph.py --topic "..."`

- Build a `networkx` graph over the relevant papers (LLM score >= 0.75, or labeled relevant).
- **Bibliographic coupling:** for each pair of papers, weight = number of shared references (from cached reference lists). Keep pairs with weight >= 2 as `coupling` edges.
- Detect communities with `networkx.algorithms.community.louvain_communities` on the coupling plus cites graph.
- Export `outputs/graph.json`, and render `outputs/graph.html` from `templates/graph.html` using **Cytoscape.js from cdnjs** (single self-contained file, data inlined):
  - preset layout: x = year, y = log10(citations + 1), small jitter to avoid overlaps;
  - node size scaled by relevance score; node color by community;
  - solid edges for `cites`, faint edges for `coupling`, with a toggle for each;
  - clicking a node opens a side panel with the title, year, scores, and the extraction card (or "no extraction yet").
- Export `outputs/notes/<slug>.md`, one per relevant paper, with the extraction as headed sections and `[[slug]]` wikilinks to its cited and coupled neighbors (Obsidian-compatible).
- **Acceptance:** `graph.html` opens in a browser with no console errors, and clicking a node shows its card.

### Milestone 6: Wrap-up

- `run_all.py --topic "..."` runs fetch, score (all three), extract, and graph. It skips labeling and evaluation, which need a human.
- `tests/test_smoke.py`: tests for the JSON validators, the coupling weight function on a toy example, and ECE on a known case. Use `pytest`.
- `README.md`: one paragraph on what it does, setup in five commands, the demo command, and a "Results" section that embeds the headline table from `scorer_comparison.md`.
- `LEARN.md`: a study guide for Elvis, ordered for learning. For each file, list the 2 to 4 concepts it uses (for example: cross-encoders vs. bi-encoders, sigmoid normalization, stratified sampling for labeling, AUROC vs. average precision under class imbalance, expected calibration error, bootstrap confidence intervals, bibliographic coupling, Louvain communities), with one sentence on why each matters here and the exact lines where it appears. End with five exercises that require changing the code (for example: add a third cross-encoder, swap the LLM model and rerun the comparison).

## 6. Default demo topics

Run the full pipeline on topic A. Leave topic B for Elvis.

- **A:** `"LLM evaluation and reliability for clinical and health text"`, criteria: *empirical studies that evaluate large language models on health or clinical tasks, including benchmarks, robustness, agreement with human raters, or failure analysis.*
- **B:** `"self-supervised learning on wearable sensor data for health"`

## 7. Definition of done

1. `outputs/scorer_comparison.md` exists with real numbers from Elvis's labels. (If labels are not yet done, generate the report on whatever labels exist and print how many more are needed for stable intervals.)
2. `outputs/graph.html` and `outputs/notes/` exist for topic A.
3. `pytest` passes.
4. `README.md` and `LEARN.md` are complete.
5. Print a final summary: papers fetched, scored, labeled, extracted; the top-line AUROC and cost per 1,000 papers for each scorer; and suggested commit messages for Elvis.

## 8. What the finished project should support (for reference, do not put on the resume until true)

> Built and validated a literature-triage tool in Python comparing open-weight cross-encoders (MiniLM, BGE) against an LLM judge for paper relevance on [N] hand-labeled papers, reporting AUROC, calibration, latency, cost, and licensing; extracted structured study design, methods, results, and limitations with evidence anchors, and mapped papers in a Litmaps-style citation graph.
