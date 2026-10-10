# LANDSCAPE_SPEC.md: paper-atlas Milestone 7 (landscape memory)

Instructions for Claude Code. Read `BUILD_SPEC.md` first; every hard constraint there (Python 3.11, Windows paths via `pathlib`, plain functions, few dependencies, cached and rate-limited HTTP, no commits or co-author lines) applies here. Then read `landscape/CONTEXT.md` in full before writing code.

## 1. Goal

Give paper-atlas a persistent, self-reinforcing map of which research fields are saturated and which are open for entry. Each topic run should add evidence to the map, update field scores, and log what changed and why. Seed state already exists in `landscape/`.

## 2. Files supplied (copy into `paper-atlas/landscape/`)

- `CONTEXT.md`: the mental model and operating rules. Load it into the LLM judge's system prompt context when the atlas writes any landscape summary.
- `fields.json`: `meta` plus a list of field records (queries, priors, metrics, scores, quadrant, signal, quality, history).
- `observations.jsonl`: append-only evidence log, one JSON object per line.
- `landscape_score.py`, `field_priors.csv`, `pubmed_counts_oct9.csv`: the original scoring script and its inputs, kept for reproducibility.

## 3. New code

- `landscape.py` with plain functions:
  - `load_fields()` and `save_fields(fields)`: read and write `landscape/fields.json` (write to a temp file, then replace, so a crash cannot corrupt it).
  - `match_topic(topic, fields)`: return field ids whose query terms overlap the topic (case-insensitive token overlap on the PubMed query; threshold in `config.py`). If none, return a new provisional field id.
  - `observe_run(topic, db_path)`: from `data/atlas.db` for that topic, compute papers by year, AI share (title or abstract contains any AI term from `meta`), review share (Semantic Scholar `publicationTypes`, add this field to `fetch.py`), median citations, share of papers in the eight top journals (list in `config.py`), Louvain community count, and from `extractions`: external validation rate, uncertainty reported rate, and open-data rate. Return a dict.
  - `append_observation(field_id, obs)`: append to `observations.jsonl` with date, source `"atlas_run"`, and the topic string.
  - `update_metrics(field, obs, alpha=0.3)`: exponentially weighted update of quality metrics; set `warrant_gap = 1 - external_validation_rate`.
  - `rescore(fields)`: refactor the logic of `landscape_score.py` into a function that takes the field list and returns it with updated `metrics`, `scores`, and `quadrant`. Fields with status `"provisional"` are listed but not ranked.
  - `log_changes(old, new)`: for any field whose quadrant or signal changed, append a `history` entry and a line to `outputs/landscape_changes.md`.
  - `propose_priors(field, obs)`: never edits priors. Writes suggestions with evidence to `landscape/review_queue.md` (for example, lower `data_access` if fewer than 20% of extracted papers use open data).
- `refresh_counts.py --source openalex`: for every active field, query OpenAlex `https://api.openalex.org/works?filter=title_and_abstract.search:<terms>,publication_year:2021-2026&group_by=publication_year` (translate the PubMed query to plain terms; store the translation in `fields.json` under `queries.openalex`), plus the same with the AI terms appended. Append results as observations with source `"openalex"`. Cache responses like every other HTTP call. Add `mailto=` from `.env` for the OpenAlex polite pool.
- `landscape_report.py`: writes `outputs/landscape.md` (ranked table with signal, regime, entry score, warrant gap, and the last change) and `outputs/landscape.html` (single self-contained scatter of crowding against momentum, GO fields highlighted, hover shows the field record; reuse the Cytoscape-free plain SVG approach, no new dependencies).

## 4. Integration

- `run_all.py --topic "..."` gains a final step: `match_topic`, `observe_run`, `append_observation`, `update_metrics`, `rescore`, `log_changes`, `propose_priors`, `save_fields`, then `landscape_report.py`.
- Add `--no-landscape` to skip it.

## 5. Acceptance checks

1. Running `python landscape.py --rescore-only` on the seed files reproduces the seed `entry_score` values within 0.01 (validates the refactor against `landscape_score.py`).
2. A topic run on `"dataset shift and performance monitoring of clinical prediction models"` appends exactly one observation, updates the `dataset_shift` field's quality metrics, and leaves priors unchanged.
3. A topic run on a topic with no matching field creates a provisional field and a `review_queue.md` entry.
4. `outputs/landscape.md` and `outputs/landscape.html` render with all active fields.
5. `pytest` covers: EWMA update on a toy case, `match_topic` on three fixed topics, and that `save_fields` round-trips `fields.json` unchanged.

## 6. LEARN.md additions

Add a section on: percentile ranking versus z-scores for heavy-tailed counts; exponentially weighted averages as evidence accumulation; why priors are human-reviewed; bibliometric caveats (vocabulary drift, near-zero bases, preprint coverage).
