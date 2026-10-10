# paper-atlas

Local tool that finds, ranks, reviews and maps research papers for a healthcare topic.
Python 3.11, conda env `atlas`, Windows-first (use `pathlib`, no shell-specific code).

## Commands
- Search UI: `python app.py` → http://127.0.0.1:8000 (prints the .env path and key status)
- Tests: `pytest` (offline: no network, no API key, no model downloads)
- Command-line pipeline: `python run_all.py --topic "..."`, then `label.py`, `evaluate.py`

## Layout
- `app.py` web server (stdlib only) → `pipeline.py` runs one search in 4 stages
- `sources.py` PubMed / OpenAlex / Europe PMC preprints / Semantic Scholar / arXiv (merged by
  DOI or title) + ClinicalTrials.gov (own tab, never ranked as papers)
- `score.py` cross-encoders (MiniLM, BGE) + LLM judge · `extract.py` structured extraction
- `pipeline.py` review writer + independent verifier + topic clusters + step log
- `opportunities.py` the first results tab: cited gaps, angles and MVPs, audited by the same
  verifier; every other tab is evidence for it
- `llm.py` the only place that calls the Anthropic API (JSON, retry once, optional cache)
- `config.py` every model name, price, threshold, source limit and lens
- `templates/app.html` the whole UI (plain HTML/CSS/JS, no build step)
- `evaluate.py`, `label.py`, `graph.py` command-line evaluation and citation map
- `landscape.py` field map updated by every run (`landscape_report.py`, `refresh_counts.py`);
  tracked seed in `landscape/`, live copy in `data/landscape/`, personal lens and fit in the
  gitignored `landscape/private/`. Runs never edit priors; they only write to the review queue
- Data (gitignored): `data/atlas.db`, `cache/` (HTTP + LLM answers), `outputs/runs/`
- `docs/ROADMAP.md` red flags, planned work and build order; read it before larger changes

## Conventions
- Plain functions, no classes unless unavoidable. Short docstrings that say why.
- Every knob lives in `config.py`.
- Allowed dependencies: requests, sentence-transformers, torch, anthropic, pypdf, networkx,
  numpy, pandas, scikit-learn, matplotlib, python-dotenv, tqdm, pytest. Ask before adding any.
- Parse external APIs against fixtures in `tests/fixtures/` (copied from documented formats).
- Fan out only work that is truly independent (sources, per-paper calls, review sections).

## Rules
- IMPORTANT: never commit `.env`, API keys, `data/`, `cache/`, `outputs/runs/` or
  `landscape/private/` (the repo is public; career targets and fit scores stay local).
- IMPORTANT: do not change `score.LLM_PROMPT`; evaluate.py results must stay comparable.
  Lens behaviour goes in `score.LENS_PROMPT`.
- The review writer never grades its own work: verification stays a separate call with its
  own context and only the papers' own text as evidence.
- Commits are authored as the repo owner (ElvisHan2022), with no AI co-author or
  "Generated with" lines. Run `pytest` before every commit; every commit should pass alone.

## Reporting progress
Before reporting progress, audit each claim against a tool result from this session. Only
report work you can point to evidence for. If something is not yet verified, say so
explicitly. If tests fail, say so with the output. If a step was skipped, state that.
