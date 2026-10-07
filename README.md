# paper-atlas

paper-atlas takes a research topic, pulls candidate papers from Semantic Scholar, and scores each one for relevance with two kinds of scorer: open-weight cross-encoders (MiniLM, BGE) that run locally for free, and an LLM judge (Claude) that runs through an API. It then checks both kinds against your own hand labels on accuracy, calibration, latency, cost, and license. For the most relevant papers it extracts study design, methods, results, and limitations, with an evidence quote behind each field. Finally it draws a Litmaps-style citation map (x = year, y = citations) and writes a folder of linked, Obsidian-ready notes.

## Setup (Windows, conda)

```bash
git clone <this repo> && cd paper-atlas
conda env create -f environment.yml
conda activate atlas
copy .env.example .env        # then paste your ANTHROPIC_API_KEY into .env (macOS/Linux: cp)
pytest
```

## Search UI

```bash
python app.py
```

This opens http://127.0.0.1:8000. Type a topic or a few keywords. Under the search bar you can pick:

- **Sources:** PubMed, OpenAlex (an open index of essentially every journal: Nature, Nature Medicine, Cell, NEJM…), Semantic Scholar, and arXiv. Papers found in more than one are merged by DOI or title.
- **Evidence lens:** different journals reward different work, so the lens tells the LLM judge what to value on top of relevance:
  - **Balanced:** relevance only.
  - **Clinical impact:** trials, external validation, deployment and patient outcomes (NEJM, Lancet, JAMA, Nature Medicine).
  - **Methods rigor:** baselines, uncertainty, validation, bias and calibration (JAMIA, npj Digital Medicine).
  - **Novelty:** new methods, models and benchmarks (NeurIPS, ICML, ML4H, CHIL).

The search then runs in four stages:

1. **Processing your query.** Claude restates it as a topic and writes relevance criteria.
2. **Identifying candidate papers.** It searches the chosen sources (up to 80 + 80 + 60 + 25 hits) and merges duplicates.
3. **Scoring the papers.** MiniLM and BGE score every candidate locally; the top 30 go to the LLM judge, which scores relevance (1-5) and, with a lens, lens fit (1-5). The final rank is 60% relevance + 40% lens fit.
4. **Finishing scoring.** It extracts the top 10, writes the review, and groups all candidates into topics.

Results open in four tabs:
- **Candidate papers:** the top 10, with scores, the judge's reasoning, evidence type (RCT, meta-analysis…) and venue type.
- **Systematic review:** five sections (study design, methods, results, limitations, discussion), with every claim linked to its papers.
- **Topic clusters:** a line chart of papers per topic over time.
- **Scoring rubric:** how the papers were ranked.

Finished searches are saved in `outputs/runs/` and listed under "Recent searches". A search costs a few cents with Haiku.

**Where the key goes:** in a file named `.env` in the `paper-atlas` folder on your computer, next to `app.py` (never in GitHub). When `python app.py` starts, it prints the exact path it reads and whether Anthropic accepted the key.

**No API key?** The app still works. It ranks papers with the local models and builds the topic chart. The LLM judge, the lens and the review need `ANTHROPIC_API_KEY` in `.env`.

## Demo (command line)

```bash
python run_all.py --topic "LLM evaluation and reliability for clinical and health text"
python label.py --topic "LLM evaluation and reliability for clinical and health text" --k 150
python evaluate.py --topic "LLM evaluation and reliability for clinical and health text"
```

`run_all.py` runs fetch, score (all three scorers), extract, and graph. Labeling and evaluation need you at the keyboard. Open `outputs/graph.html` in a browser, and open `outputs/notes/` as an Obsidian vault.

Each step also runs on its own:

| Step | Command | Output |
|---|---|---|
| Fetch | `python fetch.py --topic "..." --n 300` | papers + citation edges in `data/atlas.db` |
| Score | `python score.py --topic "..." --scorers minilm,bge,llm` | `scores` table |
| Label | `python label.py --topic "..." --k 150` | `labels` table (resumable) |
| Evaluate | `python evaluate.py --topic "..."` | `outputs/scorer_comparison.md`, `outputs/figures/` |
| Extract | `python extract.py --topic "..." --top 30` | `extractions` table |
| Graph | `python graph.py --topic "..."` | `outputs/graph.html`, `outputs/graph.json`, `outputs/notes/` |

Every HTTP response is cached in `cache/`, and every step skips work it has already done, so you can rerun steps freely. Models, prices, and thresholds all live in `config.py`.

## Results

`evaluate.py` fills in this section from `outputs/scorer_comparison.md`.

<!-- results:start -->
_No results yet: run the demo, label at least 60 papers, then run `evaluate.py`._
<!-- results:end -->
