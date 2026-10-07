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

## Demo

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
