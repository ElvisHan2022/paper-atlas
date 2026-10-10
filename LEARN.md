# LEARN.md: a study guide to paper-atlas

The files are listed in the order the data flows through them, which is also a good
order to learn them in. Each file has 2 to 4 concepts, one sentence on why each matters
here, and the lines where it happens (`file.py:line`). Read the code next to this guide.

---

## 1. `config.py` and `db.py`: where things live

| Concept | Why it matters here | Where |
|---|---|---|
| Single source of configuration | Swapping a model or changing a price is a one-line edit, so experiments stay comparable. | `config.py` (whole file) |
| SQLite as a lab notebook | Every score, label, and extraction is a row you can query with plain SQL, so results survive crashes and can be audited later. | `db.py:7-35` (schema) |
| Upsert (`ON CONFLICT ... DO UPDATE`) | Re-fetching a paper refreshes its citation count without duplicating it, which is what makes reruns safe. | `db.py:61` |
| Many-to-many link table | `paper_topics` lets one paper belong to topic A and topic B without either run clobbering the other. | `db.py:13-17`, `db.py:73` |

## 2. `fetch.py`: getting candidates (Milestone 1)

| Concept | Why it matters here | Where |
|---|---|---|
| Content-addressed HTTP cache | Hashing the full URL gives each request a stable file name, so a rerun costs zero API calls and the dataset is frozen for reproducibility. | `fetch.py:20-50` |
| Rate limiting and exponential backoff | Semantic Scholar allows about 1 request per second without a key; backing off 1, 2, 4... seconds on a 429 is how polite clients recover. | `fetch.py:33-42` |
| Pagination | Search returns 100 results per page, and many lack abstracts, so we keep paging until we hold `n` usable papers. | `fetch.py:53-72` |
| Induced subgraph | We keep only citations where *both* ends are in our set, because edges to papers we never fetched can't be drawn or scored. | `fetch.py:115` |

## 3. `score.py` and `llm.py`: three relevance scorers (Milestone 2)

| Concept | Why it matters here | Where |
|---|---|---|
| Cross-encoder vs. bi-encoder | A cross-encoder reads the query and the paper *together* through one transformer, so it is more accurate than embedding each separately (bi-encoder) but must run once per pair. With 300 papers, that cost is fine. | `score.py:53-69` |
| Sigmoid normalization | The cross-encoders output unbounded logits; `1 / (1 + e^-x)` maps them to (0, 1) so all scorers share a scale. A score in (0, 1) is still not a probability (see ECE below). | `score.py:48-50`, `score.py:64-66` |
| LLM-as-judge with a fixed rubric | A 1-5 rubric with named levels makes the LLM's scores consistent and maps cleanly to [0, 1] via `(s - 1) / 4`. | `score.py:22-36`, `score.py:81-91` |
| Validate, retry once, then record NULL | LLMs sometimes return broken JSON. A NULL is honest about the failure, while a made-up default would quietly bias the evaluation. | `llm.py:29-37`, `llm.py:40-68` |

## 4. `label.py`: making ground truth (Milestone 2)

| Concept | Why it matters here | Where |
|---|---|---|
| Stratified sampling | Random sampling from search results gives mostly easy negatives, so every scorer looks great. Equal draws per LLM rubric level put hard, borderline papers in your label set. | `label.py:17-40` |
| Interleaving | Ordering the draws as level 1, 2, 3, 4, 5, 1, 2... keeps the sample balanced even if you quit after 40 labels. | `label.py:36-39` |
| Blind labeling | The labeler never sees a score, so the labels can't be anchored to the thing being evaluated. | `label.py:43-48` |
| Deterministic resume | A seeded order plus "skip what's already in the table" means `q` and rerun lose nothing. | `label.py:58-61` |

## 5. `evaluate.py`: the headline comparison (Milestone 3)

| Concept | Why it matters here | Where |
|---|---|---|
| AUROC vs. average precision under class imbalance | AUROC asks "is a random relevant paper ranked above a random irrelevant one?" and ignores the base rate. AP focuses on the top of the list, so it drops sharply when relevant papers are rare. Report both. | `evaluate.py:135-138` |
| Thresholded agreement (Cohen's kappa, best-F1 cutoff) | Kappa corrects raw agreement for chance. Each scorer gets its own best cutoff so a badly scaled scorer isn't penalized for scale alone, and the LLM's fixed rubric cutoff (>= 4) shows the "no tuning" case. | `evaluate.py:40-47`, `evaluate.py:142-148` |
| Expected calibration error | ECE asks whether "score 0.8" means "relevant 80% of the time". Cross-encoders rank well but are often badly calibrated, which matters when you pick a cutoff. | `evaluate.py:50-62` |
| Bootstrap confidence intervals | With about 150 labels, AUROC can move by ±0.07 from sampling alone. Resampling papers 1,000 times tells you whether a gap between scorers is real. | `evaluate.py:77-92` |

Also worth reading: scorer-vs-scorer Spearman correlation (`evaluate.py:153-163`), and
`labels_needed` (`evaluate.py:268`), which estimates how many more labels you need.

## 6. `extract.py`: structured extraction (Milestone 4)

| Concept | Why it matters here | Where |
|---|---|---|
| Schema validation | Checking exact keys, types, and the `validation` enum catches the model inventing or dropping fields before they reach the notes. | `extract.py:56-74` |
| Evidence anchoring | A section anchor plus a quote of at most 15 words lets you verify each claim in seconds, and "not found" makes missing evidence visible instead of guessed. | `extract.py:34-52` |
| Stated vs. inferred limitations | Keeping the model's critique apart from the authors' own words stops the summary from putting opinions in the authors' mouths. | `extract.py:42` |
| Graceful degradation | No PDF, a paywall page, or a broken PDF all fall back to the abstract and are labeled `abstract_only`, so you know how much to trust each card. | `extract.py:101-107` |

## 7. `graph.py` and `templates/graph.html`: the map (Milestone 5)

| Concept | Why it matters here | Where |
|---|---|---|
| Bibliographic coupling | Two papers that cite the same sources are probably about the same thing, even if neither cites the other. This links recent papers that have no citations yet. | `graph.py:29-31`, `graph.py:58-67` |
| Louvain community detection | Louvain greedily groups nodes so that edges inside groups outnumber what chance predicts (modularity). The groups become the node colors, i.e. sub-areas of the field. | `graph.py:72-76` |
| Preset (semantic) layout | Placing x = year and y = log citations makes position *mean* something, as in Litmaps. A force layout would place nodes arbitrarily. Log scale stops one landmark paper from flattening the rest. | `graph.py:96-107` |
| Safe data inlining | Escaping `</` stops a paper title containing `</script>` from breaking the page. | `graph.py:243` |

## 8. `pipeline.py`, `app.py`, and `templates/app.html`: the search UI

| Concept | Why it matters here | Where |
|---|---|---|
| Scoring cascade | Free local models screen all 100 candidates, and only the top 30 reach the paid LLM. This keeps the LLM's accuracy where it matters at a fraction of the cost (the trade-off `evaluate.py` measures). | `pipeline.py:305`, `pipeline.py:86-89` |
| Bi-encoder embeddings + k-means + silhouette | Clustering needs one vector per paper, which is exactly what a bi-encoder gives (a cross-encoder only scores pairs). Silhouette picks the number of topics that separates them most cleanly. | `pipeline.py:151-183`, `pipeline.py:213-250` |
| Background job + polling | The search takes a minute, so the server runs it in a thread and the page asks "how far along?" three times a second. That's simpler than WebSockets, with the standard library only. | `app.py:35-53`, `templates/app.html:500` |
| Progress as stage spans | Each stage owns a slice of 0-100%, so progress inside a stage maps to the bar and never moves backward. The browser then eases the displayed number toward the real one. | `pipeline.py:28`, `pipeline.py:51-53`, `templates/app.html:560-567` |
| Grounded synthesis | The review is written from the structured extractions, not the raw papers, and every claim must cite [n]. The UI turns citations into links so you can check them. | `pipeline.py:114-145`, `templates/app.html:633-641` |

## 9. Agent patterns: verifier, fan-out, logs and caching

| Concept | Why it matters here | Where |
|---|---|---|
| Maker / checker separation | A model reviewing its own output in the same context skews positive. The verifier gets a fresh context, a "you didn't write this" role, and different evidence (raw abstracts and quotes). | `pipeline.py` `verify_section`, `VERIFY_PROMPT` |
| Fan out only independent work | Parallel calls pay off when pieces don't need each other: sources, per-paper judging, review sections. The review itself stays one call, because it must compare papers. | `sources.search_all`, `pipeline.judge_many`, `verify_review` |
| Append-only step log | Written line by line as the run goes, so a crash still leaves a record of how far it got and why. | `pipeline.make_run_log`, `outputs/runs/*.log.jsonl` |
| Content-addressed LLM cache | Same request means same answer from disk. Retrying a failed search costs only the unfinished steps. | `llm.ask_json(cache=True)` |
| CLAUDE.md | The briefing a future Claude Code session reads first: commands, layout, and the few rules that must not be broken. | `CLAUDE.md` |

## 10. `landscape.py`: a research landscape that learns from every run (Milestone 7)

The seed in `landscape/` scores 26 healthcare AI fields for crowding, momentum and entry
feasibility. Every topic run adds one observation and nudges the matching field.

| Concept | Why it matters here | Where |
|---|---|---|
| Percentile ranks instead of z-scores | Paper counts are heavy-tailed: radiology has 11,721 AI papers in 2025 while dataset shift has 65. A z-score would let one giant field squash everyone else toward the mean, so the mean and standard deviation describe nobody. A percentile rank only asks "what share of fields is below this one?", which is robust to outliers and keeps every score on the same 0-100 scale. Logging counts first only changes ties, not ranks; it matters for the ratios. | `landscape.pct_rank`, `landscape.rescore` |
| Exponentially weighted averages as evidence accumulation | One search is a narrow sample. `new = 0.3 × this run + 0.7 × everything before` lets evidence pile up: after about seven runs, the first one's weight has fallen below 10%. A single odd query moves a metric 30% of the way, never all of it. | `landscape.update_metrics`, `config.LANDSCAPE_ALPHA` |
| Human-reviewed priors | Data access, MVP difficulty, domain barrier and fit are judgments no count can measure. If runs rewrote them, a quirk in the extraction (for example, a model that misses dataset names) would quietly change your career map. So runs only write suggestions, with their evidence, to `review_queue.md`, and you decide. | `landscape.propose_priors` |
| Append-only evidence log | `observations.jsonl` is never edited. Any score can be recomputed from it, and a bad observation can be found and explained instead of disappearing into an average. | `landscape.append_observation` |
| Like with like | Percentiles only compare counts from the same source, so scoring switches from the PubMed seed to OpenAlex only once every active field has OpenAlex counts. | `landscape.latest_counts`, `refresh_counts.py` |
| Bibliometric caveats | **Vocabulary drift:** "large language model" barely existed in 2021, so its growth from 1 to 7,312 papers measures a new word as much as a new field (the scoring floors the 2021 count at 1, which tames the ratio but does not remove the effect). **Preprint coverage:** PubMed indexes only some medRxiv and bioRxiv preprints, and almost no arXiv. **Keyword proxies:** a query catches papers that use its words, not papers about the topic. **The warrant gap** comes from LLM extractions of the top papers, so it inherits the extractor's errors. | `landscape/CONTEXT.md` §8 |

Private data stays out of git: the owner's fit scores and career lens live in the gitignored
`landscape/private/`, and the live map in `data/landscape/`.

---

## Exercises (each one requires changing code)

1. **Add a third cross-encoder.** Add `"bge_large": "BAAI/bge-reranker-large"` to
   `CROSS_ENCODERS` and `LICENSES` in `config.py`, add it to `SCORERS` and `STYLE` in
   `evaluate.py`, and rerun `score.py --scorers bge_large` and `evaluate.py`. Does the
   larger model beat the smaller one by more than the bootstrap interval?
2. **Swap the LLM and rerun the comparison.** Change `LLM_MODEL` and both prices in
   `config.py` (for example to `claude-sonnet-5-5` at $2 / $10), run
   `score.py --scorers llm --rescore`, then `evaluate.py`. Is the extra cost worth
   the change in AUROC and ECE?
3. **Calibrate a cross-encoder.** In `evaluate.py`, fit
   `sklearn.linear_model.LogisticRegression` on the MiniLM logits against your labels
   (use cross-validation, so you never test on the training labels) and report ECE
   before and after. Why does calibration leave AUROC unchanged?
4. **Cost-aware cascade.** Write a `cascade` scorer: run BGE on everything, send only
   the middle band (say 0.2-0.8) to the LLM, and keep BGE's score elsewhere. Add it to
   the report with its real cost per 1,000 papers.
5. **Close the loop on the checker.** When the verifier flags claims, send only those claims
   back to the writer with the checker's reasons, re-check the rewrite, and stop after one
   round. Compare the share of supported claims before and after.
6. **Stream the review.** In `pipeline.py`, switch `write_review` to the streaming API and
   show the review appearing section by section in the UI instead of waiting for all of it.
7. **Co-citation edges.** Bibliographic coupling looks backward (shared references).
   Add `co_citation` edges (two papers cited together by the same later paper) using
   `/paper/{id}/citations` in `fetch.py`, draw them as a third toggle in
   `templates/graph.html`, and compare which communities each edge type produces.
8. **Concentration.** Add `concentration_hhi` to `landscape.observe_run`: fetch author
   institutions for the 20 most-cited papers from OpenAlex, and compute the
   Herfindahl-Hirschman index of institution shares. Does it separate "crowded by many
   labs" from "dominated by three"?
