# paper-atlas roadmap

Last updated: 2026-10-09. Written after the first real end-to-end search
("wearable foundation model EHR twin": 183 candidates, 30 judged, $0.147).

## Goal

Use paper-atlas to:

1. **Learn a new field fast**: what it is, the terms that matter, what the evidence says.
2. **Find a place to contribute**: the pain points a master's student could realistically address.
3. **Judge feasibility**: the knowledge and skill floor to get into the field, scored against
   the user's own background.

---

## 1. How the app works today

1. **Search.** PubMed, OpenAlex, medRxiv/bioRxiv (via Europe PMC), Semantic Scholar and arXiv
   are searched in parallel with the raw query, and duplicates are merged by DOI, arXiv id or
   title. ClinicalTrials.gov is searched alongside and shown in its own tab.
2. **Local ranking.** Two cross-encoders (MiniLM, BGE) read "topic + criteria" together with
   "title + abstract" and output a logit, squashed to 0–1 with a sigmoid and then averaged.
3. **LLM judge.** The top 30 by that average go to Claude Haiku, which scores relevance 1–5 and,
   with a lens, lens fit 1–5. Both are normalized as (s − 1) / 4.
4. **Rank.** LLM score (60% relevance + 40% lens fit when a lens is on), with the cross-encoder
   average breaking ties. The top 10 are kept.
5. **Synthesis.** The top 10 get a structured extraction (full text when an open PDF exists),
   then a five-section review, then an independent checker audits every cited claim.
6. **Topic clusters.** A bi-encoder (`all-MiniLM-L6-v2`) embeds every candidate, k-means groups
   them (k chosen by silhouette), and Haiku names each group.

## 2. Red flags (most important first)

1. **Nothing is measured.** No accuracy numbers exist for any scorer on a real topic. The scores
   look precise but are unvalidated. Fix this before trusting the strategy advice.
2. **Retrieval is the ceiling.** One raw query, keyword APIs, newest papers first. Foundational
   work (highly cited reviews, the papers that defined a field) is mostly missed, and that is
   what learning a field needs.
3. **The judge doesn't separate papers.** All top-10 papers scored 5/5, so the final order among
   them came from two small, general-purpose local models.
4. **No quality filter.** Low-tier venues and likely paper-mill output rank high in fast-moving
   topics such as "AI digital twins".
5. **No diversity.** In one run all ten papers came from a single topic cluster.
6. **"Systematic review" overclaims.** There is no protocol, no exhaustive search and no
   risk-of-bias assessment. Rename the tab to "Evidence synthesis" or "Rapid review".
7. **The writer and checker see different evidence.** The writer sees full-text extractions;
   the checker sees only abstracts and evidence quotes. Some "unsupported" verdicts are really
   the checker's blind spot.
8. **The reviewer's own conclusions are scored as failures.** Claims like "no paper reports
   cost-effectiveness" have no citation, so the checker marks them unsupported. They need their
   own neutral verdict ("reviewer's inference").
9. **Feasibility advice has no user profile and no hard data.** Without the user's background
   and computed field facts, "how hard is this to enter?" is generic opinion.
10. **Cost is invisible per step.** Only the total is logged.
11. **Trial search often finds 0.** Technical queries are a poor fit for ClinicalTrials.gov's
    keyword search.
12. **README understates cost.** It says "a few cents"; real runs are about 14¢.

## 3. The encoders

| Model | What it is | Problem | Candidate replacement |
|---|---|---|---|
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | ~22M-parameter reranker trained on Bing web queries | Trained on short web queries, not long biomedical criteria; disagrees sharply with BGE (0.82 vs 0.24 on one paper) | `ncbi/MedCPT-Cross-Encoder` (trained on PubMed search logs) |
| `BAAI/bge-reranker-base` | ~280M-parameter general reranker | Not biomedical | `BAAI/bge-reranker-v2-m3` (stronger, ~3× slower on CPU) |
| `sentence-transformers/all-MiniLM-L6-v2` | General sentence embeddings (topic clusters) | Not tuned for scientific text | `allenai/specter2` or the MedCPT article encoder |

**Rule:** switch only if `evaluate.py` shows a gain on labelled data. Check each model's
license before adopting it.

## 4. Cost: lower *and* better

Today's ~14¢ is mostly output tokens ($5/M on Haiku vs $1/M for input). These are estimates,
to be confirmed by per-step cost logging.

| Change | Cost | Quality |
|---|---|---|
| Log cost per step (in the run log and the rubric tab) | 0 | Lets us measure every change below |
| The writer emits a numbered claim list; the checker returns only `claim id, verdict, ≤15-word reason` | ↓↓ | Same or better |
| Extraction reads only the Methods and Results sections (~12k characters) and caps its output | ↓ | Same |
| A better local ranker means only the top 20 go to the judge | ↓ | ↑ if the ranker measures better |
| One call generates 3–4 search queries, including MeSH terms for PubMed | +0.3¢ | ↑↑ recall |
| The review and the TLDR use `claude-sonnet-5-5` ($2/$10 per M); bulk work stays on Haiku | +4–6¢ | ↑↑ synthesis quality |

Target modes: **Quick** (~8¢, Haiku throughout) and **Deep** (~12–15¢, Sonnet for synthesis).

## 5. New: the TLDR tab (first tab, consultant-style, top-down)

Every statement carries either a paper citation or a **judgment** tag with a confidence level.

1. **Bottom line** (3 sentences): what the field is, whether to enter, and the best entry point.
2. **Briefer**: the field in plain words; the 10–15 terms to know with one-line definitions;
   key tools, venues and groups.
3. **Body of evidence**: what is settled, what is contested, how mature the field is.
4. **Pain points ranked for a master's student**: impact × tractability × data access, each with
   evidence.
5. **Entry feasibility scorecard** (1–5 each, against the user's profile): math floor, domain
   floor, tooling and data access, compute, crowding, industry concentration, plus an overall
   score. Each score shows the numbers behind it.
6. **Risks and limits of this analysis.**
7. **Next steps**: 3 papers to read first, 2 skills to learn, one 4–6 week starter project.

**Inputs it needs:**
- A **"My background"** profile, stored in the browser and sent with each search.
- **Computed field facts**, not model opinion:
  - publication growth per year (already computed for the clusters),
  - industry vs academic author share (OpenAlex institution types),
  - venue mix,
  - share of papers with code or data,
  - whether the key tools are free or licensed.
- The **top 30 abstracts**, not just the top 10.
- **Sonnet**, not Haiku, for this one call.

### Example of the format: "pharmacodynamics pharmacokinetics"

Written from general knowledge to show the format; the real tab cites the papers it finds.

- **Bottom line:** PK/PD and pharmacometrics are applied math and statistics (ODEs and
  mixed-effects models), not structural biology; protein folding isn't required. The math floor
  is moderate and the field is far less crowded than LLM alignment. ML skills are an advantage in
  automating and speeding up model building. Main risk: patient-level data mostly sits inside
  drug companies' trials.
- **Terms:**
  - PK (ADME: absorption, distribution, metabolism, excretion) vs PD (the drug's effect)
  - clearance (CL), volume of distribution (Vd), half-life (t½ ≈ 0.693·Vd/CL), bioavailability (F)
  - Cmax, Tmax, AUC
  - compartment models
  - Emax / Hill model, EC50
  - therapeutic window, steady state
  - population PK (NLME), inter-individual variability, covariates
  - PBPK, QSP
  - exposure–response
  - MIDD (model-informed drug development)
  - TDM / MIPD (therapeutic drug monitoring, model-informed precision dosing)
- **Tools:** NONMEM (licensed industry standard), Monolix, nlmixr2 (free, R), Pumas (Julia).
- **Pain points that suit a master's student:**
  1. Automated popPK model search (an ML/optimization problem).
  2. Hybrid mechanistic + ML models (neural ODEs) that stay interpretable for regulators.
  3. Precision dosing on public data (e.g. vancomycin levels in MIMIC-IV, credentialed access).
- **Feasibility (judgment):** math 3/5, domain 3/5, data access 2/5, crowding 4/5 (uncrowded).
  Good fit for an ML-trained student.

## 6. Build order

1. **Measurement**
   - per-step cost in the run log;
   - in-app 👍/👎 on papers (faster than `label.py`; the same labels feed `evaluate.py`);
   - label ~100 papers on one topic the user knows well.
2. **Better evidence**
   - query expansion (+ MeSH);
   - seed-paper "snowballing" (references and citations of trusted papers);
   - quality signals (venue tier, citations normalized by age, retractions);
   - per-cluster diversity in the top 10;
   - top 30 abstracts to the synthesis;
   - same evidence for writer and checker, plus an "inference" verdict;
   - a short trials query;
   - rename the review tab.
3. **TLDR tab**, with the profile, the computed facts, Quick/Deep modes, and Sonnet for the
   synthesis.
4. **Encoder swap** (MedCPT, SPECTER2), only if step 1's evaluation shows a gain.
5. **Learning from feedback** ("dreaming"): 👍/👎 history becomes stored preferences that shape
   future criteria.

## 7. What the user can provide

| Input | Why | Effort |
|---|---|---|
| **Background profile** (degree, coding, math, stats, domain exposure, hours per week) | Calibrates the feasibility scorecard | 5 min, once |
| **Seed papers per field** (5–15 an expert would call essential: reviews, landmarks) | Recall test ("did the search find them?") and anchors for citation snowballing | 15 min per field |
| **Trusted venues per field** (journals and conferences people respect) | Quality signal against low-tier venues and paper mills | 10 min per field |
| **~100 labels on one familiar topic** | The only way to measure scorer accuracy; labels on an unfamiliar topic are unreliable | 30–45 min, once |
| **Landscape documents** (labs, programs, job postings, advisor notes) | Context for "where can I contribute"; stored as distilled notes, not originals (the repo is public) | As available |
| **Network allowlist for the Claude Code environment** (PubMed, OpenAlex, Europe PMC, Semantic Scholar, arXiv, ClinicalTrials.gov, Hugging Face) | Lets changes be tested against real APIs instead of fixtures | 5 min, once |
| **One exported run** (`outputs/runs/<run>.json`; contains no key) | Debugging against real output | When something looks wrong |
