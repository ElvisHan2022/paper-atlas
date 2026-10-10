# Landscape Context: Seed Memory for paper-atlas

Seeded 2026-10-09. This file is the standing mental model that paper-atlas loads before every run. `fields.json` holds the per-field state, `observations.jsonl` the evidence log, `landscape_score.py` the original scoring logic (`landscape.py` is the live version). The owner's personal lens (career targets, fit scores, personal triggers) lives in the gitignored `landscape/private/` and is appended to this context when present. Full narrative: the owner's Claude Doc "Healthcare AI Research Landscape: Saturation, Entry, and Execution".

## 1. Thesis

The binding constraint in healthcare AI has moved from building models to warranting them. Construction is routine (1,524 FDA AI-enabled devices through March 2026, 76% radiology), while 43% of 521 audited authorizations lacked published clinical validation and only 5% of 519 published LLM evaluations used real patient data (1.2% assessed calibration). The scarce inputs are realistic evaluation, trustworthy labels, post-deployment monitoring, and transport across sites and populations. Treat any new evidence as either confirming or weakening this thesis, and record which.

## 2. The six-layer value chain (where the frontier sits)

1. Data substrate: scale is closed (Epic CoMET 118M patients; Apple WBM 161,855 participants; Google LSM-2 40M hours; no released weights). Open substrates (MIMIC-IV, eICU, All of Us, UK Biobank, GLOBEM) are smaller and DUA-gated.
2. Representation: foundation models; fastest-growing literature; pretraining closed to newcomers, adaptation of open checkpoints (MOTOR, MedGemma, PaPaGei) is open.
3. Task models: most saturated layer (radiology 11,721 AI papers in PubMed 2025).
4. Evaluation and validation: demand exceeds supply (Epic Sepsis Model AUROC 0.63 external, 0.47 excluding post-recognition predictions).
5. Deployment and monitoring: frameworks exist (FDA PCCP Dec 2024; FDA lifecycle draft Jan 2025; Joint Commission and CHAI Sep 2025; FURM); empirical methods thin (112 PubMed papers on dataset shift or drift in 2025).
6. Governance and payment: CMS-HCC v28, ACO LEAD, assurance labs; ambient scribes couple documentation AI to coding intensity.

## 3. Regimes (quadrants of crowding versus momentum, percentile scores)

- Blooming (low crowding, high momentum): dataset shift, target trial emulation, ambient scribes, precision dosing, CGM, real-world evidence.
- Hot and contested (high crowding, high momentum): LLMs in medicine, algorithmic fairness, health foundation models, digital twins, AI implementation.
- Crowded, decelerating (high crowding, low momentum): radiology, sepsis, ECG, Alzheimer disease, MCI, T2D, hypertension, wearables, PPG.
- Dormant whitespace (low crowding, low momentum, large clinical base with little ML): readmission, metformin, value-based care, PK/PD.
- Quiet niche: digital biomarkers, digital phenotyping.

## 4. Current signals

- GO: dataset shift and monitoring, LLM evaluation with calibration and abstention, subgroup fairness as a module of either.
- WAIT: wearables and PPG (until the flagship ships), target trial emulation and RWE, PK/PD and precision dosing, CGM, ambient scribes, digital biomarkers, Alzheimer and MCI (plasma p-tau217 now supplies scalable labels).
- SKIP as entry: radiology, new sepsis models, EHR foundation model pretraining, digital twins, implementation science, T2D, hypertension, metformin as primary fields.

Signals are the owner's judgments. Entry scores in the tracked seed use a neutral fit of 2 for every field; with `landscape/private/fit.csv` present they use the owner's fit.

## 5. Owner's lens

Kept in `landscape/private/owner.md` (gitignored). Import it once with `python landscape.py --import-private <original package>`. Without it, recommendations use only the general rules in this file.

## 6. Metric definitions

- momentum: mean percentile of log total growth (2025/2021), log AI-subset growth, and log share-adjusted arXiv growth where measured.
- crowding: percentile of log AI-subset paper volume in 2025.
- whitespace: percentile of log clinical papers per AI paper.
- evidence_maturity: percentile of cumulative RCTs (2016 to 2025) per 2025 paper.
- feasibility: (data_access + mvp_simplicity + domain_access) / 9 * 100 from priors (each 0 to 3).
- fit: owner's fit (0 to 3, private) / 3 * 100.
- opportunity = 0.45 momentum + 0.35 whitespace + 0.20 (100 - crowding).
- entry_score = 0.40 opportunity + 0.35 feasibility + 0.25 fit.
- Atlas-derived quality metrics (filled from extractions): external_validation_rate, uncertainty_reported_rate, open_data_rate, transport_limitation_rate (stated limitations about generalization, other sites or subgroups), warrant_gap = 1 - external_validation_rate, concentration_hhi over institutions among the top-cited papers, community_velocity of the youngest Louvain community, benchmark_saturation.

## 7. Operating rules for the atlas

1. After each topic run, map the topic to existing fields by query overlap. If none matches, create a provisional field (status "provisional", default priors 2/2/2) and add it to `review_queue.md`.
2. Append one observation per run to `observations.jsonl`. Never edit or delete past observations.
3. Update field metrics as an exponentially weighted average of observations (alpha 0.3), so a single narrow query cannot swing the map.
4. Recompute scores and regimes. If a signal or regime changes, add a `history` entry with the date, the old and new values, and the observation that caused it.
5. Never change priors automatically. Write proposed prior changes, with evidence, to `review_queue.md` for the owner.
6. Keep counts honest: record the source, query string, and date for every number. Keyword growth from near-zero bases (new vocabulary) must be labeled as such.
7. Report in formal, plain prose without em-dashes and without "not X but Y" constructions.

## 8. Known measurement caveats

- PubMed preprint counts cover only NIH-indexed medRxiv and bioRxiv preprints.
- The DataCite arXiv proxy matches abstracts and comment fields, does not fold plurals, and silently matches everything (3,202,707 records) for queries over about 100 characters or opening with a nested parenthesis.
- medRxiv keyword shares come from single-month samples (42 to 122 preprints).
- On the owner's own machine, OpenAlex (`/works?filter=title_and_abstract.search:...&group_by=publication_year`) and Semantic Scholar are reachable and are the preferred sources for refreshing counts.

## 9. PK/PD case summary

Large, flat, evidence-rich field (1,974 to 2,012 PubMed papers 2021 to 2025; 1,112 RCTs since 2016) with 6% ML penetration and no top-journal AI papers 2021 to 2025; about 5 arXiv papers a year. Growing edge: model-informed precision dosing (96 to 208; AI 4 to 25). Barriers: proprietary drug-level data, domain knowledge, licensed tools. Entry play: an open benchmark of ML covariate selection and LLM model-building agents on simulated population PK data, which a 2025 CPT: PSP review explicitly requested.
