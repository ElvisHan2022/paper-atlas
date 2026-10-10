"""All knobs in one place: models, thresholds, prices, paths."""
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
# override=True: the project's .env wins over an old key set elsewhere in Windows/macOS.
load_dotenv(ROOT / ".env", override=True)

# ---- paths ----
DATA_DIR = ROOT / "data"
CACHE_DIR = ROOT / "cache"
PDF_DIR = CACHE_DIR / "pdfs"
OUTPUT_DIR = ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
NOTES_DIR = OUTPUT_DIR / "notes"
DB_PATH = DATA_DIR / "atlas.db"
GRAPH_TEMPLATE = ROOT / "templates" / "graph.html"

# ---- Semantic Scholar ----
S2_BASE = "https://api.semanticscholar.org/graph/v1"
S2_FIELDS = ("paperId,title,abstract,year,venue,citationCount,externalIds,openAccessPdf,"
             "authors,publicationTypes")
S2_PAGE_SIZE = 100          # max allowed by /paper/search
S2_SECONDS_PER_REQUEST = 1.0  # polite limit without a key
S2_SECONDS_WITH_KEY = 0.2

# ---- scorers ----
CROSS_ENCODERS = {
    "minilm": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "bge": "BAAI/bge-reranker-base",
}
LICENSES = {
    "minilm": "Apache-2.0 (runs locally)",
    "bge": "MIT (runs locally)",
    "llm": "Provider terms; paper text leaves the machine",
}
LLM_MODEL = "claude-haiku-4-5"      # fast, inexpensive judge + extractor
LLM_MAX_TOKENS_SCORE = 300
LLM_MAX_TOKENS_EXTRACT = 2000
# USD per million tokens for LLM_MODEL. Update if you change the model.
PRICE_INPUT_PER_M = 1.00
PRICE_OUTPUT_PER_M = 5.00

DEFAULT_CRITERIA = (
    "Empirical studies that evaluate large language models on health or clinical tasks, "
    "including benchmarks, robustness, agreement with human raters, or failure analysis."
)

# ---- thresholds ----
LLM_RELEVANT_CUTOFF = 0.75   # (4 - 1) / 4: rubric score >= 4
COUPLING_MIN_SHARED = 2      # keep coupling edges with >= 2 shared references
PDF_MAX_CHARS = 40_000
EVAL_TOP_K = 20              # precision@K
ECE_BINS = 10
BOOTSTRAP_RESAMPLES = 1000
RANDOM_SEED = 42

# ---- web app (app.py / pipeline.py) ----
WEB_HOST = "127.0.0.1"
WEB_PORT = 8000
WEB_LLM_SHORTLIST = 30      # only the top cross-encoder hits go to the LLM judge
WEB_TOP_N = 10              # papers shown, extracted, and reviewed
WEB_WORKERS = 6             # parallel LLM calls
REVIEW_MODEL = LLM_MODEL    # model that writes the systematic review
VERIFY_MODEL = LLM_MODEL    # independent checker; a stronger model here is a cheap upgrade
LLM_MAX_TOKENS_REVIEW = 4000
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # bi-encoder for topic clusters
CLUSTER_K_RANGE = (2, 5)    # try this many topic clusters, keep the best silhouette
RUNS_DIR = OUTPUT_DIR / "runs"

# Where to search, and how many hits to take from each before merging duplicates.
SOURCES = {"pubmed": "PubMed", "openalex": "OpenAlex (all journals)",
           "preprints": "medRxiv / bioRxiv", "semantic_scholar": "Semantic Scholar",
           "arxiv": "arXiv", "clinicaltrials": "ClinicalTrials.gov"}
TRIAL_SOURCE = "clinicaltrials"   # not ranked as papers; feeds the Clinical trials tab
SOURCE_LIMITS = {"semantic_scholar": 80, "openalex": 80, "pubmed": 60, "preprints": 40, "arxiv": 25}
TRIALS_LIMIT = 10           # ClinicalTrials.gov studies shown in the Clinical trials tab

# Venue families, matched as whole words against the lowercase venue name (first match wins).
VENUE_TYPES = [
    ("Preprint", ["arxiv", "medrxiv", "biorxiv", "ssrn", "research square"]),
    ("Clinical journal", [
        "n engl j med", "new england journal of medicine", "nejm ai", "lancet", "jama",
        "bmj", "ann intern med", "annals of internal medicine", "nat med", "nature medicine",
        "plos med", "circulation", "j clin oncol", "crit care med", "chest", "radiology",
        "mayo clin proc", "jama netw open"]),
    ("Informatics journal", [
        "j am med inform assoc", "jamia", "journal of the american medical informatics association",
        "j biomed inform", "journal of biomedical informatics", "npj digit med",
        "npj digital medicine", "jmir", "j med internet res", "bmc med inform decis mak",
        "int j med inform", "international journal of medical informatics",
        "artif intell med", "artificial intelligence in medicine", "appl clin inform",
        "ieee j biomed health inform", "amia"]),
    ("ML / AI venue", [
        "neurips", "neural information processing systems", "icml", "iclr", "aaai", "ijcai",
        "acl", "emnlp", "naacl", "coling", "findings", "ml4h", "machine learning for health",
        "chil", "mlhc", "kdd", "nat mach intell", "nature machine intelligence", "tmlr",
        "jmlr", "cvpr", "miccai"]),
    # Broad science journals last, so "Nature Medicine" is caught above as clinical.
    ("General science journal", [
        "^nature$", "^science$", "^cell$", "nat commun", "nature communications", "sci adv",
        "science advances", "sci rep", "scientific reports", "proc natl acad sci", "pnas",
        "plos one"]),
]

# Evidence lenses: different journals reward different kinds of work. The lens tells the
# LLM judge what to value on top of topical relevance, and gets its own 1-5 score.
LENSES = {
    "balanced": {"label": "Balanced", "summary": "Rank by relevance alone", "criteria": None},
    "clinical": {
        "label": "Clinical impact", "summary": "Deployment, trials, patient outcomes",
        "criteria": ("Favor evidence that changes care: prospective or randomized studies, external "
                     "or multi-site validation, real-world deployment, workflow integration, and "
                     "patient or clinician outcomes. This is what NEJM, Lancet, JAMA, Nature "
                     "Medicine and NEJM AI reward.")},
    "methods": {
        "label": "Methods rigor", "summary": "Strong evaluation design",
        "criteria": ("Favor careful evaluation: clear reference standards, strong baselines, "
                     "appropriate metrics with uncertainty, external or temporal validation, "
                     "subgroup and bias analysis, calibration, and shared code or data. This is "
                     "what JAMIA, npj Digital Medicine and JBI reward.")},
    "novelty": {
        "label": "Novelty", "summary": "New methods, models, benchmarks",
        "criteria": ("Favor new methods, models, benchmarks or datasets that clearly move the "
                     "state of the art, with convincing comparisons to prior work. This is what "
                     "NeurIPS, ICML, ICLR, ACL, ML4H and CHIL reward.")},
}
LENS_WEIGHT = 0.4            # final LLM score = 0.6 * relevance + 0.4 * lens fit

# ---- demo topic ----
DEMO_TOPIC = "LLM evaluation and reliability for clinical and health text"

# ---- landscape memory (Milestone 7): which fields are crowded, which are open ----
LANDSCAPE_SEED_DIR = ROOT / "landscape"                  # tracked seed, no personal data
LANDSCAPE_PRIVATE_DIR = LANDSCAPE_SEED_DIR / "private"   # gitignored: owner.md, fit.csv
LANDSCAPE_DIR = DATA_DIR / "landscape"                   # live copy every run updates
LANDSCAPE_ALPHA = 0.3          # weight of a new observation in the running quality average
LANDSCAPE_MATCH_THRESHOLD = 0.5  # share of the topic's words a field's query must cover
LANDSCAPE_DEFAULT_FIT = 2      # fit (0-3) used when landscape/private/fit.csv is absent
LANDSCAPE_DEFAULT_PRIORS = {"data_access": 2, "mvp_simplicity": 2, "domain_access": 2}
LANDSCAPE_MIN_EXTRACTED = 3   # fewer extracted papers than this say nothing about quality
LANDSCAPE_OPEN_DATA_MIN = 0.2  # below this open-data rate, suggest a lower data_access prior
LANDSCAPE_YEARS = (2021, 2026)  # OpenAlex count window
LANDSCAPE_COUNTS_MAX_AGE_DAYS = 7  # refresh_counts.py refetches cached counts older than this
# Same terms as each field's queries.pubmed_ai_suffix; decide whether a paper is "AI".
LANDSCAPE_AI_TERMS = ["machine learning", "deep learning", "artificial intelligence",
                      "large language model", "neural network"]
# Words too common in health topics to say which field a topic belongs to.
LANDSCAPE_GENERIC_WORDS = {
    "clinical", "health", "healthcare", "medical", "patient", "model", "study", "data",
    "using", "based", "approach", "method", "analysis", "application", "research", "review",
    "evaluation", "new", "toward", "towards", "via", "use"}
# The eight top journals the landscape counts AI papers in (name -> spellings seen in venues).
TOP_JOURNALS = {
    "Nature": ["nature"],
    "Nature Medicine": ["nature medicine", "nat med"],
    "NEJM": ["new england journal of medicine", "n engl j med", "nejm"],
    "Lancet": ["lancet", "the lancet"],
    "JAMA": ["jama", "journal of the american medical association"],
    "Lancet Digital Health": ["lancet digital health", "the lancet digital health",
                              "lancet digit health"],
    "npj Digital Medicine": ["npj digital medicine", "npj digit med"],
    "NEJM AI": ["nejm ai"],
}
# A paper "uses open data" when its extracted data source names one of these.
OPEN_DATA_NAMES = [
    "mimic", "eicu", "physionet", "uk biobank", "all of us", "nhanes", "ptb-xl", "chexpert",
    "adni", "nacc", "globem", "vitaldb", "ohiot1dm", "seer", "tcga", "medqa", "pubmedqa",
    "healthbench", "kaggle", "publicly available", "public dataset", "open dataset",
    "open-source dataset", "openly available"]
# A stated limitation is about transport when it mentions one of these.
TRANSPORT_TERMS = [
    "generaliz", "generalis", "external validation", "single-center", "single center",
    "single-centre", "single centre", "single-site", "single site", "single institution",
    "subgroup", "transport", "other populations", "other settings", "other hospitals",
    "distribution shift", "dataset shift", "diverse populations"]
