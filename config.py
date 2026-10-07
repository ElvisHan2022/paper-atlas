"""All knobs in one place: models, thresholds, prices, paths."""
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

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
S2_FIELDS = "paperId,title,abstract,year,venue,citationCount,externalIds,openAccessPdf,authors"
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
WEB_CANDIDATES = 100        # papers pulled per search (one Semantic Scholar page)
WEB_LLM_SHORTLIST = 30      # only the top cross-encoder hits go to the LLM judge
WEB_TOP_N = 10              # papers shown, extracted, and reviewed
WEB_WORKERS = 6             # parallel LLM calls
REVIEW_MODEL = LLM_MODEL    # model that writes the systematic review
LLM_MAX_TOKENS_REVIEW = 4000
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # bi-encoder for topic clusters
CLUSTER_K_RANGE = (2, 5)    # try this many topic clusters, keep the best silhouette
RUNS_DIR = OUTPUT_DIR / "runs"

# ---- demo topic ----
DEMO_TOPIC = "LLM evaluation and reliability for clinical and health text"
