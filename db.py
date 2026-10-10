"""SQLite helpers. One file, plain SQL, so you can open data/atlas.db in any viewer."""
import json
import sqlite3

import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    paper_id TEXT PRIMARY KEY,
    title TEXT, abstract TEXT, year INTEGER, venue TEXT, citation_count INTEGER,
    doi TEXT, arxiv_id TEXT, pdf_url TEXT, authors_json TEXT, source TEXT, topic TEXT,
    publication_types TEXT
);
-- A paper can turn up under several topics; papers.topic keeps the first one,
-- this table keeps all of them.
CREATE TABLE IF NOT EXISTS paper_topics (
    paper_id TEXT, topic TEXT, PRIMARY KEY (paper_id, topic)
);
CREATE TABLE IF NOT EXISTS edges (
    src TEXT, dst TEXT, kind TEXT, weight REAL DEFAULT 1,
    PRIMARY KEY (src, dst, kind)
);
CREATE TABLE IF NOT EXISTS scores (
    paper_id TEXT, topic TEXT, scorer TEXT, score REAL, rationale TEXT,
    latency_ms REAL, input_tokens INTEGER, output_tokens INTEGER, cost_usd REAL,
    PRIMARY KEY (paper_id, topic, scorer)
);
CREATE TABLE IF NOT EXISTS labels (
    paper_id TEXT, topic TEXT, relevant INTEGER, labeled_at TEXT,
    PRIMARY KEY (paper_id, topic)
);
CREATE TABLE IF NOT EXISTS extractions (
    paper_id TEXT, topic TEXT, json TEXT, source_text_kind TEXT,
    PRIMARY KEY (paper_id, topic)
);
"""


def connect(path=None):
    """Open the database (creating folders and tables on first use)."""
    path = path or config.DB_PATH
    if str(path) != ":memory:":
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    # timeout: wait up to 30 s for another writer instead of failing with "database is locked".
    conn = sqlite3.connect(path, timeout=30)
    conn.row_factory = sqlite3.Row  # rows behave like dicts: row["title"]
    if str(path) != ":memory:":
        conn.execute("PRAGMA journal_mode=WAL")  # readers no longer block the writer
    init_schema(conn)
    return conn


def init_schema(conn):
    conn.executescript(SCHEMA)
    # Databases made before publication types were stored lack the column; add it.
    columns = {r[1] for r in conn.execute("PRAGMA table_info(papers)")}
    if "publication_types" not in columns:
        conn.execute("ALTER TABLE papers ADD COLUMN publication_types TEXT")
    conn.commit()


def upsert_paper(conn, p, topic):
    """p is a paper dict in Semantic Scholar's shape (sources.py converts the others)."""
    ext = p.get("externalIds") or {}
    pdf = (p.get("openAccessPdf") or {}).get("url") or None
    authors = [a.get("name") for a in (p.get("authors") or [])]
    conn.execute(
        """INSERT INTO papers (paper_id, title, abstract, year, venue, citation_count, doi,
                                 arxiv_id, pdf_url, authors_json, source, topic,
                                 publication_types)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
           ON CONFLICT(paper_id) DO UPDATE SET
             title=excluded.title, abstract=excluded.abstract, year=excluded.year,
             venue=excluded.venue, citation_count=excluded.citation_count,
             doi=excluded.doi, arxiv_id=excluded.arxiv_id, pdf_url=excluded.pdf_url,
             authors_json=excluded.authors_json,
             publication_types=COALESCE(excluded.publication_types, papers.publication_types)""",
        (p["paperId"], p.get("title"), p.get("abstract"), p.get("year"), p.get("venue"),
         p.get("citationCount") or 0, ext.get("DOI"), ext.get("ArXiv"), pdf,
         json.dumps(authors), p.get("source", "semantic_scholar"), topic,
         json.dumps(p["publicationTypes"]) if p.get("publicationTypes") is not None else None),
    )
    conn.execute("INSERT OR IGNORE INTO paper_topics VALUES (?,?)", (p["paperId"], topic))


def papers_for_topic(conn, topic):
    return conn.execute(
        """SELECT p.* FROM papers p JOIN paper_topics t ON p.paper_id = t.paper_id
           WHERE t.topic = ? ORDER BY p.paper_id""",
        (topic,),
    ).fetchall()


def add_edge(conn, src, dst, kind, weight=1.0):
    conn.execute(
        "INSERT OR REPLACE INTO edges VALUES (?,?,?,?)", (src, dst, kind, weight)
    )


def save_score(conn, paper_id, topic, scorer, score, rationale=None, latency_ms=None,
               input_tokens=None, output_tokens=None, cost_usd=None):
    conn.execute(
        "INSERT OR REPLACE INTO scores VALUES (?,?,?,?,?,?,?,?,?)",
        (paper_id, topic, scorer, score, rationale, latency_ms,
         input_tokens, output_tokens, cost_usd),
    )


def scores_for_topic(conn, topic, scorer):
    """Dict paper_id -> score (None when the scorer failed)."""
    rows = conn.execute(
        "SELECT paper_id, score FROM scores WHERE topic=? AND scorer=?", (topic, scorer)
    ).fetchall()
    return {r["paper_id"]: r["score"] for r in rows}


def save_label(conn, paper_id, topic, relevant, labeled_at):
    conn.execute(
        "INSERT OR REPLACE INTO labels VALUES (?,?,?,?)",
        (paper_id, topic, relevant, labeled_at),
    )


def labels_for_topic(conn, topic):
    """Dict paper_id -> 0/1. Skipped papers (relevant IS NULL) are left out."""
    rows = conn.execute(
        "SELECT paper_id, relevant FROM labels WHERE topic=? AND relevant IS NOT NULL",
        (topic,),
    ).fetchall()
    return {r["paper_id"]: r["relevant"] for r in rows}


def save_extraction(conn, paper_id, topic, data, source_text_kind):
    conn.execute(
        "INSERT OR REPLACE INTO extractions VALUES (?,?,?,?)",
        (paper_id, topic, json.dumps(data), source_text_kind),
    )


def extractions_for_topic(conn, topic):
    """Dict paper_id -> (dict, source_text_kind)."""
    rows = conn.execute(
        "SELECT paper_id, json, source_text_kind FROM extractions WHERE topic=?", (topic,)
    ).fetchall()
    return {r["paper_id"]: (json.loads(r["json"]), r["source_text_kind"]) for r in rows}
