"""Search PubMed, OpenAlex and arXiv alongside Semantic Scholar, then merge the results.

Every source is turned into the same shape Semantic Scholar uses (paperId, title, abstract,
year, venue, citationCount, externalIds, openAccessPdf, authors), plus three extras:
`source`, `url`, and `publicationTypes`. That way the rest of the pipeline doesn't care
where a paper came from.

PubMed:  https://www.ncbi.nlm.nih.gov/books/NBK25501/  (E-utilities, free, ~3 requests/s)
arXiv:   https://info.arxiv.org/help/api/user-manual.html  (Atom XML, 1 request per 3 s)
OpenAlex: https://docs.openalex.org  (free, open index of ~250M works across all journals)
"""
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import xml.etree.ElementTree as ET

import config
import fetch

PUBMED = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
ARXIV = "https://export.arxiv.org/api/query"
ATOM = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
STOPWORDS = set("a an and are as at by for from in into is of on or the to with using via "
                "based toward towards study studies".split())


def _text(el):
    """All text inside an XML element, including text inside <i>, <sup>, etc."""
    return re.sub(r"\s+", " ", "".join(el.itertext())).strip() if el is not None else ""


# ---------------- PubMed ----------------

def pubmed_search(query, n):
    """Top-n PubMed hits by relevance, with abstracts. Two calls: esearch, then efetch."""
    key = os.getenv("NCBI_API_KEY")
    wait = 0.11 if key else 0.34          # NCBI allows 10/s with a key, 3/s without
    common = {"db": "pubmed", "tool": "paper-atlas", **({"api_key": key} if key else {})}
    found = fetch.cached_get(f"{PUBMED}/esearch.fcgi",
                             {**common, "term": query, "retmax": n, "sort": "relevance",
                              "retmode": "json"}, "json", wait)
    ids = ((found or {}).get("esearchresult") or {}).get("idlist") or []
    if not ids:
        return []
    xml = fetch.cached_get(f"{PUBMED}/efetch.fcgi",
                           {**common, "id": ",".join(ids), "retmode": "xml",
                            "rettype": "abstract"}, "text", wait)
    papers = parse_pubmed(xml or "")
    order = {pmid: i for i, pmid in enumerate(ids)}  # efetch doesn't keep relevance order
    return sorted(papers, key=lambda p: order.get(p["paperId"][5:], len(order)))


def parse_pubmed(xml):
    out = []
    for art in ET.fromstring(xml).iter("PubmedArticle") if xml.strip() else []:
        cit = art.find("MedlineCitation")
        a = cit.find("Article")
        pmid = _text(cit.find("PMID"))
        abstract = " ".join(
            (f"{t.get('Label').title()}: " if t.get("Label") else "") + _text(t)
            for t in a.findall("Abstract/AbstractText"))
        journal = a.find("Journal")
        year = _text(journal.find("JournalIssue/PubDate/Year")) or \
            _text(journal.find("JournalIssue/PubDate/MedlineDate"))[:4] or \
            _text(a.find("ArticleDate/Year"))
        ids = {i.get("IdType"): _text(i) for i in art.findall("PubmedData/ArticleIdList/ArticleId")}
        authors = []
        for au in a.findall("AuthorList/Author"):
            name = " ".join(x for x in (_text(au.find("ForeName")), _text(au.find("LastName"))) if x)
            authors.append({"name": name or _text(au.find("CollectiveName"))})
        pmc = ids.get("pmc")
        out.append({
            "paperId": f"PMID:{pmid}",
            "title": _text(a.find("ArticleTitle")).rstrip("."),
            "abstract": abstract or None,
            "year": int(year) if year[:4].isdigit() else None,
            "venue": _text(journal.find("ISOAbbreviation")) or _text(journal.find("Title")),
            "citationCount": None,  # PubMed doesn't count citations
            "externalIds": {"DOI": ids.get("doi"), "PubMed": pmid, "PubMedCentral": pmc},
            "openAccessPdf": {"url": f"https://europepmc.org/articles/{pmc}?pdf=render"} if pmc else None,
            "authors": [x for x in authors if x["name"]],
            "publicationTypes": [_text(t) for t in a.findall("PublicationTypeList/PublicationType")],
            "source": "pubmed",
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        })
    return out


# ---------------- arXiv ----------------

def arxiv_query(query, joiner):
    words = [w for w in re.findall(r"[A-Za-z0-9\-]+", query.lower()) if w not in STOPWORDS]
    return f" {joiner} ".join(f"all:{w}" for w in words[:6]) or f"all:{query}"


def arxiv_search(query, n):
    """Top-n arXiv hits. Tries 'all words' first, then 'any word' if that finds too little."""
    papers = []
    for joiner in ("AND", "OR"):
        xml = fetch.cached_get(ARXIV, {"search_query": arxiv_query(query, joiner),
                                       "start": 0, "max_results": n, "sortBy": "relevance"},
                               "text", 3.0)
        papers = parse_arxiv(xml or "")
        if len(papers) >= min(5, n):
            break
    return papers


def parse_arxiv(xml):
    out = []
    root = ET.fromstring(xml) if xml.strip() else None
    for e in root.findall("a:entry", ATOM) if root is not None else []:
        raw_id = _text(e.find("a:id", ATOM)).rsplit("/abs/", 1)[-1]
        arxiv_id = re.sub(r"v\d+$", "", raw_id)
        if not arxiv_id or not _text(e.find("a:title", ATOM)):
            continue  # arXiv returns a single empty "error" entry for bad queries
        journal = _text(e.find("arxiv:journal_ref", ATOM))
        published = _text(e.find("a:published", ATOM))
        out.append({
            "paperId": f"ARXIV:{arxiv_id}",
            "title": _text(e.find("a:title", ATOM)),
            "abstract": _text(e.find("a:summary", ATOM)) or None,
            "year": int(published[:4]) if published[:4].isdigit() else None,
            "venue": journal or "arXiv",
            "citationCount": None,
            "externalIds": {"DOI": _text(e.find("arxiv:doi", ATOM)) or None, "ArXiv": arxiv_id},
            "openAccessPdf": {"url": f"https://arxiv.org/pdf/{arxiv_id}"},
            "authors": [{"name": _text(au.find("a:name", ATOM))} for au in e.findall("a:author", ATOM)],
            "publicationTypes": ["Preprint"],
            "source": "arxiv",
            "url": f"https://arxiv.org/abs/{arxiv_id}",
        })
    return out


# ---------------- OpenAlex (every journal) ----------------

OPENALEX = "https://api.openalex.org/works"
OPENALEX_FIELDS = ("id,doi,display_name,publication_year,type,cited_by_count,primary_location,"
                   "best_oa_location,authorships,abstract_inverted_index")


def openalex_search(query, n):
    """Top-n OpenAlex works with abstracts. OpenAlex indexes ~250M works across journals."""
    params = {"search": query, "filter": "has_abstract:true", "per-page": min(n, 200),
              "select": OPENALEX_FIELDS}
    if os.getenv("OPENALEX_EMAIL"):      # optional: joins OpenAlex's faster "polite pool"
        params["mailto"] = os.getenv("OPENALEX_EMAIL")
    if os.getenv("OPENALEX_API_KEY"):
        params["api_key"] = os.getenv("OPENALEX_API_KEY")
    data = fetch.cached_get(OPENALEX, params, "json", 0.2)
    return parse_openalex(data or {})


def rebuild_abstract(inverted):
    """OpenAlex stores abstracts as {word: [positions]}; put the words back in order."""
    if not inverted:
        return None
    slots = sorted((pos, word) for word, positions in inverted.items() for pos in positions)
    return " ".join(word for _, word in slots)


def parse_openalex(data):
    out = []
    for w in data.get("results") or []:
        work_id = (w.get("id") or "").rsplit("/", 1)[-1]
        if not work_id or not w.get("display_name"):
            continue
        source = (w.get("primary_location") or {}).get("source") or {}
        pdf = (w.get("best_oa_location") or {}).get("pdf_url") or \
            (w.get("primary_location") or {}).get("pdf_url")
        doi = (w.get("doi") or "").replace("https://doi.org/", "") or None
        kind = {"review": "Review", "preprint": "Preprint"}.get(w.get("type"))
        out.append({
            "paperId": f"OPENALEX:{work_id}",
            "title": w["display_name"],
            "abstract": rebuild_abstract(w.get("abstract_inverted_index")),
            "year": w.get("publication_year"),
            "venue": source.get("display_name") or "",
            "citationCount": w.get("cited_by_count"),
            "externalIds": {"DOI": doi, "OpenAlex": work_id},
            "openAccessPdf": {"url": pdf} if pdf else None,
            "authors": [{"name": (a.get("author") or {}).get("display_name")}
                        for a in w.get("authorships") or [] if (a.get("author") or {}).get("display_name")],
            "publicationTypes": [kind] if kind else [],
            "source": "openalex",
            "url": f"https://doi.org/{doi}" if doi else f"https://openalex.org/{work_id}",
        })
    return out


# ---------------- Semantic Scholar + merging ----------------

def s2_search(query, n):
    papers = fetch.search_papers(query, n)
    for p in papers:
        p["source"] = "semantic_scholar"
        p["url"] = f"https://www.semanticscholar.org/paper/{p['paperId']}"
        p["publicationTypes"] = p.get("publicationTypes") or []
    return papers


SEARCHERS = {"semantic_scholar": s2_search, "openalex": openalex_search,
             "pubmed": pubmed_search, "arxiv": arxiv_search}


def dedupe_key(p):
    """Same DOI, or failing that the same title with punctuation and case removed."""
    doi = ((p.get("externalIds") or {}).get("DOI") or "").lower().strip()
    return "doi:" + doi if doi else "title:" + re.sub(r"[^a-z0-9]", "", (p["title"] or "").lower())[:150]


def merge(results_by_source):
    """Combine lists from several sources, keeping one record per paper.

    Earlier sources win ties (Semantic Scholar and OpenAlex have citation counts); later
    ones fill gaps like a missing abstract, PDF link, venue, or publication types.
    `sources` lists everywhere the paper was found.
    """
    merged, order = {}, []
    preferred = ["semantic_scholar", "openalex", "pubmed", "arxiv"]
    for source in preferred + [s for s in results_by_source if s not in preferred]:
        for p in results_by_source.get(source, []):
            if not p.get("title") or not p.get("abstract"):
                continue
            key = dedupe_key(p)
            if key not in merged:
                merged[key] = {**p, "sources": [source]}
                order.append(key)
                continue
            base = merged[key]
            base["sources"].append(source)
            for field in ("abstract", "year", "venue", "openAccessPdf", "citationCount"):
                if not base.get(field) and p.get(field):
                    base[field] = p[field]
            if base.get("venue") == "arXiv" and p.get("venue") not in (None, "", "arXiv"):
                base["venue"] = p["venue"]
            base["publicationTypes"] = list(dict.fromkeys(
                (base.get("publicationTypes") or []) + (p.get("publicationTypes") or [])))
            ext = base.setdefault("externalIds", {}) or {}
            for k, v in (p.get("externalIds") or {}).items():
                ext.setdefault(k, v)
    return [merged[k] for k in order]


def search_all(query, sources, limits=None):
    """Search the chosen sources in parallel; one failing source is reported, not fatal.

    The sources are independent of each other, so this is the one place where fanning out
    is pure gain: total time is the slowest source, not the sum of all of them.
    """
    limits = limits or config.SOURCE_LIMITS
    results, errors = {}, {}
    with ThreadPoolExecutor(max_workers=len(sources) or 1) as pool:
        futures = {pool.submit(SEARCHERS[s], query, limits[s]): s for s in sources}
        for future in as_completed(futures):
            source = futures[future]
            try:
                results[source] = future.result()
            except Exception as e:
                errors[source] = str(e)
    return merge(results), {s: len(v) for s, v in results.items()}, errors


# ---------------- what kind of paper is this? ----------------

def venue_type(venue, sources=(), publication_types=()):
    """Clinical journal, informatics journal, ML/AI venue, preprint, or other journal."""
    v = (venue or "").lower()
    for label, patterns in config.VENUE_TYPES:
        if any(re.search(rf"\b{pat}\b", v) for pat in patterns):
            return label
    if not v and ("arxiv" in sources or "Preprint" in publication_types):
        return "Preprint"
    return "Journal" if v else "Unknown venue"


EVIDENCE_TAGS = [  # (publication type as PubMed or Semantic Scholar spells it, our tag)
    ("Meta-Analysis", "Meta-analysis"), ("MetaAnalysis", "Meta-analysis"),
    ("Systematic Review", "Systematic review"),
    ("Randomized Controlled Trial", "RCT"),
    ("Clinical Trial", "Clinical trial"), ("ClinicalTrial", "Clinical trial"),
    ("Observational Study", "Observational"), ("Comparative Study", "Comparative study"),
    ("Validation Study", "Validation study"), ("Evaluation Study", "Evaluation study"),
    ("Multicenter Study", "Multicenter"),
    ("Review", "Review"), ("Case Reports", "Case report"), ("CaseReport", "Case report"),
    ("Preprint", "Preprint"),
]


def evidence_tags(publication_types):
    """Healthcare evidence labels, strongest first, without duplicates."""
    types = set(publication_types or [])
    return list(dict.fromkeys(tag for raw, tag in EVIDENCE_TAGS if raw in types))
