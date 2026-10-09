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

import requests

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


# ---------------- Preprints: medRxiv, bioRxiv, ... via Europe PMC ----------------
# The bioRxiv/medRxiv API itself only lists by date and category (no keyword search), so we
# use Europe PMC, which indexes those servers plus Research Square, SSRN and others.

EUROPEPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"


def preprint_search(query, n):
    """Top-n preprints (medRxiv, bioRxiv, ...) matching the query, with abstracts."""
    data = fetch.cached_get(EUROPEPMC, {"query": f"({query}) AND SRC:PPR", "format": "json",
                                        "resultType": "core", "pageSize": min(n, 100)},
                            "json", 0.2)
    return parse_europepmc(data or {})


def parse_europepmc(data):
    out = []
    for r in ((data.get("resultList") or {}).get("result")) or []:
        if not r.get("id") or not r.get("title"):
            continue
        server = (r.get("bookOrReportDetails") or {}).get("publisher") or "Preprint"
        links = ((r.get("fullTextUrlList") or {}).get("fullTextUrl")) or []
        pdf = next((u["url"] for u in links if u.get("documentStyle") == "pdf" and u.get("url")), None)
        authors = [{"name": " ".join(x for x in (a.get("firstName"), a.get("lastName")) if x)
                    or a.get("fullName")} for a in ((r.get("authorList") or {}).get("author")) or []]
        abstract = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", r.get("abstractText") or "")).strip()
        doi = r.get("doi")
        out.append({
            "paperId": f"PPR:{r['id']}",
            "title": r["title"].rstrip("."),
            "abstract": abstract or None,
            "year": int(r["pubYear"]) if str(r.get("pubYear") or "")[:4].isdigit() else None,
            "venue": server,
            "citationCount": r.get("citedByCount"),
            "externalIds": {"DOI": doi, "EuropePMC": r["id"]},
            "openAccessPdf": {"url": pdf} if pdf else None,
            "authors": [a for a in authors if a["name"]],
            "publicationTypes": ["Preprint"],
            "source": "preprints",
            "url": f"https://doi.org/{doi}" if doi else f"https://europepmc.org/article/PPR/{r['id']}",
        })
    return out


# ---------------- ClinicalTrials.gov (shown in its own tab, not ranked as papers) ----------------

CLINICALTRIALS = "https://clinicaltrials.gov/api/v2/studies"


def trial_search(query, n):
    data = fetch.cached_get(CLINICALTRIALS, {"query.term": query, "pageSize": n,
                                             "format": "json"}, "json", 0.3)
    return parse_trials(data or {})


def _pretty(code):
    """RECRUITING -> Recruiting, NOT_YET_RECRUITING -> Not yet recruiting."""
    return (code or "").replace("_", " ").capitalize()


def parse_trials(data):
    out = []
    for study in data.get("studies") or []:
        ps = study.get("protocolSection") or {}
        ident = ps.get("identificationModule") or {}
        if not ident.get("nctId"):
            continue
        status = ps.get("statusModule") or {}
        design = ps.get("designModule") or {}
        enroll = design.get("enrollmentInfo") or {}
        phases = [p for p in design.get("phases") or [] if p != "NA"]
        out.append({
            "nct_id": ident["nctId"],
            "title": ident.get("briefTitle") or ident.get("officialTitle") or ident["nctId"],
            "status": _pretty(status.get("overallStatus")),
            "study_type": _pretty(design.get("studyType")),
            "phases": [p.replace("PHASE", "Phase ").replace("EARLY_", "Early ") for p in phases],
            "enrollment": enroll.get("count"),
            "enrollment_type": _pretty(enroll.get("type")),
            "start": (status.get("startDateStruct") or {}).get("date"),
            "completion": (status.get("primaryCompletionDateStruct") or {}).get("date"),
            "sponsor": ((ps.get("sponsorCollaboratorsModule") or {}).get("leadSponsor") or {}).get("name"),
            "conditions": (ps.get("conditionsModule") or {}).get("conditions") or [],
            "interventions": [i.get("name") for i in
                              (ps.get("armsInterventionsModule") or {}).get("interventions") or []
                              if i.get("name")],
            "summary": (ps.get("descriptionModule") or {}).get("briefSummary"),
            "has_results": bool(study.get("hasResults")),
            "url": f"https://clinicaltrials.gov/study/{ident['nctId']}",
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
             "pubmed": pubmed_search, "preprints": preprint_search, "arxiv": arxiv_search}


def dedupe_keys(p):
    """Every identity a paper can be matched on: DOI, arXiv id, and (if long enough) title.

    One paper often arrives as several records: the journal version, the arXiv version
    (DOI 10.48550/arXiv.<id>), and an arXiv API record with no DOI at all. Matching on any
    shared key catches all of them. Short titles ("Introduction") are too generic to trust.
    """
    keys = []
    ext = p.get("externalIds") or {}
    doi = (ext.get("DOI") or "").lower().strip()
    if doi:
        keys.append("doi:" + doi)
        arxiv_doi = re.match(r"10\.48550/arxiv\.(.+)", doi)
        if arxiv_doi:
            keys.append("arxiv:" + re.sub(r"v\d+$", "", arxiv_doi.group(1)))
    if ext.get("ArXiv"):
        keys.append("arxiv:" + re.sub(r"v\d+$", "", str(ext["ArXiv"]).lower()))
    title = re.sub(r"[^a-z0-9]", "", (p.get("title") or "").lower())[:150]
    if len(title) >= 20:
        keys.append("title:" + title)
    return keys or ["id:" + str(p.get("paperId"))]


def merge(results_by_source):
    """Combine lists from several sources, keeping one record per paper.

    Earlier sources win ties (Semantic Scholar and OpenAlex have citation counts); later
    ones fill gaps like a missing abstract, PDF link, venue, or publication types. A
    journal venue replaces a preprint one. `sources` lists everywhere the paper was found.
    """
    records, index = [], {}          # index: any dedupe key -> position in records
    preferred = ["semantic_scholar", "openalex", "pubmed", "preprints", "arxiv"]
    for source in preferred + [s for s in results_by_source if s not in preferred]:
        for p in results_by_source.get(source, []):
            if not p.get("title") or not p.get("abstract"):
                continue
            keys = dedupe_keys(p)
            hit = next((index[k] for k in keys if k in index), None)
            if hit is None:
                records.append({**p, "sources": [source]})
                hit = len(records) - 1
            else:
                fill_gaps(records[hit], p, source)
            for k in keys:
                index.setdefault(k, hit)
    return records


def fill_gaps(base, p, source):
    if source not in base["sources"]:
        base["sources"].append(source)
    for field in ("abstract", "year", "venue", "openAccessPdf", "citationCount"):
        if not base.get(field) and p.get(field):
            base[field] = p[field]
    if venue_type(base.get("venue")) in ("Preprint", "Unknown venue") and \
            venue_type(p.get("venue")) not in ("Preprint", "Unknown venue"):
        base["venue"] = p["venue"]       # prefer where it was published over the preprint
    base["publicationTypes"] = list(dict.fromkeys(
        (base.get("publicationTypes") or []) + (p.get("publicationTypes") or [])))
    ext = base.setdefault("externalIds", {}) or {}
    for k, v in (p.get("externalIds") or {}).items():
        if v and not ext.get(k):
            ext[k] = v


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
                errors[source] = explain_source_error(source, e)
    return merge(results), {s: len(v) for s, v in results.items()}, errors


def explain_source_error(source, e):
    """Turn a failed request into a short reason a person can act on."""
    text = str(e)
    if "429" in text or "Gave up" in text:
        reason = "it is rate-limiting requests right now (its free tier is busy)"
        if source == "semantic_scholar":
            reason += "; a free S2_API_KEY in .env avoids this"
        return reason
    if "403" in text:
        return "it refused the request (403)"
    if isinstance(e, (requests.ConnectionError, requests.Timeout)):
        return "no connection to it (network or firewall)"
    return text[:200]


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
