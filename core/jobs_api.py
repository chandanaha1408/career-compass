"""
Live job search.

Order of sources: JSearch (Google for Jobs results) -> Adzuna (India) -> the
offline sample dataset. Results are normalised to one format, de-duplicated and
cached in SQLite so a demo doesn't use up the free API quota.
Also builds ready-made search links for Naukri, LinkedIn, Indeed and Internshala.
"""
import csv
import hashlib
import json
import logging
import os
import time
from urllib.parse import quote_plus

import requests

import config
from core.llm import scrub

log =logging.getLogger(__name__)
TIMEOUT = 15


def _offline_jobs():
    with open(os.path.join(config.DATA_DIR, "jobs.csv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["source"] = "Sample dataset"
        r["salary_min"] = float(r["salary_min"] or 0) or None
        r["salary_max"] = float(r["salary_max"] or 0) or None
        r["salary_currency"] = "INR"
    return rows


def _jsearch(query, location, job_type, key, provider):
    params = {"query": f"{query} in {location}" if location else query, "page": "1", "num_pages": "1",
              "country": "in", "date_posted": "month"}
    if job_type == "Internship":
        params["employment_types"] = "INTERN"
    elif job_type == "Full-time":
        params["employment_types"] = "FULLTIME"
    if provider == "rapidapi":
        url = "https://jsearch.p.rapidapi.com/search"
        headers = {"X-RapidAPI-Key": key, "X-RapidAPI-Host": "jsearch.p.rapidapi.com"}
    else:
        url = "https://api.openwebninja.com/jsearch/search"
        headers = {"x-api-key": key}
    resp = requests.get(url, headers=headers, params=params, timeout=TIMEOUT)
    resp.raise_for_status()
    out = []
    for j in resp.json().get("data", []) or []:
        loc = ", ".join(x for x in (j.get("job_city"), j.get("job_state")) if x) or j.get("job_country", "")
        smin, smax, period = j.get("job_min_salary"), j.get("job_max_salary"), (j.get("job_salary_period") or "").upper()
        mult = {"MONTH": 12, "HOUR": 2000, "WEEK": 52}.get(period, 1)
        out.append({
            "id": "js-" + str(j.get("job_id", ""))[:40], "title": j.get("job_title", ""),
            "company": j.get("employer_name", ""), "location": "Remote" if j.get("job_is_remote") else loc,
            "job_type": (j.get("job_employment_type") or "").replace("FULLTIME", "Full-time").replace("INTERN", "Internship").title(),
            "experience": "", "description": j.get("job_description", "") or "",
            "url": safe_url(j.get("job_apply_link") or j.get("job_google_link")),
            "posted": (j.get("job_posted_at_datetime_utc") or "")[:10],
            "salary_min": smin * mult if smin else None, "salary_max": smax * mult if smax else None,
            "salary_currency": j.get("job_salary_currency") or "", "source": "Google for Jobs (JSearch)"})
    return out


def _adzuna(query, location, job_type, app_id, app_key):
    params = {"app_id": app_id, "app_key": app_key, "what": query, "results_per_page": 20,
              "content-type": "application/json"}
    if location and location.lower() != "remote":
        params["where"] = location
    resp = requests.get("https://api.adzuna.com/v1/api/jobs/in/search/1", params=params, timeout=TIMEOUT)
    resp.raise_for_status()
    out = []
    for j in resp.json().get("results", []):
        out.append({
            "id": "az-" + str(j.get("id", "")), "title": j.get("title", "").replace("<strong>", "").replace("</strong>", ""),
            "company": (j.get("company") or {}).get("display_name", ""),
            "location": (j.get("location") or {}).get("display_name", ""),
            "job_type": (j.get("contract_time") or "").replace("_", "-").title(), "experience": "",
            "description": (j.get("description") or "").replace("<strong>", "").replace("</strong>", ""),
            "url": safe_url(j.get("redirect_url")), "posted": (j.get("created") or "")[:10],
            "salary_min": j.get("salary_min"), "salary_max": j.get("salary_max"),
            "salary_currency": "INR", "source": "Adzuna (description is a short snippet)"})
    return out


def search(queries, location, job_type, settings, db=None, refresh=False):
    """Return (jobs, notes). `settings` holds API keys; `db` provides caching."""
    notes, jobs = [], []
    jkey, akey, aid = settings.get("jsearch_key"), settings.get("adzuna_key"), settings.get("adzuna_id")
    if not jkey and not (akey and aid):
        notes.append("No job API key added yet, so these are sample postings. Add a free JSearch or Adzuna key in Settings for live jobs.")
        return _offline_jobs(), notes
    for q in queries:
        cache_key = hashlib.md5(json.dumps([q, location, job_type, bool(jkey), bool(akey)]).encode()).hexdigest()
        cached = None if refresh or db is None else db.cache_get(cache_key, config.JOB_CACHE_HOURS)
        if cached is not None:
            jobs += cached
            continue
        found = []
        if jkey:
            try:
                found = _jsearch(q, location, job_type, jkey, settings.get("jsearch_provider", "openwebninja"))
            except Exception as e:
                notes.append(f"JSearch request failed ({_short(e)}).")
        if not found and akey and aid:
            try:
                found = _adzuna(q, location, job_type, aid, akey)
            except Exception as e:
                notes.append(f"Adzuna request failed ({_short(e)}).")
        if found and db is not None:
            db.cache_set(cache_key, found)
        jobs += found
    if not jobs:
        notes.append("No live results came back, so sample postings are shown instead.")
        return _offline_jobs(), notes
    return dedupe(jobs), notes


def _short(e):
    """Short, safe error text. Adzuna puts app_id/app_key in the URL, and requests includes the
    URL in its errors, so the query string is hidden before the message is shown or saved."""
    s = scrub(e)
    return s[:90] + "..." if len(s) > 90 else s


def safe_url(url):
    """Only keep normal web links. A link like 'javascript:...' from an API would run code when clicked."""
    url = (url or "").strip()
    return url if url.lower().startswith(("http://", "https://")) else ""


def dedupe(jobs):
    seen, out = set(), []
    for j in jobs:
        key = (j.get("title", "").lower().strip(), j.get("company", "").lower().strip())
        if key not in seen:
            seen.add(key)
            out.append(j)
    return out


def search_links(role_title, location):
    """Ready-made searches on popular Indian job sites (plain links, no scraping)."""
    q = role_title.split("(")[0].strip()
    loc = location or "India"
    slug = "-".join(q.lower().replace("/", " ").split())
    loc_slug = "-".join(loc.lower().split())
    return [
        ("Naukri", f"https://www.naukri.com/{slug}-jobs-in-{loc_slug}"),
        ("LinkedIn", f"https://www.linkedin.com/jobs/search/?keywords={quote_plus(q)}&location={quote_plus(loc)}"),
        ("Indeed", f"https://in.indeed.com/jobs?q={quote_plus(q)}&l={quote_plus(loc)}"),
        ("Internshala", f"https://www.google.com/search?q={quote_plus('site:internshala.com ' + q)}"),
    ]
