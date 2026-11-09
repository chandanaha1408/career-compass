"""
Market insights from the postings fetched for the user's target role:
skill demand (% of postings mentioning each skill), emerging skills found by
keyphrase extraction, and salary ranges.
"""
import re
import statistics
from collections import Counter

from core.matcher import analyze_job
from nlp.keyphrases import extract
from nlp.skills import CATEGORY, ALIASES, dictionary_skills


def skill_demand(jobs):
    """{skill: share of postings (0-1)} for postings that mention at least one skill."""
    counts, n = Counter(), 0
    for j in jobs:
        found = set(dictionary_skills(j.get("title", "") + ". " + j.get("description", "")))
        if found:
            n += 1
            counts.update(found)
    return {s: c / n for s, c in counts.most_common()} if n else {}


def _norm(s):
    return re.sub(r"[^a-z0-9+#]", "", s.lower())


def emerging_skills(jobs, min_postings=2):
    """Keyphrases that recur across postings but are not in our skill taxonomy.
    Single words are kept only if they look technical (digits, +, #, dots)."""
    known = set(CATEGORY) | set(ALIASES)
    known_norm = {_norm(k) for k in known}
    known_words = {w for k in known for w in re.split(r"[ /]", k)}
    counts = Counter()
    for j in jobs:
        for p in set(extract(j.get("description", ""), top_n=10)):
            words = p.split()
            if _norm(p) in known_norm or any(w in known_words for w in words):
                continue
            if len(words) == 1 and not re.search(r"[\d+#.]", p):
                continue
            counts[p] += 1
    return [(p, c) for p, c in counts.most_common(12) if c >= min_postings]


def salary_summary(jobs):
    vals = []
    for j in jobs:
        lo, hi = j.get("salary_min"), j.get("salary_max")
        if (j.get("salary_currency") or "INR") != "INR":
            continue
        if lo or hi:
            lo, hi = float(lo or hi), float(hi or lo)
            if 50000 <= (lo + hi) / 2 <= 10000000:     # ignore obviously broken values
                vals.append((lo, hi))
    if not vals:
        return None
    mids = [(a + b) / 2 for a, b in vals]
    return {"count": len(vals), "min": min(a for a, _ in vals), "max": max(b for _, b in vals),
            "median": statistics.median(mids)}


def lpa(x):
    return f"{x / 100000:.1f} LPA" if x else "-"
