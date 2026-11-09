"""
Preparation advisor: readiness score, skill gaps, week-by-week study plan,
free learning resources (incl. interview preparation) and project ideas.
"""
import json
import os
from urllib.parse import quote_plus

import config
from core.roles import ROLE_BY_ID
from core.skillset import effective_skills, weighted_coverage
from nlp.preprocess import normalize_for_matching
from nlp.skills import EFFORT, category_of, _project_blocks, _mentioned

_RES = json.load(open(os.path.join(config.DATA_DIR, "resources.json"), encoding="utf-8"))
_PROJ = json.load(open(os.path.join(config.DATA_DIR, "projects.json"), encoding="utf-8"))
LEVEL_VALUE = {"beginner": 0.4, "intermediate": 0.7, "advanced": 1.0}

FUNDAMENTALS = {   # topic -> (skills that count as evidence, extra keywords)
    "DSA and coding rounds": ({"data structures", "algorithms"}, ["leetcode", "hackerrank", "codechef", "codeforces", "geeksforgeeks"]),
    "OOPs": ({"oops"}, []),
    "DBMS": ({"dbms", "sql"}, []),
    "Operating Systems": ({"operating systems"}, []),
    "Computer Networks": ({"computer networks", "networking"}, []),
    "Aptitude and reasoning": (set(), []),
}
SELF_LEVELS = ["Not started", "Basics", "Comfortable", "Confident"]


def fundamentals_status(profile, self_levels):
    skills, _ = effective_skills(profile)
    low = (profile.get("text") or "").lower()
    out = {}
    for topic, (ev_skills, words) in FUNDAMENTALS.items():
        evidence = bool(ev_skills & set(skills)) or any(w in low for w in words)
        self_v = int(self_levels.get(topic, 0)) / 3
        out[topic] = {"evidence": evidence, "self": int(self_levels.get(topic, 0)),
                      "value": max(self_v, 0.5 if evidence else 0.0)}
    return out


def portfolio_relevance(profile, role):
    blocks = _project_blocks((profile.get("sections") or {}).get("projects", []))
    role_skills = role["must"] + role["nice"]
    relevant = sum(1 for b in blocks if any(_mentioned(s, normalize_for_matching(b)) for s in role_skills))
    return min(relevant / 2, 1.0), relevant


def readiness(profile, role_id, demand=None, resume_score=70, self_levels=None):
    role = ROLE_BY_ID[role_id]
    skills, learning = effective_skills(profile)
    weights = {s: 1 + demand.get(s, 0) for s in role["must"] + role["nice"]} if demand else None
    coverage = weighted_coverage(role["must"], role["nice"], skills, learning, weights)
    have = [s for s in role["must"] + role["nice"] if s in skills]
    depth = sum(LEVEL_VALUE[skills[s]] for s in have) / len(have) if have else 0.0
    portfolio, n_proj = portfolio_relevance(profile, role)
    fund = fundamentals_status(profile, self_levels or {})
    fund_v = sum(f["value"] for f in fund.values()) / len(fund)
    parts = {"coverage": coverage, "depth": depth, "portfolio": portfolio, "resume": resume_score / 100,
             "fundamentals": fund_v}
    w = dict(config.READINESS_WEIGHTS)
    if not role["software"]:            # spread the fundamentals weight across the others
        extra = w.pop("fundamentals")
        parts.pop("fundamentals")
        total = sum(w.values())
        w = {k: v + extra * v / total for k, v in w.items()}
    score = round(100 * sum(w[k] * parts[k] for k in w))
    label = ("Strong candidate" if score >= 80 else "Interview-ready" if score >= 60 else
             "Getting there" if score >= 40 else "Not ready yet")
    names = {"coverage": "Skill coverage", "depth": "Skill depth", "portfolio": "Portfolio",
             "resume": "Resume quality", "fundamentals": "Placement fundamentals"}
    weakest = min(parts, key=parts.get)
    return {"score": score, "label": label, "parts": {names[k]: round(v * 100) for k, v in parts.items()},
            "weights": {names[k]: round(v * 100) for k, v in w.items()}, "weakest": names[weakest],
            "relevant_projects": n_proj, "fundamentals": fund if role["software"] else None}


def skill_gaps(profile, role_id, demand=None, near_miss_skills=()):
    role = ROLE_BY_ID[role_id]
    skills, learning = effective_skills(profile)
    demand = demand or {}
    gaps = []
    for s in role["must"] + role["nice"]:
        must = s in role["must"]
        if s in skills and not (must and skills[s] == "beginner"):
            continue
        status = "level up" if s in skills else "learning" if s in learning else "missing"
        priority = (2 if must else 1) * (1 + demand.get(s, 0)) + (1 if s in near_miss_skills else 0)
        base = EFFORT.get(category_of(s), 15)
        hours = round(base * (0.5 if status == "level up" else 0.6 if status == "learning" else 1.0))
        gaps.append({"skill": s, "must": must, "status": status, "priority": round(priority, 2),
                     "demand": round(100 * demand.get(s, 0)), "hours": hours,
                     "resources": resources_for(s, "intermediate" if status == "level up" else "beginner")})
    return sorted(gaps, key=lambda g: -g["priority"])


def resources_for(skill, level="beginner", limit=3):
    items = _RES["skills"].get(skill, [])
    order = {"beginner": 0, "all": 1, "intermediate": 2, "advanced": 3}
    target = order.get(level, 0)
    items = sorted(items, key=lambda r: abs(order.get(r["level"], 1) - target))
    out = items[:limit]
    if len(out) < 2:     # fallback searches always work
        q = quote_plus(f"learn {skill} tutorial for beginners" if level == "beginner" else f"{skill} intermediate tutorial")
        out.append({"title": f"Video tutorials on {skill}", "provider": "YouTube search",
                    "url": f"https://www.youtube.com/results?search_query={q}", "level": level,
                    "format": "video", "certificate": "none"})
        out.append({"title": f"{skill} on freeCodeCamp", "provider": "freeCodeCamp search",
                    "url": f"https://www.freecodecamp.org/news/search/?query={quote_plus(skill)}", "level": "all",
                    "format": "text", "certificate": "none"})
    return out[:limit]


def interview_topics(role_id):
    role = ROLE_BY_ID[role_id]
    if role["software"]:
        topics = ["DSA and coding rounds", "OOPs", "DBMS", "Operating Systems", "Computer Networks", "Aptitude and reasoning"]
        if role_id == "software-engineer":
            topics.append("System design basics")
    else:
        topics = ["Aptitude and reasoning"]
    if set(role["must"] + role["nice"]) & {"sql"}:
        topics.append("SQL interview practice")
    if set(role["must"]) & {"machine learning", "nlp", "computer vision", "deep learning"}:
        topics.append("ML concepts refresher")
    topics.append("HR and behavioural rounds")
    return [{"topic": t, "resources": _RES["interview"].get(t, [])} for t in dict.fromkeys(topics)]


def project_ideas(role_id, missing):
    ideas = _PROJ.get(role_id, [])
    scored = [dict(p, covers=[s for s in p["skills"] if s in missing]) for p in ideas]
    return sorted(scored, key=lambda p: -len(p["covers"]))


def study_plan(profile, role_id, hours_per_day=2, weeks=8, demand=None, near_miss_skills=(), done=frozenset()):
    role = ROLE_BY_ID[role_id]
    gaps = skill_gaps(profile, role_id, demand, near_miss_skills)
    weekly = max(1, hours_per_day) * 7
    interview_cap = round(weekly * (0.3 if role["software"] else 0.15))
    skill_cap = weekly - interview_cap
    queue = [{"type": "skill", "key": f"skill:{g['skill']}", "title": f"{'Level up' if g['status'] == 'level up' else 'Learn'} {g['skill']}",
              "skill": g["skill"], "remaining": g["hours"], "resources": g["resources"]} for g in gaps]
    ideas = project_ideas(role_id, {g["skill"] for g in gaps})
    if ideas:
        p = ideas[0]
        queue.insert(min(3, len(queue)), {"type": "project", "key": f"project:{p['title']}", "title": f"Build project: {p['title']}",
                                          "skill": None, "remaining": p["weeks"] * 6, "resources": [], "project": p})
    topics = interview_topics(role_id)
    rotating = [t for t in topics if t["topic"] not in ("DSA and coding rounds", "HR and behavioural rounds")]
    plan = []
    for w in range(1, weeks + 1):
        items, cap = [], skill_cap
        while queue and cap > 0:
            q = queue[0]
            alloc = min(q["remaining"], cap)
            seen = any(i["key"] == q["key"] for wk in plan for i in wk["items"])
            items.append({"type": q["type"], "key": q["key"], "title": q["title"] + (" (continue)" if seen else ""),
                          "hours": alloc, "resources": q["resources"], "project": q.get("project"),
                          "done": q["key"] in done})
            q["remaining"] -= alloc
            cap -= alloc
            if q["remaining"] <= 0:
                queue.pop(0)
        if interview_cap:
            if role["software"]:
                dsa = next(t for t in topics if t["topic"] == "DSA and coding rounds")
                items.append({"type": "interview", "key": f"interview:dsa:w{w}", "title": "DSA practice: solve 8-10 problems",
                              "hours": round(interview_cap * 0.6), "resources": dsa["resources"][:2], "done": f"interview:dsa:w{w}" in done})
                rest = interview_cap - round(interview_cap * 0.6)
            else:
                rest = interview_cap
            if w == weeks:
                t = next(t for t in topics if t["topic"] == "HR and behavioural rounds")
            elif rotating:
                t = rotating[(w - 1) % len(rotating)]
            else:
                t = topics[0]
            key = f"interview:{t['topic']}:w{w}"
            items.append({"type": "interview", "key": key, "title": f"Interview prep: {t['topic']}",
                          "hours": rest, "resources": t["resources"][:2], "done": key in done})
        plan.append({"week": w, "items": items, "hours": sum(i["hours"] for i in items)})
    later = [q["title"] for q in queue]
    return {"weeks": plan, "later": later, "gaps": gaps, "weekly_hours": weekly}
