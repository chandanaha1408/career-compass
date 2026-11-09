"""
Two-stage job matching with explanations.

Stage 1 (fast, all jobs): bi-encoder embeddings - the resume and each job are
        embedded separately and compared with cosine similarity.
Stage 2 (accurate, top N): a cross-encoder reads the job requirements and the
        resume together and scores relevance; blended 50/50 with stage 1.

Fit score = 40% semantic + 35% skills + 15% keywords (TF-IDF) + 10% level/education
Without neural models, semantic similarity falls back to character n-gram TF-IDF.
"""
import hashlib

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import config
from core.roles import resume_vector
from core.skillset import effective_skills, group_coverage, profile_text, credit, group_credit
from nlp import jd_analyzer, models
from nlp.skills import dictionary_skills

_analysis_cache = {}


def analyze_job(job):
    key = hashlib.md5((job.get("title", "") + job.get("description", "")).encode()).hexdigest()
    if key not in _analysis_cache:
        text = f"{job.get('description', '')} Experience: {job.get('experience', '')}"
        a = jd_analyzer.analyze(text)
        if not a["must"] and not a["nice"]:      # e.g. very short API snippets
            a["must"] = sorted(dictionary_skills(job.get("title", "") + ". " + job.get("description", "")))
            a["must_groups"] = [[s] for s in a["must"]]
        _analysis_cache[key] = a
    return _analysis_cache[key]


def verdict(score):
    for threshold, label in config.VERDICTS:
        if score >= threshold:
            return label
    return config.VERDICTS[-1][1]


def _job_text(job, a):
    # brackets matter: without them the `or` never runs and short postings are matched on the title only
    return f"{job.get('title', '')}. " + (" ".join(r["text"] for r in a["requirements"]) or job.get("description", ""))


def _level_fit(profile, job, a):
    exp = profile.get("experience_years") or 0
    if a["fresher_friendly"] or a["min_years"] <= exp + 0.5:
        fit = 1.0
    elif a["min_years"] <= exp + 2:
        fit = 0.5
    else:
        fit = 0.1
    if a["degree_required"] and not (profile.get("education") or {}).get("degrees"):
        fit *= 0.7
    return fit


def rank_jobs(profile, jobs, rerank=True):
    if not jobs:
        return []
    skills, learning = effective_skills(profile)
    analyses = [analyze_job(j) for j in jobs]
    resume_txt = " ".join(profile_text(profile))
    job_txts = [_job_text(j, a) for j, a in zip(jobs, analyses)]

    # keyword overlap (TF-IDF word uni/bigrams)
    tf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True).fit_transform([resume_txt] + job_txts)
    kw = np.clip(cosine_similarity(tf[0], tf[1:]).ravel() * 2.5, 0, 1)

    # stage 1 semantic similarity
    rv = resume_vector(profile)
    if rv is not None:
        jv = models.embed(job_txts)
        sem = np.clip(((jv @ rv) - 0.15) / 0.45, 0, 1)
        sem_method = "embedding"
    else:
        ch = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True).fit_transform([resume_txt] + job_txts)
        sem = np.clip(cosine_similarity(ch[0], ch[1:]).ravel() * 1.6, 0, 1)
        sem_method = "tfidf"

    w = config.FIT_WEIGHTS
    results = []
    for i, (job, a) in enumerate(zip(jobs, analyses)):
        sk = group_coverage(a["must_groups"], a["nice_groups"], skills, learning) if (a["must"] or a["nice"]) else float(sem[i])
        fit = _level_fit(profile, job, a)
        results.append({"job": job, "analysis": a, "semantic": float(sem[i]), "skills": sk,
                        "keywords": float(kw[i]), "fit": fit, "sem_method": sem_method})
    _total(results, w)

    # stage 2: cross-encoder re-ranking of the top candidates
    if rerank and models.get_cross_encoder() is not None:
        results.sort(key=lambda r: -r["score"])
        top = results[:config.RERANK_TOP_N]
        passage = resume_txt[:1500]
        pairs = [(r["job"].get("title", "") + ". " + "; ".join(x["text"] for x in r["analysis"]["requirements"][:8]), passage)
                 for r in top]
        ce = models.cross_scores(pairs)
        if ce is not None:
            for r, c in zip(top, ce):
                r["cross"] = float(c)
                r["semantic"] = 0.5 * r["semantic"] + 0.5 * float(c)
                r["sem_method"] = "embedding + cross-encoder"
            _total(results, w)

    for r in results:
        a = r["analysis"]
        r["matched"] = [s for s in a["must"] + a["nice"] if s in skills]
        r["missing_must"] = [" or ".join(g) for g in a["must_groups"] if group_credit(g, skills, learning) < 0.75]
        r["missing_nice"] = [" or ".join(g) for g in a["nice_groups"] if group_credit(g, skills, learning) < 0.75]
        r["verdict"] = verdict(r["score"])
    results.sort(key=lambda r: -r["score"])
    return results


def _total(results, w):
    for r in results:
        r["score"] = round(100 * (w["semantic"] * r["semantic"] + w["skills"] * r["skills"] +
                                  w["keywords"] * r["keywords"] + w["fit"] * r["fit"]), 1)


def explain(profile, job):
    """Pair each job requirement with the most similar resume line."""
    a = analyze_job(job)
    skills, learning = effective_skills(profile)
    reqs = a["requirements"] or [{"text": s, "kind": "info", "skills": []} for s in a["summary"]]
    units = profile_text(profile, max_lines=120)
    if not reqs or not units:
        return []
    vecs = models.embed([r["text"] for r in reqs] + units)
    if vecs is not None:
        sims = models.cosine_matrix(vecs[:len(reqs)], vecs[len(reqs):])
        strong_t, partial_t = 0.55, 0.40
    else:
        tf = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit([r["text"] for r in reqs] + units)
        sims = cosine_similarity(tf.transform([r["text"] for r in reqs]), tf.transform(units))
        strong_t, partial_t = 0.30, 0.12
    rows = []
    for i, r in enumerate(reqs):
        j = int(np.argmax(sims[i]))
        sim = float(sims[i, j])
        have = [s for s in r["skills"] if credit(s, skills, learning) >= 0.75]
        if r["skills"]:
            if len(have) == len(r["skills"]):
                status = "strong"
            elif have or sim >= partial_t:
                status = "partial"
            else:
                status = "missing"
        else:
            status = "strong" if sim >= strong_t else "partial" if sim >= partial_t else "missing"
        rows.append({"requirement": r["text"], "kind": r["kind"], "status": status,
                     "evidence": units[j] if status != "missing" else None, "similarity": round(sim, 2),
                     "skills_have": have, "skills_missing": [s for s in r["skills"] if s not in have]})
    return rows


def near_misses(results, target=55):
    """Which 1-2 skills would lift the most below-target jobs to at least a good match?"""
    w = config.FIT_WEIGHTS["skills"]
    gains = {}
    for r in results:
        if r["score"] >= target or not (1 <= len(r["missing_must"]) <= 2) or r.get("risk_level") == "high":
            continue
        a = r["analysis"]
        total_w = 2 * len(a["must_groups"]) + len(a["nice_groups"])
        new_skill = min(1.0, r["skills"] + 2 * len(r["missing_must"]) / max(total_w, 1))
        if r["score"] + 100 * w * (new_skill - r["skills"]) >= target:
            key = tuple(sorted(r["missing_must"]))
            gains.setdefault(key, []).append(r["job"].get("title", ""))
    return sorted(({"skills": list(k), "count": len(v), "jobs": v[:5]} for k, v in gains.items()),
                  key=lambda x: (-x["count"], len(x["skills"])))[:6]
