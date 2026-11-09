"""
Role prediction: which job roles fit this resume best?

score = 0.6 x weighted skill coverage of the role's must/nice skills
      + 0.4 x semantic similarity between the resume and the role description
Semantic similarity uses sentence embeddings (mean of resume line embeddings),
or TF-IDF cosine similarity as a fallback.
"""
import json
import os

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import config
from core.skillset import effective_skills, weighted_coverage, profile_text
from nlp import models

ROLES = json.load(open(os.path.join(config.DATA_DIR, "roles.json"), encoding="utf-8"))
ROLE_BY_ID = {r["id"]: r for r in ROLES}


def role_text(r):
    return f"{r['title']}. {r['description']} Skills: {', '.join(r['must'] + r['nice'])}."


def resume_vector(profile):
    lines = profile_text(profile)
    vecs = models.embed(lines)
    if vecs is None:
        return None
    v = vecs.mean(axis=0)
    return v / (np.linalg.norm(v) or 1)


def semantic_role_scores(profile, mode="auto"):
    """Similarity of the resume to every role, scaled to 0-1."""
    rv = resume_vector(profile) if mode in ("auto", "embedding") else None
    if rv is not None:
        role_vecs = models.embed([role_text(r) for r in ROLES])
        sims = role_vecs @ rv
        return np.clip((sims - 0.15) / 0.45, 0, 1), "embedding"
    docs = [role_text(r) for r in ROLES] + [" ".join(profile_text(profile))]
    tf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True).fit_transform(docs)
    sims = cosine_similarity(tf[-1], tf[:-1]).ravel()
    return np.clip(sims / 0.3, 0, 1), "tfidf"


def predict_roles(profile, top=3, mode="auto", skill_weight=0.6):
    skills, learning = effective_skills(profile)
    sem, method = semantic_role_scores(profile, mode)
    out = []
    for r, s in zip(ROLES, sem):
        cov = weighted_coverage(r["must"], r["nice"], skills, learning)
        conf = skill_weight * cov + (1 - skill_weight) * float(s)
        out.append({"id": r["id"], "title": r["title"], "confidence": round(conf * 100),
                    "skill_score": round(cov * 100), "semantic": round(float(s) * 100),
                    "missing_must": [m for m in r["must"] if m not in skills],
                    "method": method})
    out.sort(key=lambda x: -x["confidence"])
    return out[:top] if top else out
