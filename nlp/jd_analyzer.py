"""
Reads a job description the way a person would:
- separates must-have from nice-to-have requirements using cue words and
  "Preferred:" style blocks,
- extracts years of experience, degree requirements and fresher suitability,
- writes a 3-sentence extractive summary with TextRank.
"""
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from nlp.preprocess import sentences
from nlp.skills import dictionary_skills, skill_spans

MUST_CUES = re.compile(r"\b(must|required|requirement|requirements|mandatory|essential|need(ed)?|should have|"
                       r"strong|proficien\w*|expertise|minimum|solid|hands-on|good knowledge|working knowledge)\b", re.I)
NICE_CUES = re.compile(r"\b(prefer\w*|plus|bonus|nice to have|good to have|desirable|advantage\w*|"
                       r"familiarity|exposure|optional|added advantage|is a plus)\b", re.I)
NICE_BLOCK = re.compile(r"^\s*(preferred|nice to have|good to have|bonus|plus|desirable)\b[^:]{0,30}:", re.I)
MUST_BLOCK = re.compile(r"^\s*(requirements?|required|must have|qualifications?|what you need|skills required)\b[^:]{0,30}:", re.I)
FRESHER_CUES = re.compile(r"\b(fresher|freshers|entry[- ]level|graduate trainee|trainee|intern|internship|"
                          r"0\s*-\s*[12]\s*(years|yrs)|campus|new grad|no experience required)\b", re.I)
DEGREE_PAT = re.compile(r"\b(b\.?e|b\.?tech|m\.?tech|bca|mca|b\.?sc|m\.?sc|mba|bachelor'?s?|master'?s?|degree|graduate)\b", re.I)
YEARS_PAT = re.compile(r"(\d+(?:\.\d+)?)\s*(?:\+|\s*-\s*\d+)?\s*(?:years|yrs)", re.I)
BOILERPLATE = re.compile(r"\b(we are|our company|about us|equal opportunity|apply now|benefits|perks|stipend|salary|ctc)\b", re.I)


def _split_blocks(text):
    """Split 'Requirements: a. b. Preferred: c.' into labelled segments."""
    parts = re.split(r"(?=\b(?:Requirements?|Required|Must have|Qualifications?|Preferred|Nice to have|"
                     r"Good to have|Bonus|Plus|Desirable)\s*:)", text, flags=re.I)
    out = []
    for p in parts:
        label = "nice" if NICE_BLOCK.match(p) else "must" if MUST_BLOCK.match(p) else None
        out.append((label, p))
    return out


def alternative_groups(sentence):
    """Find 'A or B' / 'A, B or C' / 'A/B' alternatives. Returns list of sets."""
    spans = skill_spans(sentence)
    groups, i = [], 0
    while i < len(spans) - 1:
        chain = [spans[i]]
        j = i
        while j < len(spans) - 1:
            gap = sentence[spans[j][1]:spans[j + 1][0]].lower()
            if re.fullmatch(r"\s*(,|/|\bor\b|,\s*or\b)\s*", gap):
                chain.append(spans[j + 1])
                j += 1
            else:
                break
        gaps = " ".join(sentence[a[1]:b[0]] for a, b in zip(chain, chain[1:])).lower()
        if len(chain) > 1 and (" or " in f" {gaps} " or "/" in gaps):
            groups.append({c[2] for c in chain})
        i = j + 1
    return groups


def _grouped(skills, groups):
    """Turn a flat skill set into requirement groups (alternatives share a group)."""
    out, used = [], set()
    for g in groups:
        g = g & skills
        if len(g) > 1 and not (g & used):
            out.append(sorted(g))
            used |= g
    out += [[s] for s in sorted(skills - used)]
    return out


def analyze(text):
    reqs, must_skills, nice_skills, groups = [], set(), set(), []
    any_cue = False
    for label, block in _split_blocks(text):
        for sent in sentences(block):
            skills = set(dictionary_skills(sent))
            groups += alternative_groups(sent)
            nice = bool(NICE_CUES.search(sent)) or label == "nice"
            must = (bool(MUST_CUES.search(sent)) or label == "must") and not nice
            any_cue |= nice or must
            kind = "nice" if nice else "must" if must else "info"
            if skills or kind != "info":
                if not BOILERPLATE.search(sent) or skills:
                    reqs.append({"text": re.sub(r"^\w[\w ]{0,25}:\s*", "", sent).strip(), "kind": kind,
                                 "skills": sorted(skills)})
            if kind == "nice":
                nice_skills |= skills
            else:
                must_skills |= skills
    if not any_cue:                 # no cue words at all: treat every skill as required
        must_skills |= nice_skills
        nice_skills = set()
    nice_skills -= must_skills
    years = [float(y) for y in YEARS_PAT.findall(text) if float(y) < 25]
    min_years = min(years) if years else 0.0
    fresher = bool(FRESHER_CUES.search(text)) or min_years == 0
    return {
        "must": sorted(must_skills), "nice": sorted(nice_skills),
        "must_groups": _grouped(must_skills, groups), "nice_groups": _grouped(nice_skills, groups),
        "requirements": [r for r in reqs if len(r["text"].split()) >= 3][:25],
        "min_years": min_years, "fresher_friendly": fresher,
        "degree_required": bool(DEGREE_PAT.search(text)),
        "summary": summarize(text, 3),
    }


def summarize(text, n=3):
    """TextRank: sentences are nodes, TF-IDF cosine similarity are edge weights,
    PageRank finds the most central sentences. Returns them in original order."""
    sents = [s for s in sentences(text) if len(s.split()) >= 4]
    if len(sents) <= n:
        return [re.sub(r"^(Requirements?|Preferred|Required|Qualifications?)\s*:\s*", "", x, flags=re.I) for x in sents]
    tfidf = TfidfVectorizer(stop_words="english").fit_transform(sents)
    sim = cosine_similarity(tfidf)
    np.fill_diagonal(sim, 0)
    row_sums = sim.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    trans = sim / row_sums
    scores = np.ones(len(sents)) / len(sents)
    for _ in range(50):                       # power iteration with damping 0.85
        scores = 0.15 / len(sents) + 0.85 * trans.T @ scores
    top = sorted(np.argsort(-scores)[:n])
    return [re.sub(r"^(Requirements?|Preferred|Required|Qualifications?)\s*:\s*", "", sents[i], flags=re.I) for i in top]
