"""
Three-layer skill extraction with context awareness and skill levels.

Layer 1  Dictionary: whole-word matching against ~330 known skills and aliases.
Layer 2  Semantic:   noun phrases (spaCy) or word n-grams are compared with skill
                     names by embedding similarity, so "dataframe manipulation in
                     pandas" or "built REST backends" map to known skills.
Layer 3  Discovery:  tech-looking terms that match nothing (e.g. "LangGraph")
                     are returned as "possible skills" for the user to confirm.

Context: skills mentioned with learning intent ("currently learning Docker") or
negation ("no experience with Java") are not counted as skills the person has.
Levels:  listed only -> beginner, used in one project -> intermediate,
         several projects or an internship -> advanced.
"""
import json
import os
import re

import config
from nlp import models
from nlp.preprocess import clean_text, normalize_for_matching

_DATA = json.load(open(os.path.join(config.DATA_DIR, "skills.json"), encoding="utf-8"))
CATEGORY = {s: cat for cat, group in _DATA["categories"].items() for s in group}
ALIASES = _DATA["aliases"]
CASE_SENSITIVE = _DATA["ambiguous_case_sensitive"]
EFFORT = _DATA["effort_hours"]
ALL_TERMS = sorted(set(CATEGORY) | set(ALIASES), key=len, reverse=True)

# boundaries that also work for c++, c#, node.js, ci/cd
_PAT = {t: re.compile(r"(?<![\w+#.])" + re.escape(t) + r"(?![\w+#]|\.\w)") for t in ALL_TERMS}
_PAT_CS = {t: re.compile(r"(?<![\w+#.])" + re.escape(v) + r"(?![\w+#]|\.\w)") for t, v in CASE_SENSITIVE.items()}

LEARNING_CUES = re.compile(r"(currently|actively)?\s*((?<!machine )(?<!deep )(?<!reinforcement )(?<!transfer )\blearning\b|want to learn|plan(ning)? to learn|eager to learn|"
                           r"interested in learning|exploring|beginner in|basic knowledge of|aspiring to learn)", re.I)
NEGATION_CUES = re.compile(r"\b(no|not|never|without|lack(ing)?)\b(?:\s+\w+){0,3}\s*(experience|knowledge|familiar|exposure)?", re.I)
LEARN_LEMMAS = {"learn", "explore", "study"}

STOP_TERMS = {"cgpa", "sgpa", "gpa", "be", "b.e", "btech", "cse", "ise", "ece", "eee", "vtu", "puc", "sslc", "cbse",
              "icse", "hr", "ceo", "pdf", "usa", "india", "it", "ltd", "pvt", "inc", "llp", "mnc", "ii", "iii", "iv",
              "resume", "curriculum vitae", "github", "linkedin", "email", "phone", "mobile", "ppt", "ai", "tech"}


def canonical(term):
    return ALIASES.get(term, term)


def category_of(skill):
    return CATEGORY.get(skill, "Other")


def _context(sentence, start):
    """Classify the mention at char offset `start`: 'has', 'learning' or 'negated'."""
    # "learn" inside a skill name (scikit-learn, machine learning) is not learning intent
    spans = skill_spans(sentence)

    def in_skill(pos):
        return any(a <= pos < b for a, b, _ in spans)

    offset = max(0, start - 45)
    before = sentence[offset:start]
    if any(not in_skill(offset + m.end() - 1) for m in LEARNING_CUES.finditer(before) if m.group().strip()):
        return "learning"
    neg = NEGATION_CUES.search(before)
    if neg and neg.end() >= len(before) - 25:
        return "negated"
    nlp = models.get_spacy()
    if nlp is not None:      # dependency parse: is the skill governed by "learn" or a negated verb?
        doc = nlp(sentence)
        for tok in doc:
            if tok.idx <= start < tok.idx + len(tok.text) + 1:
                for anc in tok.ancestors:
                    if anc.lemma_.lower() in LEARN_LEMMAS and not in_skill(anc.idx):
                        return "learning"
                    if any(c.dep_ == "neg" for c in anc.children):
                        return "negated"
                break
    return "has"


def _split_mentions(text):
    """Yield (sentence, lowercase sentence) units for context checks."""
    for ln in clean_text(text).split("\n"):
        for s in re.split(r"(?<=[.;!?])\s+", ln):
            if s.strip():
                yield s, s.lower()


def dictionary_skills(text):
    """Layer 1. Returns {skill: {"has": n, "learning": n, "negated": n}}."""
    found = {}
    for sent, low in _split_mentions(text):
        taken = []   # character spans already matched (avoid 'java' inside 'javascript')
        for term in ALL_TERMS:
            if term in CASE_SENSITIVE:
                pat, hay = _PAT_CS[term], sent
            else:
                pat, hay = _PAT[term], low
            for m in pat.finditer(hay):
                if any(a <= m.start() < b for a, b in taken):
                    continue
                taken.append((m.start(), m.end()))
                ctx = _context(sent, m.start())
                skill = canonical(term)
                found.setdefault(skill, {"has": 0, "learning": 0, "negated": 0})[ctx] += 1
    return found


def skill_spans(sentence):
    """[(start, end, skill)] for every dictionary skill in one sentence, in order."""
    low, taken, out = sentence.lower(), [], []
    for term in ALL_TERMS:
        pat, hay = (_PAT_CS[term], sentence) if term in CASE_SENSITIVE else (_PAT[term], low)
        for m in pat.finditer(hay):
            if any(a <= m.start() < b for a, b in taken):
                continue
            taken.append((m.start(), m.end()))
            out.append((m.start(), m.end(), canonical(term)))
    return sorted(out)


def _candidate_phrases(text):
    nlp = models.get_spacy()
    if nlp is not None:
        doc = nlp(clean_text(text)[:50000])
        phrases = {c.text.strip() for c in doc.noun_chunks}
    else:
        phrases = set()
        for ln in clean_text(text).split("\n"):
            words = re.findall(r"[A-Za-z][A-Za-z0-9+#.]*", ln)
            for n in (2, 3):
                phrases.update(" ".join(words[i:i + n]) for i in range(len(words) - n + 1))
    return [p for p in phrases if 2 <= len(p.split()) <= 5 and len(p) < 60]


def semantic_skills(text, already):
    """Layer 2. Map phrases to known skills by meaning (needs the embedding model)."""
    already_words = {w for sk in already for w in sk.split()}
    phrases = [p for p in _candidate_phrases(text) if not (set(p.lower().split()) & already_words)]
    skills = [s for s in CATEGORY if s not in already and category_of(s) != "Soft Skills"]
    if not phrases or not skills:
        return {}
    vecs = models.embed(phrases[:400] + skills)
    if vecs is None:
        return {}
    p_vec, s_vec = vecs[:len(phrases[:400])], vecs[len(phrases[:400]):]
    sims = models.cosine_matrix(p_vec, s_vec)
    out = {}
    for i, phrase in enumerate(phrases[:400]):
        j = int(sims[i].argmax())
        if sims[i, j] >= config.SEMANTIC_SKILL_THRESHOLD:
            skill = skills[j]
            if float(sims[i, j]) > out.get(skill, {}).get("score", 0):
                out[skill] = {"score": round(float(sims[i, j]), 2), "phrase": phrase}
    return out


def possible_skills(text, known, sections=None):
    """Layer 3. Tech-looking terms not in the taxonomy. Acronyms are only trusted inside
    the skills section; CamelCase words and *.js names inside skills/projects/experience."""
    if sections:
        skill_txt = " , ".join(sections.get("skills", []))
        work_txt = " , ".join(sections.get("skills", []) + sections.get("projects", []) + sections.get("experience", []))
    else:
        skill_txt = work_txt = text
    cands = set(re.findall(r"(?<![-\w])[A-Z]{2,6}(?![-\w])", skill_txt))
    cands |= set(re.findall(r"\b[A-Z][a-z]+[A-Z][A-Za-z]+\b|\b\w+\.js\b", work_txt))
    known_words = {w for t in ALL_TERMS for w in re.split(r"[ /]", t)}
    headings = {"skills", "projects", "tools", "languages", "frameworks", "education", "experience"}
    out = []
    for c in sorted(cands):
        low = c.lower()
        if low in STOP_TERMS or low in headings or canonical(low) in known or low in CATEGORY or low in ALIASES:
            continue
        if low in known_words:      # e.g. "API" from "REST API"
            continue
        out.append(c)
    return out[:15]


def skill_levels(sections, skills):
    """Estimate level per skill from where it appears in the resume."""
    def count_in(sec):
        txt = normalize_for_matching(" \n ".join(sections.get(sec, [])))
        return {s for s in skills if _mentioned(s, txt)}
    in_exp = count_in("experience")
    proj_lines = sections.get("projects", [])
    proj_hits = {s: 0 for s in skills}
    for block in _project_blocks(proj_lines):
        low = normalize_for_matching(block)
        for s in skills:
            if _mentioned(s, low):
                proj_hits[s] += 1
    levels = {}
    for s in skills:
        if s in in_exp or proj_hits[s] >= 2:
            levels[s] = "advanced"
        elif proj_hits[s] == 1:
            levels[s] = "intermediate"
        else:
            levels[s] = "beginner"
    return levels


def _mentioned(skill, low_text):
    terms = [skill] + [a for a, c in ALIASES.items() if c == skill]
    for t in terms:
        if t in CASE_SENSITIVE:
            continue    # ambiguous short names are not trusted in lowercase text
        if _PAT[t].search(low_text):
            return True
    return False


def _project_blocks(proj_lines):
    """Group project lines into one block per project (title line + its bullets)."""
    blocks, cur = [], []
    for ln in proj_lines:
        is_title = (len(ln.split()) <= 9 and not ln.endswith(".")) or re.search(r"\|\s*(github|link)|\(\d{4}\)", ln, re.I)
        if is_title and cur:
            blocks.append(" ".join(cur))
            cur = []
        cur.append(ln)
    if cur:
        blocks.append(" ".join(cur))
    return blocks


def extract_skills(text, sections=None):
    """Full pipeline. Returns dict with skills (name -> info), learning, possible."""
    dict_found = dictionary_skills(text)
    have = {s for s, c in dict_found.items() if c["has"] > 0}
    learning = sorted(s for s, c in dict_found.items() if c["has"] == 0 and c["learning"] > 0)
    sem = semantic_skills(text, set(dict_found))
    levels = skill_levels(sections or {}, have | set(sem))
    skills = {}
    for s in sorted(have):
        skills[s] = {"category": category_of(s), "level": levels.get(s, "beginner"), "source": "dictionary"}
    for s, info in sem.items():
        skills[s] = {"category": category_of(s), "level": levels.get(s, "beginner"), "source": "semantic",
                     "evidence": info["phrase"], "similarity": info["score"]}
    return {"skills": skills, "learning": learning,
            "possible": possible_skills(text, set(skills) | set(learning), sections)}


def skills_in_text(text):
    """Quick extraction for job descriptions: set of canonical skills mentioned."""
    return set(dictionary_skills(text))
