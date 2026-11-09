"""
Splits a resume into sections (education, skills, projects, experience ...).

Step 1: recognise heading lines ("Technical Skills", "Academic Projects").
Step 2: lines before any heading, or under unknown headings, are classified by
        their content - keyword scoring, upgraded to embedding similarity with
        section prototypes when the sentence-embedding model is available.
"""
import re

from nlp.preprocess import lines
from nlp import models

HEADINGS = {
    "summary": ["summary", "objective", "career objective", "profile", "about me", "professional summary"],
    "education": ["education", "academic details", "academics", "qualification", "qualifications", "educational qualification"],
    "skills": ["skills", "technical skills", "key skills", "core competencies", "technologies", "tech stack", "tools",
               "skill set", "technical proficiency", "programming skills"],
    "projects": ["projects", "academic projects", "personal projects", "key projects", "project work", "mini projects"],
    "experience": ["experience", "work experience", "internship", "internships", "professional experience",
                   "employment", "work history", "industrial training", "training"],
    "certifications": ["certifications", "certificates", "courses", "online courses", "licenses"],
    "achievements": ["achievements", "awards", "honors", "accomplishments", "extracurricular", "extra curricular",
                     "activities", "positions of responsibility", "leadership"],
    "personal": ["personal details", "personal information", "personal profile", "declaration", "hobbies", "interests",
                 "languages known"],
}
_HEAD_LOOKUP = {h: sec for sec, hs in HEADINGS.items() for h in hs}

KEYWORDS = {
    "education": r"\b(b\.?e|b\.?tech|m\.?tech|bca|mca|bsc|msc|cgpa|sgpa|gpa|university|college|institute|school|puc|12th|10th|sslc|cbse|percentage)\b",
    "experience": r"\b(intern|internship|worked at|company|pvt|ltd|employer|role:|duration)\b",
    "projects": r"\b(project|developed|built|implemented|designed|created|github\.com)\b",
    "skills": r"\b(languages|frameworks|tools|libraries|proficient|familiar)\b",
    "certifications": r"\b(certified|certificate|certification|coursera|udemy|nptel|course)\b",
    "personal": r"\b(date of birth|dob|nationality|marital|gender|father|languages known|hobbies)\b",
}
PROTOTYPES = {
    "education": "Bachelor of Engineering in Information Science, university, CGPA 8.5, graduation year",
    "experience": "Software intern at a company, worked on backend features for three months",
    "projects": "Built a web application using Python and React that lets users track expenses",
    "skills": "Programming languages Python Java SQL; frameworks React Django; tools Git Docker",
    "certifications": "Completed online certification course in machine learning from Coursera",
    "achievements": "Won first prize in college hackathon, coordinator of coding club",
}


def detect_heading(line):
    """Return the section name if this line is a heading, else None."""
    raw = line.strip().rstrip(":").strip()
    if len(raw) > 40 or len(raw.split()) > 5:
        return None
    key = re.sub(r"[^a-z ]", "", raw.lower()).strip()
    if key in _HEAD_LOOKUP:
        return _HEAD_LOOKUP[key]
    # e.g. "TECHNICAL SKILLS & TOOLS" -> first matching heading phrase
    for h, sec in sorted(_HEAD_LOOKUP.items(), key=lambda x: -len(x[0])):
        if key.startswith(h) and (raw.isupper() or raw.istitle() or line.strip().endswith(":")):
            return sec
    return None


def _classify_block(block_lines):
    text = " ".join(block_lines)
    low = text.lower()
    scores = {sec: len(re.findall(pat, low)) for sec, pat in KEYWORDS.items()}
    vec = models.embed([text[:500]] + list(PROTOTYPES.values()))
    if vec is not None:   # semantic similarity to each section prototype
        sims = models.cosine_matrix(vec[:1], vec[1:])[0]
        for sec, s in zip(PROTOTYPES, sims):
            scores[sec] = scores.get(sec, 0) + float(s) * 4
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "other"


def split_sections(text):
    """Return {section: [lines]} including a 'header' block (name/contact)."""
    sections, current, buffer = {}, None, []
    all_lines = lines(text)
    for i, ln in enumerate(all_lines):
        sec = detect_heading(ln)
        if sec:
            if buffer:
                name = current or ("header" if not sections else _classify_block(buffer))
                sections.setdefault(name, []).extend(buffer)
            current, buffer = sec, []
        else:
            buffer.append(ln)
    if buffer:
        name = current or ("header" if not sections else _classify_block(buffer))
        sections.setdefault(name, []).extend(buffer)
    # a resume with no headings at all: classify paragraph-sized chunks
    if list(sections) == ["header"] and len(all_lines) > 8:
        head, rest = all_lines[:3], all_lines[3:]
        sections = {"header": head}
        for i in range(0, len(rest), 4):
            chunk = rest[i:i + 4]
            sections.setdefault(_classify_block(chunk), []).extend(chunk)
    return sections
