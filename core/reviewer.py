"""
Resume feedback using linguistic analysis.

For each bullet in projects/experience: does it start with a strong action verb
(part-of-speech tagging / lemma), is it in passive voice (dependency labels
nsubjpass/auxpass), does it show a measurable result, is it vague or too long?
Plus section, contact, length, ATS-format and bias-risk checks.
"""
import re

from nlp.models import get_spacy

ACTION_VERBS = set("""achieve analyse analyze architect automate build collaborate conduct configure construct
coordinate create debug decrease define deliver deploy design develop devise document engineer enhance establish
evaluate execute expand extract generate guide identify implement improve increase integrate introduce launch lead
maintain manage mentor migrate model monitor optimise optimize organise organize perform plan present prototype
publish redesign reduce refactor research resolve scale secure simplify solve streamline support test train
transform visualise visualize win write program predict classify detect scrape clean analyse""".split())
WEAK_PHRASES = ["responsible for", "worked on", "helped with", "helped in", "involved in", "duties included",
                "tasked with", "participated in", "assisted in", "worked with", "part of a team", "various", "etc"]
METRIC = re.compile(r"\d|%|\b(percent|times|x faster|users|accuracy|reduced|increased)\b", re.I)
PASSIVE_FALLBACK = re.compile(r"\b(was|were|is|are|been|being)\s+(\w+ed|built|done|made|written)\b(\s+by\b)?", re.I)


IRREGULAR = {"built": "build", "wrote": "write", "written": "write", "led": "lead", "won": "win", "made": "make",
             "ran": "run", "drove": "drive", "taught": "teach", "designed": "design", "set": "set", "found": "find"}
ACTION_VERBS |= {"make", "run", "drive", "teach", "set", "find", "lead", "create", "use"}


def _lemma_first_word(text):
    nlp = get_spacy()
    if nlp is not None:
        doc = nlp(text)
        for t in doc:
            if not t.is_punct:
                return t.lemma_.lower(), doc
        return "", doc
    w = (re.findall(r"[A-Za-z]+", text) or [""])[0].lower()
    if w in IRREGULAR:
        return IRREGULAR[w], None
    for suf, rep in (("ied", "y"), ("ed", ""), ("ing", ""), ("es", ""), ("s", "")):
        if w.endswith(suf) and w[:-len(suf)] + rep in ACTION_VERBS:
            return w[:-len(suf)] + rep, None
        if w.endswith(suf) and w[:-len(suf)] + "e" in ACTION_VERBS:
            return w[:-len(suf)] + "e", None
    return w, None


def _is_passive(text, doc):
    if doc is not None:
        return any(t.dep_ in ("nsubjpass", "auxpass") for t in doc)
    return bool(PASSIVE_FALLBACK.search(text))


def analyze_bullets(sections):
    lines = sections.get("projects", []) + sections.get("experience", [])
    out = []
    for ln in lines:
        words = ln.split()
        if len(words) < 4:      # titles, dates, short labels
            continue
        body = re.split(r"\s[-\u2013:|]\s", ln, maxsplit=1)
        body = body[1] if len(body) == 2 and len(body[0].split()) <= 6 else ln   # "Title - Built ..."
        lemma, doc = _lemma_first_word(body)
        issues = []
        if lemma not in ACTION_VERBS:
            issues.append("Start with a strong action verb (e.g. Built, Designed, Reduced).")
        weak = [p for p in WEAK_PHRASES if re.search(r"\b" + re.escape(p) + r"\b", ln, re.I)]
        if weak:
            issues.append(f"Replace weak phrasing: \u201c{weak[0]}\u201d.")
        if _is_passive(body, doc):
            issues.append("Use active voice: say what you did, not what was done.")
        if not METRIC.search(ln):
            issues.append("Add a measurable result (numbers, %, users, accuracy, time saved).")
        if len(words) > 32:
            issues.append("Shorten this bullet to one or two lines.")
        out.append({"text": ln, "issues": issues})
    return out


def review(profile):
    sec = profile.get("sections") or {}
    findings = []   # (severity, area, message)

    def add(sev, area, msg):
        findings.append({"severity": sev, "area": area, "message": msg})

    for s, label in (("education", "Education"), ("skills", "Skills"), ("projects", "Projects")):
        if not sec.get(s):
            add("high", "Sections", f"Add a clear \u201c{label}\u201d section with that heading.")
    if not sec.get("experience") and profile.get("level") != "student":
        add("low", "Sections", "Add internships, training or freelance work under \u201cExperience\u201d if you have any.")
    if not profile.get("email"):
        add("high", "Contact", "Add a professional email address.")
    if not profile.get("phone"):
        add("medium", "Contact", "Add a phone number.")
    links = profile.get("links") or {}
    if not links.get("github"):
        add("medium", "Links", "Add your GitHub profile so recruiters can see your code.")
    if not links.get("linkedin"):
        add("low", "Links", "Add your LinkedIn profile URL.")
    wc = profile.get("word_count", 0)
    if wc < 250:
        add("medium", "Length", f"Your resume is short ({wc} words). Describe your projects and impact in more detail.")
    elif wc > 900 or profile.get("pages", 1) > 2:
        add("medium", "Length", f"Your resume is long ({wc} words). Freshers should aim for one page (about 400-700 words).")
    ats = profile.get("ats_flags") or []
    if "tables" in ats:
        add("medium", "ATS format", "Tables can scramble text in applicant tracking systems. Use simple lines instead.")
    if "images" in ats:
        add("low", "ATS format", "Images and icons are ignored by ATS software; make sure no information is only in images.")
    if "multi_column" in ats:
        add("medium", "ATS format", "Two-column layouts are often read in the wrong order by ATS. A single column is safer.")
    for item in profile.get("personal_flags") or []:
        add("medium", "Bias risk", f"Remove \u201c{item}\u201d. It isn't needed to judge your skills and can trigger unconscious bias.")
    if len(profile.get("skills") or {}) < 6:
        add("medium", "Skills", "List your technical skills clearly, grouped by type (languages, frameworks, tools).")

    bullets = analyze_bullets(sec)
    weak_bullets = [b for b in bullets if b["issues"]]
    if bullets:
        no_metric = sum(1 for b in bullets if any("measurable" in i for i in b["issues"]))
        if no_metric / len(bullets) > 0.5:
            add("medium", "Impact", f"{no_metric} of {len(bullets)} bullets have no measurable result.")
    weights = {"high": 12, "medium": 6, "low": 3}
    score = 100 - sum(weights[f["severity"]] for f in findings)
    if bullets:
        score -= round(15 * len(weak_bullets) / len(bullets))
    return {"score": max(0, min(100, score)), "findings": findings, "bullets": bullets,
            "weak_bullets": len(weak_bullets)}


def compare(old, new):
    """Version comparison between two analysed resumes."""
    ro, rn = review(old), review(new)
    so, sn = set(old.get("skills") or {}), set(new.get("skills") or {})
    return {"old_score": ro["score"], "new_score": rn["score"], "added": sorted(sn - so), "removed": sorted(so - sn),
            "old_issues": len(ro["findings"]) + ro["weak_bullets"], "new_issues": len(rn["findings"]) + rn["weak_bullets"]}
