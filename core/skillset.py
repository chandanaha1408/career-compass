"""Helpers for the user's *effective* skill set: resume skills + skills they
confirmed + skills completed in the learning tracker."""

LEVEL_CREDIT = {"advanced": 1.0, "intermediate": 0.9, "beginner": 0.75}
LEARNING_CREDIT = 0.25


def effective_skills(profile):
    skills = {s: i.get("level", "beginner") for s, i in (profile.get("skills") or {}).items()}
    for s, lvl in (profile.get("extra_skills") or {}).items():
        if s not in skills or lvl == "intermediate" and skills[s] == "beginner":
            skills[s] = lvl
    learning = set(profile.get("learning") or []) - set(skills)
    return skills, learning


def credit(skill, skills, learning):
    if skill in skills:
        return LEVEL_CREDIT.get(skills[skill], 0.75)
    if skill in learning:
        return LEARNING_CREDIT
    return 0.0


def group_credit(group, skills, learning):
    """A requirement group is satisfied by its best alternative."""
    return max(credit(s, skills, learning) for s in group)


def group_coverage(must_groups, nice_groups, skills, learning, weights=None):
    """Like weighted_coverage, but each group of alternatives counts once."""
    weights = weights or {}
    total = got = 0.0
    for groups, base in ((must_groups, 2.0), (nice_groups, 1.0)):
        for g in groups:
            w = base * max(weights.get(s, 1.0) for s in g)
            total += w
            got += w * group_credit(g, skills, learning)
    return got / total if total else 0.0


def weighted_coverage(must, nice, skills, learning, weights=None):
    """Share of required skills covered, must-haves count double.
    `weights` optionally scales each skill (e.g. market demand)."""
    weights = weights or {}
    total = got = 0.0
    for s in must:
        w = 2.0 * weights.get(s, 1.0)
        total += w
        got += w * credit(s, skills, learning)
    for s in nice:
        w = 1.0 * weights.get(s, 1.0)
        total += w
        got += w * credit(s, skills, learning)
    return got / total if total else 0.0


def profile_text(profile, max_lines=60):
    """Lines that describe what the person can do (used for semantic matching)."""
    sec = profile.get("sections") or {}
    lines = []
    for name in ("summary", "skills", "projects", "experience", "certifications", "achievements", "other"):
        lines += sec.get(name, [])
    if not lines:
        lines = (profile.get("text") or "").split("\n")
    skills, _ = effective_skills(profile)
    lines.append("Skills: " + ", ".join(sorted(skills)))
    return [l for l in lines if len(l.split()) >= 2][:max_lines]
