"""
Evaluation for the report. Run:  python scripts/evaluate.py
1. Role matching accuracy (top-1 / top-3) for: TF-IDF only, embeddings only, hybrid, hybrid + skills.
2. Skill extraction precision / recall / F1: dictionary only vs full NLP pipeline.
Results are printed and written to docs/results/evaluation.md
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.parser import analyze_text            # noqa: E402
from core.roles import predict_roles            # noqa: E402
from nlp import models                          # noqa: E402
from nlp.skills import dictionary_skills        # noqa: E402

DATA = json.load(open(os.path.join(os.path.dirname(__file__), "..", "tests", "eval", "labelled.json"), encoding="utf-8"))


def role_eval():
    methods = {"TF-IDF only (baseline)": dict(mode="tfidf", skill_weight=0.0),
               "Embeddings only": dict(mode="embedding", skill_weight=0.0),
               "Skills only": dict(mode="tfidf", skill_weight=1.0),
               "Hybrid (skills + TF-IDF)": dict(mode="tfidf", skill_weight=0.6),
               "Hybrid (skills + embeddings)": dict(mode="embedding", skill_weight=0.6)}
    rows = []
    profiles = [(d, {**analyze_text(d["text"]), "extra_skills": {}}) for d in DATA]
    for name, kw in methods.items():
        if kw["mode"] == "embedding" and models.get_embedder() is None:
            rows.append((name, None, None))
            continue
        t1 = t3 = 0
        for d, p in profiles:
            ranked = [r["id"] for r in predict_roles(p, top=None, **kw)]
            t1 += ranked[0] == d["role"]
            t3 += d["role"] in ranked[:3]
        rows.append((name, t1 / len(DATA), t3 / len(DATA)))
    return rows


def prf(pred, gold):
    tp = len(pred & gold)
    p = tp / len(pred) if pred else 0
    r = tp / len(gold) if gold else 0
    return p, r, (2 * p * r / (p + r) if p + r else 0)


def skill_eval():
    out = {}
    for name in ("Dictionary only", "Full NLP pipeline"):
        ps, rs, fs = [], [], []
        for d in DATA:
            gold = set(d["skills"])
            if name == "Dictionary only":
                pred = set(dictionary_skills(d["text"]))           # counts "learning Docker" too
            else:
                pred = set(analyze_text(d["text"])["skills"])      # context-aware + semantic layer
            p, r, f = prf(pred, gold)
            ps.append(p); rs.append(r); fs.append(f)
        n = len(DATA)
        out[name] = (sum(ps) / n, sum(rs) / n, sum(fs) / n)
    return out


if __name__ == "__main__":
    status = models.status()
    lines = ["# Evaluation results", "", f"Labelled resumes: {len(DATA)}", "",
             "Active components: " + ", ".join(f"{k} ({'yes' if v else 'no'})" for k, v in status.items()), "",
             "## 1. Role matching accuracy", "", "| Method | Top-1 | Top-3 |", "|---|---|---|"]
    for name, a1, a3 in role_eval():
        lines.append(f"| {name} | {'n/a (model not installed)' if a1 is None else f'{a1:.0%}'} | {'' if a3 is None else f'{a3:.0%}'} |")
    lines += ["", "## 2. Skill extraction", "", "| Method | Precision | Recall | F1 |", "|---|---|---|---|"]
    for name, (p, r, f) in skill_eval().items():
        lines.append(f"| {name} | {p:.2f} | {r:.2f} | {f:.2f} |")
    lines += ["", "Re-run after installing all models to fill in the embedding rows.", ""]
    text = "\n".join(lines)
    print(text)
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "docs", "results"), exist_ok=True)
    open(os.path.join(os.path.dirname(__file__), "..", "docs", "results", "evaluation.md"), "w").write(text)
