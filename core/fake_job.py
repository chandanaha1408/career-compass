"""
Fake job detection = transparent red-flag rules + an optional trained classifier.

Rules catch clear scam patterns (fees, WhatsApp-only contact, Aadhaar/OTP requests)
and explain *why* a posting is risky. The classifier (TF-IDF + logistic regression,
trained by scripts/train_fake_job.py on the Kaggle "Real or Fake Job Posting"
dataset) adds a learned probability. Without a trained model, rules work alone.
"""
import json
import os
import re

import config

_RULES = [dict(r, rx=re.compile(r["pattern"], re.I))
          for r in json.load(open(os.path.join(config.DATA_DIR, "fake_job_rules.json"), encoding="utf-8"))]
MODEL_PATH = os.path.join(config.MODELS_DIR, "fake_job_model.joblib")
_model = {"loaded": False, "obj": None}


def _get_model():
    if not _model["loaded"]:
        _model["loaded"] = True
        if os.path.exists(MODEL_PATH):
            try:
                import joblib
                _model["obj"] = joblib.load(MODEL_PATH)
            except Exception:
                _model["obj"] = None
    return _model["obj"]


def model_available():
    return _get_model() is not None


def rule_check(text):
    hits = [r for r in _RULES if r["rx"].search(text)]
    p_safe = 1.0
    for r in hits:                        # combine like independent probabilities
        p_safe *= (1 - r["weight"])
    return 1 - p_safe, [r["reason"] for r in hits]


def check(job):
    text = " ".join(str(job.get(k, "")) for k in ("title", "company", "description"))
    rule_score, reasons = rule_check(text)
    model = _get_model()
    model_prob = None
    if model is not None and len(text.split()) > 30:   # too-short snippets confuse the model
        model_prob = float(model.predict_proba([text])[0][1])
        score = max(rule_score, 0.5 * rule_score + 0.5 * model_prob)
        if model_prob > 0.7:
            reasons.append("The trained model finds this text similar to known fake postings")
    else:
        score = rule_score
    level = "high" if score >= 0.6 else "caution" if score >= 0.3 else "low"
    return {"score": round(score, 2), "level": level, "reasons": reasons,
            "model_prob": None if model_prob is None else round(model_prob, 2)}
