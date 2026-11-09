"""
Train the fake job classifier.

1. Download "Real / Fake Job Posting Prediction" (fake_job_postings.csv) from Kaggle.
2. Put the CSV in the data/ folder.
3. Run:  python scripts/train_fake_job.py

Compares rules only, TF-IDF + logistic regression and (if installed) sentence
embeddings + logistic regression. Saves the TF-IDF model to models/ (the website
loads it automatically) and writes docs/results/fake_job.md
"""
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from core.fake_job import rule_check   # noqa: E402
from nlp import models                 # noqa: E402

CSV = os.path.join(ROOT, "data", "fake_job_postings.csv")
if not os.path.exists(CSV):
    sys.exit("Put fake_job_postings.csv (from Kaggle) in the data/ folder first.")

df = pd.read_csv(CSV).fillna("")
cols = ["title", "company_profile", "description", "requirements", "benefits"]
df["text"] = df[cols].agg(" ".join, axis=1)
X_tr, X_te, y_tr, y_te = train_test_split(df["text"], df["fraudulent"], test_size=0.2, stratify=df["fraudulent"], random_state=42)
print(f"Train {len(X_tr)}, test {len(X_te)}, fake share {df['fraudulent'].mean():.1%} (imbalanced -> class_weight='balanced')")

rows = []
def report(name, pred):
    p, r, f, _ = precision_recall_fscore_support(y_te, pred, average="binary", zero_division=0)
    rows.append((name, p, r, f))
    print(f"{name:35s} precision {p:.2f}  recall {r:.2f}  F1 {f:.2f}")

report("Rules only", [int(rule_check(t)[0] >= 0.3) for t in X_te])

tfidf_model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True),
                            LogisticRegression(max_iter=2000, class_weight="balanced"))
tfidf_model.fit(X_tr, y_tr)
report("TF-IDF + logistic regression", tfidf_model.predict(X_te))

if models.get_embedder() is not None:
    print("Embedding texts (takes a few minutes on CPU)...")
    enc = lambda s: models.get_embedder().encode([t[:2000] for t in s], batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(enc(X_tr.tolist()), y_tr)
    report("Embeddings + logistic regression", clf.predict(enc(X_te.tolist())))
else:
    print("sentence-transformers not available: skipping the embedding model.")

os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
joblib.dump(tfidf_model, os.path.join(ROOT, "models", "fake_job_model.joblib"))
md = ["# Fake job detection results", "", "| Method | Precision | Recall | F1 |", "|---|---|---|---|"]
md += [f"| {n} | {p:.2f} | {r:.2f} | {f:.2f} |" for n, p, r, f in rows]
os.makedirs(os.path.join(ROOT, "docs", "results"), exist_ok=True)
open(os.path.join(ROOT, "docs", "results", "fake_job.md"), "w").write("\n".join(md) + "\n")
print("Saved models/fake_job_model.joblib - restart the website to use it.")
