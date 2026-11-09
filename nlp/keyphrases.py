"""
Keyphrase extraction (the KeyBERT technique) used to discover emerging skills in
job postings that are not yet in our skill list.

1. Candidate phrases: 1-3 word n-grams without stop words.
2. Each candidate and the whole document are embedded.
3. Candidates most similar to the document are chosen, with Maximal Marginal
   Relevance (MMR) so the results are diverse rather than near-duplicates.
Fallback without embeddings: highest TF-IDF scoring phrases.
"""
import re

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from nlp import models

GENERIC = set("""experience team work skills knowledge strong good ability years candidate candidates role job company
customers customer business clients client opportunity responsibilities requirements preferred plus required
understanding environment working develop development developer developers build building using use etc including
excellent communication written verbal degree engineering bachelor freshers fresher salary stipend month apply
looking join hiring growth new projects project tools tool based high quality support solutions services product
products design time end best world apis api basic basics data analytics backend apps app familiarity
applications application deployment learning interest concepts programming knowledge exposure hands strong solid
like used uses builds growing fast proficient proficiency similar related tasks real systems system platform platforms scalable"""
.split())


def _candidates(doc, ngram=(1, 3)):
    try:
        cv = CountVectorizer(ngram_range=ngram, stop_words="english", token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9+#.]+\b")
        cv.fit([doc])
    except ValueError:
        return []
    out = []
    for c in cv.get_feature_names_out():
        words = c.split()
        if any(w in GENERIC for w in words) or len(c) < 3 or re.search(r"\b(b\.\w+|m\.\w+|rs|lpa|\d+)\b", c):
            continue
        out.append(c)
    return out


def extract(doc, top_n=5, diversity=0.5):
    cands = _candidates(doc)
    if not cands:
        return []
    vecs = models.embed([doc[:2000]] + cands[:300])
    if vecs is None:
        return _tfidf_fallback(doc, cands, top_n)
    doc_vec, cand_vec = vecs[:1], vecs[1:]
    doc_sim = (cand_vec @ doc_vec.T).ravel()
    chosen = [int(doc_sim.argmax())]
    while len(chosen) < min(top_n, len(cands[:300])):
        rest = [i for i in range(len(cand_vec)) if i not in chosen]
        redundancy = (cand_vec[rest] @ cand_vec[chosen].T).max(axis=1)
        mmr = (1 - diversity) * doc_sim[rest] - diversity * redundancy
        chosen.append(rest[int(mmr.argmax())])
    return [cands[i] for i in chosen]


def _tfidf_fallback(doc, cands, top_n):
    tf = TfidfVectorizer(vocabulary=list(dict.fromkeys(cands)), ngram_range=(1, 3),
                         token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9+#.]+\b")
    scores = tf.fit_transform([doc.lower()]).toarray()[0]
    order = np.argsort(-scores)
    return [tf.get_feature_names_out()[i] for i in order[:top_n] if scores[i] > 0]
