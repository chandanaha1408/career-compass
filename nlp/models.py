"""
Loads the heavy NLP models once and shares them across the app.

Every loader fails gracefully: if a library or model is missing, it returns None
and the rest of the system falls back to lighter techniques (regex, TF-IDF).
This keeps the website usable on any machine and during offline demos.
"""
import logging
import numpy as np

import config

log = logging.getLogger(__name__)
_cache = {}


def _load(name, loader):
    if name not in _cache:
        if config.LIGHT_MODE:
            _cache[name] = None
        else:
            try:
                _cache[name] = loader()
                log.info("Loaded %s", name)
            except Exception as e:  # library missing, no internet on first run, etc.
                log.warning("Could not load %s (%s). Using fallback.", name, e)
                _cache[name] = None
    return _cache[name]


def get_spacy():
    def loader():
        import spacy
        return spacy.load(config.SPACY_MODEL)
    return _load("spacy", loader)


def get_embedder():
    def loader():
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(config.EMBED_MODEL)
    return _load("embedder", loader)


def get_cross_encoder():
    def loader():
        from sentence_transformers import CrossEncoder
        return CrossEncoder(config.CROSS_MODEL)
    return _load("cross_encoder", loader)


def status():
    """Which NLP components are active (shown on the settings page)."""
    return {
        "spaCy (linguistic analysis)": get_spacy() is not None,
        "Sentence embeddings (semantic matching)": get_embedder() is not None,
        "Cross-encoder (re-ranking)": get_cross_encoder() is not None,
    }


_emb_cache = {}


def embed(texts):
    """Return L2-normalised embeddings as a numpy array, or None if unavailable.
    Short repeated texts (skill names, role descriptions) are cached."""
    model = get_embedder()
    if model is None or not texts:
        return None
    fresh = {}
    missing = [t for t in dict.fromkeys(texts) if t not in _emb_cache]
    if missing:
        vecs = model.encode(missing, normalize_embeddings=True, convert_to_numpy=True,
                            show_progress_bar=False, batch_size=32)
        if len(_emb_cache) > 20000:
            _emb_cache.clear()
        for t, v in zip(missing, vecs):
            fresh[t] = v
            if len(t) < 400:          # cache only short texts to bound memory
                _emb_cache[t] = v
    return np.vstack([_emb_cache[t] if t in _emb_cache else fresh[t] for t in texts])


def cosine_matrix(a, b):
    """Cosine similarity between rows of two normalised matrices."""
    return np.asarray(a) @ np.asarray(b).T


def cross_scores(pairs):
    """Relevance scores 0-1 for (query, passage) pairs using the cross-encoder."""
    model = get_cross_encoder()
    if model is None or not pairs:
        return None
    logits = np.asarray(model.predict(pairs, show_progress_bar=False), dtype=float)
    return 1 / (1 + np.exp(-logits))   # sigmoid turns raw logits into 0-1
