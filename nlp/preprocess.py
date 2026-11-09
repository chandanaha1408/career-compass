"""
Text preprocessing: cleaning, sentence splitting, tokenisation and lemmatisation.
Uses spaCy when available, otherwise simple regex rules.
"""
import re
import unicodedata

from nlp.models import get_spacy

BULLETS = "•●▪■◦‣∙·–—*➢➤►✓✔"


def clean_text(text):
    """Normalise unicode, unify bullets and whitespace, keep line breaks."""
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("\r", "\n").replace("\t", " ")
    text = re.sub(f"[{re.escape(BULLETS)}]", "\n- ", text)
    text = re.sub(r"[ \u00a0]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def lines(text):
    """Non-empty, stripped lines (resumes are line-oriented documents)."""
    out = []
    for ln in clean_text(text).split("\n"):
        ln = ln.strip(" -:|")
        if ln:
            out.append(ln)
    return out


def sentences(text):
    """Split text into sentences. spaCy's parser handles abbreviations better."""
    text = clean_text(text)
    nlp = get_spacy()
    if nlp is not None:
        doc = nlp(text[:100000])
        sents = [s.text.strip() for s in doc.sents]
    else:
        sents = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])|\n+", text)
    return [s.strip(" -") for s in sents if len(s.strip(" -")) > 2]


def tokens(text, lemmatize=True):
    """Lowercase tokens without stop words; lemmas when spaCy is present."""
    nlp = get_spacy()
    if nlp is not None:
        doc = nlp(text[:100000])
        return [(t.lemma_ if lemmatize else t.text).lower() for t in doc
                if not t.is_stop and not t.is_punct and not t.is_space]
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    return [w for w in re.findall(r"[a-z0-9+#./]+", text.lower()) if w not in ENGLISH_STOP_WORDS]


def normalize_for_matching(text):
    """Lowercased single-line text used by keyword and skill matching."""
    return re.sub(r"\s+", " ", clean_text(text).lower())
