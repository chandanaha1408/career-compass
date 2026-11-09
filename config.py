"""Central settings for Career Compass. Edit values here rather than inside modules."""
import os
import secrets

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
INSTANCE_DIR = os.path.join(BASE_DIR, "instance")
DB_PATH = os.path.join(INSTANCE_DIR, "career_compass.db")
os.makedirs(INSTANCE_DIR, exist_ok=True)

# NLP models (downloaded automatically on first use, then cached on disk)
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CROSS_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
SPACY_MODEL = "en_core_web_sm"
# Set CC_LIGHT_MODE=1 to skip the neural models (TF-IDF fallback only)
LIGHT_MODE = os.environ.get("CC_LIGHT_MODE", "0") == "1"

# Job fit score weights (must add up to 1.0)
FIT_WEIGHTS = {"semantic": 0.40, "skills": 0.35, "keywords": 0.15, "fit": 0.10}
# Readiness score weights (must add up to 1.0)
READINESS_WEIGHTS = {"coverage": 0.35, "depth": 0.20, "portfolio": 0.15, "resume": 0.15, "fundamentals": 0.15}
VERDICTS = [(70, "Strong match"), (55, "Good match"), (40, "Partial match"), (0, "Weak match")]
RERANK_TOP_N = 20            # how many jobs the cross-encoder re-ranks
JOB_CACHE_HOURS = 12         # reuse API results for this long to save free quota
SEMANTIC_SKILL_THRESHOLD = 0.72  # min similarity to map a phrase to a known skill

def _secret_key():
    path = os.path.join(INSTANCE_DIR, "secret.key")
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(secrets.token_hex(32))
    return open(path).read().strip()

SECRET_KEY = os.environ.get("CC_SECRET_KEY") or _secret_key()
MAX_UPLOAD_MB = 5
