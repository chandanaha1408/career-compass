"""Pre-download all NLP models so the first website start is fast. Run once with internet."""
import subprocess
import sys

subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=False)
from sentence_transformers import CrossEncoder, SentenceTransformer  # noqa: E402
SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
print("All models downloaded.")
