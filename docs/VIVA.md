# Viva preparation

## Project in one minute
Career Compass reads a student's resume with NLP, predicts suitable roles, fetches
real job postings, ranks them with a two-stage semantic matcher, explains every
requirement, flags fake jobs, and builds a study plan with free resources.
It is a Flask website with SQLite, running locally, with fallbacks so it works offline.

## Likely questions

**Why not just keyword matching?**
Keywords miss synonyms and context ("built REST backends" vs "API development") and can be gamed by
stuffing. We combine meaning (embeddings), skills with evidence, and keywords.

**What is TF-IDF?**
Term frequency x inverse document frequency: a word scores high if it is frequent in one document
but rare across others. Texts become vectors; cosine similarity compares them.

**What is cosine similarity?**
The cosine of the angle between two vectors: 1 = same direction (similar), 0 = unrelated.
It ignores document length.

**What are sentence embeddings?**
A transformer model (all-MiniLM-L6-v2) turns a sentence into a 384-number vector so sentences
with similar meaning are close together, even with different words.

**Bi-encoder vs cross-encoder?**
A bi-encoder embeds resume and job separately: fast, used for all jobs. A cross-encoder reads both
texts together with attention between them: more accurate but slower, so it only re-ranks the top 20.

**How are skills extracted?**
Three layers: dictionary with aliases and safe word boundaries; semantic mapping of noun phrases to
known skills; discovery of unknown tech terms for the user to confirm. Context rules stop
"learning Docker" or "no experience with Java" from counting.

**How do you decide skill level?**
From evidence location: skills section only = beginner, one project = intermediate,
several projects or internship = advanced.

**How do you know must-have vs nice-to-have?**
Cue words (required, must, strong vs preferred, plus, bonus) and labelled blocks ("Preferred:").
"Django or Flask" becomes one alternative group satisfied by either.

**What is TextRank?**
Sentences are graph nodes, similarities are edges, and PageRank finds the most central sentences
for the 3-line summary.

**What is KeyBERT / MMR?**
Candidate phrases are embedded and compared to the document; Maximal Marginal Relevance balances
relevance and diversity so results aren't near-duplicates.

**How does fake job detection work?**
Transparent rules (fees, WhatsApp-only, Aadhaar/OTP, guaranteed jobs) give reasons; a TF-IDF +
logistic regression model trained on the Kaggle dataset adds a probability. The dataset is
imbalanced (few fakes), so we use class_weight="balanced" and report precision, recall and F1.

**Why precision/recall instead of accuracy?**
With ~5% fakes, predicting "real" for everything gives 95% accuracy but catches no scams.

**How is bias handled?**
Personal details (DOB, gender, religion, photo) are flagged for removal and never used in scoring.

**What does the LLM do and how do you stop hallucination?**
Gemini only rewrites bullets for a chosen job. The prompt forbids adding skills or numbers
([add number] placeholders), output is JSON, and it is labelled AI-generated for review.

**Security?**
Passwords are salted hashes (Werkzeug), every form has a CSRF token, uploads are size-limited,
and API keys stay in the local database.

**What if the internet or models are unavailable?**
Each component falls back: sample jobs, rule-based feedback, TF-IDF instead of embeddings.

**Limitations**
Small labelled evaluation set; job API snippets can be short (Adzuna); resume layouts with tables
or two columns can extract imperfectly; skill taxonomy is curated, not exhaustive.

## Future scope
- Fine-tune embeddings on real resume-job pairs
- Larger skill taxonomy (ESCO) and Indian-language resumes
- Recruiter mode with bulk screening and blind review
- Cloud deployment and mobile-friendly PWA
- Collect user feedback on matches to learn better weights
