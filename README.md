# Career Compass
### AI-Based Resume Screening and Job Matching System

A website that runs on your own computer. Students upload a resume and get:
predicted roles, live jobs ranked by fit, a requirement-by-requirement explanation
of every match, fake-job warnings, market insights, resume feedback, and a
personalised study plan with free learning resources.

**Quick start:** see [docs/SETUP.md](docs/SETUP.md). In short:
```
pip install -r requirements.txt
python scripts/download_models.py
python app.py            # open http://localhost:5000
```

## Pages
| Page | What it does |
|---|---|
| Profile | Upload resume; skills with levels; learning vs known skills; possible new skills to confirm; top roles; bias-risk check |
| Jobs | Live postings (JSearch / Adzuna) or sample postings, ranked by fit; fake-job warnings; quick wins; links to Naukri, LinkedIn, Indeed, Internshala; paste any job description |
| Match details | Score breakdown, must-have vs nice-to-have skills, each requirement paired with your closest resume line, 3-line job summary |
| Market insights | Most-demanded skills for your role, your gaps, salary range, emerging terms |
| Resume feedback | Quality score, bullet analysis, ATS and bias checks, version comparison, Gemini suggestions |
| Prepare | Readiness score, fundamentals self-assessment, week-by-week plan, free resources, interview prep, project ideas |
| Tracker | Saved jobs with status, readiness-over-time chart, completed plan items |
| Settings | API keys, system status |

## How the code is organised
```
app.py              starts the website (Flask app, CSRF protection, error pages)
config.py           all weights, thresholds and model names in one place
nlp/                language processing
  models.py         loads spaCy, sentence embeddings and cross-encoder once; falls back if missing
  preprocess.py     cleaning, sentence splitting, tokens and lemmas
  sections.py       finds resume sections by heading, or by content when headings are missing
  skills.py         3-layer skill extraction, learning/negation context, skill levels
  jd_analyzer.py    must vs nice-to-have, "A or B" alternatives, years, degree, TextRank summary
  keyphrases.py     KeyBERT-style keyphrase extraction with MMR (emerging skills)
core/               application logic
  parser.py         reads PDF/DOCX/TXT; contact, education, experience, bias-risk, ATS flags
  skillset.py       the user's effective skills and coverage calculations
  roles.py          role prediction (skills + semantic similarity)
  matcher.py        two-stage job scoring (bi-encoder then cross-encoder) and explanations
  jobs_api.py       JSearch -> Adzuna -> sample data, caching, job-site links
  fake_job.py       red-flag rules + trained classifier
  market.py         skill demand, emerging terms, salaries
  reviewer.py       bullet analysis (action verbs, passive voice, metrics) and checks
  llm.py            Gemini suggestions with no-invention rules
  advisor.py        readiness, gaps, study plan, resources, interview prep, projects
web/                website: routes.py (pages), auth.py (accounts), db.py (SQLite), templates/, static/
data/               skills, roles, sample jobs, learning resources, project ideas, fake-job rules
scripts/            evaluate.py, train_fake_job.py, download_models.py
tests/              unit tests and labelled evaluation resumes
docs/               setup guide, architecture, viva Q&A, results
```

## Key formulas
- **Job fit** = 40% meaning similarity + 35% skill match + 15% keyword overlap + 10% level/education fit
- **Skill match**: must-have skills count double; a group like "Django or Flask" is satisfied by either;
  advanced/intermediate/beginner skills earn 1.0/0.9/0.75 credit; skills being learned earn 0.25
- **Readiness** = 35% coverage (weighted by market demand) + 20% depth + 15% portfolio + 15% resume quality
  + 15% placement fundamentals (fundamentals weight is redistributed for non-software roles)
- **Skill level**: listed only = beginner, one project = intermediate, several projects or internship = advanced

All weights live in `config.py`.

## Works offline
Every feature has a fallback: no job API key -> sample postings and job-site links;
no Gemini key -> rule-based feedback still works; NLP models missing -> TF-IDF methods.
Set `CC_LIGHT_MODE=1` to force the lightweight mode.

## Tests and evaluation
```
python -m unittest discover tests
python scripts/evaluate.py          # writes docs/results/evaluation.md
python scripts/train_fake_job.py    # after downloading the Kaggle dataset
```
The included labelled set has 12 clean resumes, so scores are high. Add real,
messier resumes to `tests/eval/labelled.json` for a more meaningful comparison.
