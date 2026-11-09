# Changes (November 2026)

This folder is not a git repository, so this file is the record of what changed and why.
Update your viva explanations for these files.

## Test results
| | Before | After |
|---|---|---|
| Command | `python -m unittest discover tests` (pytest was not installed) | `python -m pytest tests` (unittest also works) |
| Result | 12 tests, all passed | 24 tests, all passed (12 old + 11 new website tests + 1 skill-context test) |

## Bug fixes

### 1. Short job postings were matched on the title only (`core/matcher.py`, `_job_text`)
**What:** Added brackets: `title + ". " + (requirements or description)`.
**Why:** In Python, `+` runs before `or`. The old line was read as `(title + ". " + requirements) or description`.
The left side always contains at least `". "`, so it is never empty, and the description was never used.
For postings with no detected requirement lines (for example, short Adzuna snippets), only the job title went into
the similarity and keyword scores. Now the description is used, as the original design intended.

### 2. The Jobs page crashed on a bad filter value (`web/routes.py`, `jobs`)
**What:** `request.args.get("min", 0, type=int)`, limited to the range 0 to 100.
**Why:** `int("abc")` raised a ValueError, so `/jobs?min=abc` showed a 500 error. Flask's `type=int` falls back
to the default value when the conversion fails.

### 3. Pasted job descriptions are size-limited (`web/routes.py`, `match_custom`)
**What:** Pasted text is trimmed to `MAX_JD_CHARS = 20000` characters.
**Why:** With no limit, a very large paste kept the NLP pipeline busy for a long time. A real job description is far
shorter than 20,000 characters.

## Security fixes

### 4. API keys could appear on screen in error messages (`core/llm.py`, `core/jobs_api.py`)
**What:**
- Gemini: the key is now sent in the `x-goog-api-key` **header** instead of in the URL (`?key=...`).
- New function `scrub()` in `core/llm.py` hides everything after `?` in a URL. `_short()` in
  `core/jobs_api.py` uses it for the JSearch and Adzuna error notes.
- Network errors now show a short message such as "Could not reach Gemini (ConnectionError)".

**Why:** The `requests` library includes the full URL in its error messages. Adzuna sends `app_id` and `app_key` in
the URL, and Gemini used to do the same. When the internet failed, the secret key was shown on the page, and the
Adzuna notes were also saved in the `results` table.

### 5. Open redirect after login (`web/auth.py`, new `is_safe_next`)
**What:** The `?next=` value must start with `/`, must not start with `//`, and must not contain `\`.
**Why:** The old check blocked `//evil.com` but not `/\evil.com`. Browsers treat `\` like `/`, so a link such as
`/login?next=/\evil.com` could send a user to another website straight after they logged in. Attackers use this
trick in phishing.

### 6. Only normal web links for job postings (`core/jobs_api.py`, new `safe_url`)
**What:** Apply links from JSearch and Adzuna are kept only if they start with `http://` or `https://`.
**Why:** These links come from outside the project and are placed in `href="..."`. Jinja escapes quotes, but it
cannot stop a `javascript:` link, which runs code when clicked. Now such a link is dropped.

## Tests

### 7. New website tests (`tests/test_web.py`)
Flask's test client sends requests without a browser. Each test uses a **temporary database** and **light mode**,
so the tests are fast, need no internet, and never touch your real data. The tests cover:
- signup and login, wrong password, and pages that need login
- a form with a wrong CSRF token is rejected (400)
- the open-redirect check (fix 5)
- API keys are hidden in error messages (fix 4)
- `javascript:` links are dropped (fix 6)
- uploading a `.exe` gives the "Upload a PDF, DOCX or TXT file" message
- uploading the sample resume and opening the Profile and Jobs pages, including `?min=abc` (fix 2)
- a 150,000-character pasted job description still works (fix 3)
- the job text uses the description when no requirements are found (fix 1)

The test helper sets the CSRF token again before every POST. Login and signup call `session.clear()`, which also
removes the token; this is correct security behaviour (it stops "session fixation" attacks).

### 8. pytest installed (`requirements.txt`)
`pytest` was installed in the venv and added to `requirements.txt`, so `python -m pytest tests` works.
The old command `python -m unittest discover tests` still works too.

## Later fix: known skills wrongly marked as "learning" (`nlp/skills.py`, `_context`)
**What:** `_context` now finds the skill names in the sentence first (`skill_spans`). A "learn" word that is part of a
skill name, such as scikit-**learn** or machine **learning**, is no longer treated as learning intent. This applies
to both the regex cue check and the spaCy dependency check. New test: `test_learn_inside_skill_names_is_not_learning`.
**Why:** In the sample resume, the line "AI/ML: Machine Learning, NLP, scikit-learn, pandas, numpy, TensorFlow" put
pandas, numpy, scikit-learn and TensorFlow under "Learning (not counted yet)". There were two causes:
1. spaCy splits "scikit-learn" into `scikit` `-` `learn` and tags `learn` as a verb. pandas and numpy hang under it in
   the parse tree, so the "governed by learn" rule fired.
2. The 45-character look-back window cut "Machine Learning" to just "Learning", so the regex rule that ignores
   "machine learning" could no longer see the word "machine".

**Result:** the sample resume now shows 26 skills instead of 23. Data Scientist rises from 64% to 71%, and pandas is no
longer a missing must-have. "Currently learning Docker" and "No experience with Java" still work as before.
Tests: 24 passed.

## Found but not changed (you chose to leave these)
- Two signups with the same email at the same moment can hit the database UNIQUE rule and show a 500 error (`web/auth.py`).
- `nlp/skills.py` `_context` runs spaCy on the same sentence once for every skill found in it (slower uploads with full models).
- `core/matcher.py` `_analysis_cache` keeps growing while the server runs.
