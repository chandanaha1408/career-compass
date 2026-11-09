"""
Optional Gemini integration (free key from Google AI Studio).
Used only for tailored resume suggestions. Strict instructions forbid adding
skills or experience that are not in the original resume.
"""
import json
import re

import requests

DEFAULT_MODEL = "gemini-2.5-flash"
URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

PROMPT = """You are a careful resume coach for Indian students and freshers.
Improve the resume for the job below. STRICT RULES:
- Never add skills, tools, companies, numbers or experience that are not in the resume.
- If a metric is missing, write [add number] as a placeholder instead of inventing one.
- Keep the person's meaning; improve clarity, action verbs and relevance to the job.
Return ONLY valid JSON with this shape:
{{"summary": "2-3 sentence professional summary tailored to the job",
  "bullets": [{{"original": "...", "improved": "..."}}],
  "advice": ["short, specific tip", "..."],
  "honest_gaps": ["job requirement the resume does not show"]}}
Rewrite at most 6 bullets.

JOB:
{job}

RESUME:
{resume}"""


class LLMError(Exception):
    pass


def scrub(text):
    """Remove anything after '?' in URLs so API keys passed as query parameters are never shown."""
    return re.sub(r"(https?://\S*?|/\S*?)\?\S*", r"\1?[hidden]", str(text))


def generate(prompt, api_key, model=None):
    if not api_key:
        raise LLMError("Add a Gemini API key in Settings to use AI suggestions.")
    try:
        # the key goes in a header, not the URL, so it can never appear in an error message or log
        resp = requests.post(URL.format(model=model or DEFAULT_MODEL), headers={"x-goog-api-key": api_key}, timeout=60,
                             json={"contents": [{"parts": [{"text": prompt}]}],
                                   "generationConfig": {"temperature": 0.4, "responseMimeType": "application/json"}})
    except requests.RequestException as e:
        raise LLMError(f"Could not reach Gemini ({type(e).__name__}). Check your internet connection.")
    if resp.status_code != 200:
        try:
            msg = resp.json().get("error", {}).get("message", "")
        except ValueError:
            msg = ""
        raise LLMError(f"Gemini returned an error ({resp.status_code}): {scrub(msg or resp.text)[:200]}")
    try:
        return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        raise LLMError("Gemini returned an empty response. Try again.")


def tailor(resume_text, job_text, api_key, model=None):
    raw = generate(PROMPT.format(job=job_text[:4000], resume=resume_text[:6000]), api_key, model)
    cleaned = re.sub(r"^```(json)?|```$", "", raw.strip(), flags=re.M).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return {"summary": "", "bullets": [], "advice": [cleaned[:1500]], "honest_gaps": []}
    for k in ("bullets", "advice", "honest_gaps"):
        data.setdefault(k, [])
    data.setdefault("summary", "")
    return data
