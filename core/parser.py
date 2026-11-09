"""
Resume parsing: reads PDF/DOCX/TXT, extracts contact details, education,
experience, sections, skills, and flags personal details (bias-risk check)
and ATS formatting problems.
"""
import datetime as dt
import io
import re

from nlp.preprocess import clean_text
from nlp.sections import split_sections
from nlp.skills import extract_skills


class ResumeReadError(Exception):
    pass


def read_file(file_bytes, filename):
    """Return (text, ats_flags, pages)."""
    name = filename.lower()
    flags = []
    if name.endswith(".pdf"):
        try:
            import pdfplumber
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                pages = len(pdf.pages)
                texts, images, tables = [], 0, 0
                for p in pdf.pages:
                    texts.append(p.extract_text() or "")
                    images += len(p.images)
                    tables += len(p.find_tables())
                    words = p.extract_words()
                    if words and _looks_multicolumn(words, p.width):
                        flags.append("multi_column")
                text = "\n".join(texts)
            if images:
                flags.append("images")
            if tables:
                flags.append("tables")
        except Exception:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(file_bytes))
            pages = len(reader.pages)
            text = "\n".join(p.extract_text() or "" for p in reader.pages)
    elif name.endswith(".docx"):
        from docx import Document
        doc = Document(io.BytesIO(file_bytes))
        text = "\n".join(p.text for p in doc.paragraphs)
        for t in doc.tables:   # many templates hide content in tables
            for row in t.rows:
                text += "\n" + " | ".join(c.text for c in row.cells)
        if doc.tables:
            flags.append("tables")
        if doc.inline_shapes:
            flags.append("images")
        pages = max(1, round(len(text.split()) / 450))
    elif name.endswith(".txt"):
        text = file_bytes.decode("utf-8", errors="ignore")
        pages = max(1, round(len(text.split()) / 450))
    else:
        raise ResumeReadError("Upload a PDF, DOCX or TXT file.")
    if len(text.strip()) < 80:
        raise ResumeReadError("Very little text could be read. If this is a scanned PDF, "
                              "upload a text-based PDF or a DOCX file instead.")
    return clean_text(text), sorted(set(flags)), pages


def _looks_multicolumn(words, width):
    """Heuristic: many lines start near the middle of the page -> two columns."""
    mid_starts = sum(1 for w in words if 0.4 * width < w["x0"] < 0.6 * width)
    return mid_starts > 0.15 * len(words) and len(words) > 80


EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
LINKS = {"linkedin": re.compile(r"linkedin\.com/in/[\w-]+", re.I), "github": re.compile(r"github\.com/[\w-]+", re.I)}
DEGREES = ["b.e", "b.tech", "m.tech", "m.e", "bca", "mca", "b.sc", "bsc", "m.sc", "msc", "mba", "bba", "b.com", "phd", "diploma"]
PERSONAL = {
    "date_of_birth": (r"\b(date of birth|d\.?o\.?b\.?)\b", "Date of birth"),
    "gender": (r"\bgender\s*[:\-]|\bsex\s*[:\-]", "Gender"),
    "marital": (r"\bmarital status\b|\b(married|unmarried|single)\b\s*$", "Marital status"),
    "religion": (r"\breligion\b|\bcaste\b", "Religion or caste"),
    "father": (r"\b(father'?s|mother'?s) name\b", "Parent's name"),
    "nationality": (r"\bnationality\b", "Nationality"),
    "age": (r"\bage\s*[:\-]\s*\d{2}\b", "Age"),
    "photo": (r"\bphoto(graph)?\b", "Photo"),
    "address": (r"\b(permanent|residential|home) address\b|\bpin\s*code\b", "Full home address"),
}


def extract_phone(text):
    for cand in re.findall(r"\+?[\d\s()-]{9,20}\d", text):
        digits = re.sub(r"\D", "", cand)
        if 10 <= len(digits) <= 13 and not re.fullmatch(r"(19|20)\d{2}\D*(19|20)\d{2}", cand.strip()):
            return cand.strip()
    return None


def extract_name(header_lines):
    for ln in header_lines[:4]:
        words = ln.split()
        if 1 < len(words) <= 4 and all(w.replace(".", "").isalpha() for w in words) and not EMAIL.search(ln):
            return ln.title()
    return None


def extract_education(text):
    low = text.lower()
    degrees = sorted({d.upper() for d in DEGREES if re.search(r"(?<![a-z])" + re.escape(d) + r"(?![a-z])", low)})
    cgpa = re.search(r"\b(cgpa|gpa|sgpa)\s*[:\-]?\s*(\d{1,2}(?:\.\d{1,2})?)", low)
    years = [int(y) for y in re.findall(r"\b(20[1-3]\d)\b", text)]
    grad_year = max(years) if years else None
    return {"degrees": degrees, "cgpa": float(cgpa.group(2)) if cgpa and float(cgpa.group(2)) <= 10 else None,
            "graduation_year": grad_year}


def extract_experience(sections, text):
    """Years of experience from explicit phrases and number of internships."""
    yrs = [float(y) for y in re.findall(r"(\d+(?:\.\d+)?)\+?\s*(?:years|yrs)\b", text.lower()) if float(y) < 30]
    months = [float(m) for m in re.findall(r"(\d+)\s*months?", " ".join(sections.get("experience", [])).lower())]
    internships = len(re.findall(r"\bintern(ship)?\b", " ".join(sections.get("experience", [])), re.I))
    total = max(yrs) if yrs else round(sum(months) / 12, 1)
    return min(total, 30.0), internships


def personal_details(text):
    found = []
    for key, (pat, label) in PERSONAL.items():
        if re.search(pat, text, re.I | re.M):
            found.append(label)
    return found


def level_from(exp_years, grad_year, internships):
    this_year = dt.date.today().year
    if exp_years >= 1.5:
        return "experienced"
    if grad_year and grad_year > this_year:
        return "student"
    return "fresher"


def analyze_resume(file_bytes, filename):
    text, ats_flags, pages = read_file(file_bytes, filename)
    return analyze_text(text, ats_flags, pages)


def analyze_text(text, ats_flags=None, pages=1):
    sections = split_sections(text)
    skills = extract_skills(text, sections)
    edu = extract_education(" ".join(sections.get("education", [])) or text)
    exp_years, internships = extract_experience(sections, text)
    emails = EMAIL.findall(text)
    return {
        "text": text,
        "name": extract_name(sections.get("header", []) or text.split("\n")[:4]),
        "email": emails[0] if emails else None,
        "phone": extract_phone(text),
        "links": {k: bool(p.search(text)) for k, p in LINKS.items()},
        "education": edu,
        "experience_years": exp_years,
        "internships": internships,
        "level": level_from(exp_years, edu["graduation_year"], internships),
        "sections": sections,
        "skills": skills["skills"],
        "learning": skills["learning"],
        "possible": skills["possible"],
        "personal_flags": personal_details(text),
        "ats_flags": ats_flags or [],
        "pages": pages,
        "word_count": len(text.split()),
    }
