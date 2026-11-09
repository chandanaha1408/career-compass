"""Website tests using Flask's test client (no browser, no real server).
Run:  python -m pytest tests   or   python -m unittest discover tests

Each test uses a fresh temporary database and light mode (no neural models, no internet),
so the tests are fast and never touch your real data.
"""
import io
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config                                                # noqa: E402
from app import create_app                                   # noqa: E402
from core.jobs_api import _short, safe_url                   # noqa: E402
from core.llm import scrub                                   # noqa: E402
from nlp import models                                       # noqa: E402
from web.auth import is_safe_next                            # noqa: E402

SAMPLE = os.path.join(os.path.dirname(__file__), "samples", "sample_resume.txt")
TOKEN = "test-token"


class WebTests(unittest.TestCase):
    def setUp(self):
        # temporary database + light mode, restored afterwards so other tests are not affected
        self.tmp = tempfile.TemporaryDirectory()
        self.saved = (config.DB_PATH, config.LIGHT_MODE, dict(models._cache))
        config.DB_PATH = os.path.join(self.tmp.name, "test.db")
        config.LIGHT_MODE = True
        models._cache.clear()
        self.client = create_app().test_client()

    def tearDown(self):
        config.DB_PATH, config.LIGHT_MODE, cache = self.saved
        models._cache.clear()
        models._cache.update(cache)
        self.tmp.cleanup()

    def post(self, url, data=None, **kw):
        # login/signup clear the session (which removes the CSRF token), so put a known token back each time
        with self.client.session_transaction() as s:
            s["csrf"] = TOKEN
        return self.client.post(url, data={"csrf_token": TOKEN, **(data or {})}, **kw)

    def signup(self, email="asha@example.com"):
        return self.post("/signup", {"name": "Asha", "email": email, "password": "secret123"})

    def upload_sample(self):
        with open(SAMPLE, "rb") as f:
            return self.post("/profile/upload", {"resume": (io.BytesIO(f.read()), "resume.txt")},
                             content_type="multipart/form-data")

    # ---- accounts
    def test_signup_and_login(self):
        self.assertEqual(self.signup().status_code, 302)
        self.post("/logout")
        r = self.post("/login", {"email": "asha@example.com", "password": "secret123"})
        self.assertEqual(r.status_code, 302)
        self.assertTrue(r.headers["Location"].endswith("/profile"))

    def test_wrong_password(self):
        self.signup()
        self.post("/logout")
        r = self.post("/login", {"email": "asha@example.com", "password": "wrong"})
        self.assertEqual(r.status_code, 200)        # stays on the login page
        self.assertIn(b"incorrect", r.data)

    def test_pages_need_login(self):
        r = self.client.get("/profile")
        self.assertEqual(r.status_code, 302)
        self.assertIn("/login", r.headers["Location"])

    # ---- security
    def test_csrf_rejected(self):
        r = self.client.post("/signup", data={"name": "X", "email": "x@example.com", "password": "secret123",
                                              "csrf_token": "wrong"})
        self.assertEqual(r.status_code, 400)

    def test_open_redirect_blocked(self):
        for bad in ("//evil.com", "/\\evil.com", "https://evil.com", ""):
            self.assertFalse(is_safe_next(bad), bad)
        self.assertTrue(is_safe_next("/jobs"))
        self.signup()
        self.post("/logout")
        r = self.post("/login?next=/%5Cevil.com", {"email": "asha@example.com", "password": "secret123"})
        self.assertTrue(r.headers["Location"].endswith("/profile"))

    def test_api_keys_hidden_in_errors(self):
        msg = "404 Client Error for url: https://api.adzuna.com/v1/api/jobs/in/search/1?app_id=ID123&app_key=KEY456"
        self.assertNotIn("KEY456", _short(Exception(msg)))
        self.assertNotIn("AIzaSECRET", scrub("Max retries exceeded with url: /v1beta/m:generateContent?key=AIzaSECRET (x)"))

    def test_only_web_links_allowed(self):
        self.assertEqual(safe_url("javascript:alert(1)"), "")
        self.assertEqual(safe_url(None), "")
        self.assertEqual(safe_url("https://example.com/job"), "https://example.com/job")

    def test_job_text_uses_description_when_no_requirements(self):
        from core.matcher import _job_text
        job = {"title": "Data Analyst", "description": "Short snippet about dashboards"}
        self.assertIn("dashboards", _job_text(job, {"requirements": []}))

    # ---- uploads and pages
    def test_wrong_file_type(self):
        self.signup()
        r = self.post("/profile/upload", {"resume": (io.BytesIO(b"MZ fake program"), "virus.exe")},
                      content_type="multipart/form-data", follow_redirects=True)
        self.assertIn(b"Upload a PDF, DOCX or TXT file", r.data)

    def test_upload_and_jobs(self):
        self.signup()
        r = self.upload_sample()
        self.assertEqual(r.status_code, 302)
        self.assertEqual(self.client.get("/profile").status_code, 200)
        self.assertEqual(self.client.get("/jobs").status_code, 200)
        self.assertEqual(self.client.get("/jobs?min=abc").status_code, 200)     # used to crash with 500

    def test_huge_pasted_jd_is_trimmed(self):
        self.signup()
        self.upload_sample()
        jd = "Requirements: Python and SQL. " * 5000                             # about 150,000 characters
        r = self.post("/match/custom", {"jd": jd, "title": "Big JD"})
        self.assertEqual(r.status_code, 200)


if __name__ == "__main__":
    unittest.main()
