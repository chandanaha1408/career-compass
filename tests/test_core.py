"""Automated tests. Run:  python -m unittest discover tests"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.fake_job import check                              # noqa: E402
from core.matcher import rank_jobs                           # noqa: E402
from core.parser import analyze_text, extract_phone          # noqa: E402
from core.reviewer import analyze_bullets                    # noqa: E402
from nlp.jd_analyzer import alternative_groups, analyze      # noqa: E402
from nlp.skills import dictionary_skills, extract_skills     # noqa: E402

SAMPLE = open(os.path.join(os.path.dirname(__file__), "samples", "sample_resume.txt"), encoding="utf-8").read()


class SkillTests(unittest.TestCase):
    def test_aliases_and_symbols(self):
        found = dictionary_skills("Worked with sklearn, C++, C#, Node.js and CI/CD.")
        for s in ("scikit-learn", "c++", "c#", "node.js", "ci/cd"):
            self.assertIn(s, found)

    def test_java_not_inside_javascript(self):
        self.assertNotIn("java", dictionary_skills("Strong JavaScript skills"))

    def test_learning_not_counted(self):
        r = extract_skills("Skills: Python\nCurrently learning Docker.")
        self.assertIn("python", r["skills"])
        self.assertIn("docker", r["learning"])
        self.assertNotIn("docker", r["skills"])

    def test_machine_learning_is_not_a_learning_cue(self):
        r = extract_skills("Skills: Machine Learning, pandas, numpy")
        self.assertIn("pandas", r["skills"])

    def test_learn_inside_skill_names_is_not_learning(self):
        # "scikit-learn" and a cut-off "Machine Learning" used to mark the whole line as learning
        r = extract_skills("AI/ML: Machine Learning, NLP, scikit-learn, pandas, numpy, TensorFlow")
        for s in ("scikit-learn", "pandas", "numpy", "tensorflow"):
            self.assertIn(s, r["skills"])
            self.assertNotIn(s, r["learning"])
        r = extract_skills("Currently learning machine learning with TensorFlow")
        self.assertIn("tensorflow", r["learning"])

    def test_semester_not_sem(self):
        self.assertNotIn("sem", dictionary_skills("Sem 5 results"))


class ParserTests(unittest.TestCase):
    def test_sample_resume(self):
        p = analyze_text(SAMPLE)
        self.assertEqual(p["name"], "Rahul Sharma")
        self.assertEqual(p["email"], "rahul.sharma@email.com")
        self.assertEqual(p["education"]["cgpa"], 8.6)
        self.assertIn("projects", p["sections"])

    def test_phone(self):
        self.assertEqual(extract_phone("call +91 98765 43210 now"), "+91 98765 43210")
        self.assertIsNone(extract_phone("2023 - 2027"))


class JobTests(unittest.TestCase):
    def test_must_vs_nice(self):
        a = analyze("Requirements: Strong Python and SQL. Preferred: Docker is a plus.")
        self.assertIn("python", a["must"])
        self.assertIn("docker", a["nice"])

    def test_alternatives(self):
        self.assertEqual(alternative_groups("Experience with Django or Flask."), [{"django", "flask"}])

    def test_alternative_satisfied(self):
        p = analyze_text("Skills: Python, Flask, SQL, Git\nProjects\nBuilt a Flask REST API with SQL.")
        job = {"title": "Python Dev", "description": "Requirements: Python. REST APIs with Django or Flask."}
        r = rank_jobs({**p, "extra_skills": {}}, [job], rerank=False)[0]
        self.assertEqual(r["missing_must"], [])

    def test_fake_job_rules(self):
        fake = check({"title": "Earn daily", "description": "Pay registration fee of Rs 500. WhatsApp only. Share OTP."})
        self.assertEqual(fake["level"], "high")
        real = check({"title": "Python Developer", "description": "Requirements: Python and SQL. Apply on our careers page."})
        self.assertEqual(real["level"], "low")


class ReviewerTests(unittest.TestCase):
    def test_bullets(self):
        b = analyze_bullets({"projects": ["Responsible for the backend of the college app",
                                          "Built a Flask API used by 400 students"]})
        self.assertTrue(b[0]["issues"])
        self.assertEqual(b[1]["issues"], [])


if __name__ == "__main__":
    unittest.main()
