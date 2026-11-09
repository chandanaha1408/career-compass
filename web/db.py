"""SQLite storage. One small wrapper class keeps all SQL in one place."""
import datetime as dt
import json
import sqlite3

from flask import g

import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, name TEXT NOT NULL, email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL, created TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS resumes (id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL, filename TEXT,
    uploaded TEXT NOT NULL, analysis TEXT NOT NULL, quality INTEGER);
CREATE TABLE IF NOT EXISTS profiles (user_id INTEGER PRIMARY KEY, resume_id INTEGER, data TEXT NOT NULL,
    roles TEXT, targets TEXT, location TEXT, job_type TEXT, version TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS job_cache (key TEXT PRIMARY KEY, data TEXT NOT NULL, fetched_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS results (user_id INTEGER PRIMARY KEY, data TEXT NOT NULL, created TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS saved_jobs (id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL, job_id TEXT NOT NULL,
    job TEXT NOT NULL, score REAL, status TEXT NOT NULL DEFAULT 'saved', saved_at TEXT, updated_at TEXT,
    UNIQUE(user_id, job_id));
CREATE TABLE IF NOT EXISTS progress (user_id INTEGER NOT NULL, key TEXT NOT NULL, done_at TEXT NOT NULL,
    PRIMARY KEY (user_id, key));
CREATE TABLE IF NOT EXISTS fundamentals (user_id INTEGER NOT NULL, topic TEXT NOT NULL, level INTEGER NOT NULL,
    PRIMARY KEY (user_id, topic));
CREATE TABLE IF NOT EXISTS readiness_history (id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL, role_id TEXT NOT NULL,
    score INTEGER NOT NULL, day TEXT NOT NULL, UNIQUE(user_id, role_id, day));
CREATE TABLE IF NOT EXISTS settings (user_id INTEGER PRIMARY KEY, data TEXT NOT NULL);
"""


def now():
    return dt.datetime.now().isoformat(timespec="seconds")


class DB:
    def __init__(self, path=None):
        self.conn = sqlite3.connect(path or config.DB_PATH)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    def q(self, sql, args=(), one=False):
        cur = self.conn.execute(sql, args)
        rows = cur.fetchall()
        return (rows[0] if rows else None) if one else rows

    def x(self, sql, args=()):
        cur = self.conn.execute(sql, args)
        self.conn.commit()
        return cur.lastrowid

    # users
    def create_user(self, name, email, pw_hash):
        return self.x("INSERT INTO users (name, email, password_hash, created) VALUES (?,?,?,?)", (name, email, pw_hash, now()))

    def user_by_email(self, email):
        return self.q("SELECT * FROM users WHERE email = ?", (email,), one=True)

    def user(self, uid):
        return self.q("SELECT * FROM users WHERE id = ?", (uid,), one=True)

    # resumes and profiles
    def add_resume(self, uid, filename, analysis, quality):
        return self.x("INSERT INTO resumes (user_id, filename, uploaded, analysis, quality) VALUES (?,?,?,?,?)",
                      (uid, filename, now(), json.dumps(analysis), quality))

    def resumes(self, uid):
        return self.q("SELECT id, filename, uploaded, quality FROM resumes WHERE user_id = ? ORDER BY id DESC", (uid,))

    def resume(self, uid, rid):
        row = self.q("SELECT * FROM resumes WHERE user_id = ? AND id = ?", (uid, rid), one=True)
        return json.loads(row["analysis"]) if row else None

    def profile(self, uid):
        row = self.q("SELECT * FROM profiles WHERE user_id = ?", (uid,), one=True)
        if not row:
            return None
        data = json.loads(row["data"])
        data.update(roles=json.loads(row["roles"] or "[]"), targets=json.loads(row["targets"] or "[]"),
                    location=row["location"] or "", job_type=row["job_type"] or "Any",
                    version=row["version"], resume_id=row["resume_id"])
        return data

    def save_profile(self, uid, data, roles=None, targets=None, location=None, job_type=None, resume_id=None):
        old = self.profile(uid)
        keep = lambda new, key, default: new if new is not None else (old.get(key) if old else default)
        core = {k: v for k, v in data.items() if k not in ("roles", "targets", "location", "job_type", "version", "resume_id")}
        self.x("REPLACE INTO profiles (user_id, resume_id, data, roles, targets, location, job_type, version) VALUES (?,?,?,?,?,?,?,?)",
               (uid, keep(resume_id, "resume_id", None), json.dumps(core), json.dumps(keep(roles, "roles", [])),
                json.dumps(keep(targets, "targets", [])), keep(location, "location", ""), keep(job_type, "job_type", "Any"),
                dt.datetime.now().isoformat()))

    # job cache and results
    def cache_get(self, key, max_hours):
        row = self.q("SELECT * FROM job_cache WHERE key = ?", (key,), one=True)
        if row and dt.datetime.fromisoformat(row["fetched_at"]) > dt.datetime.now() - dt.timedelta(hours=max_hours):
            return json.loads(row["data"])
        return None

    def cache_set(self, key, data):
        self.x("REPLACE INTO job_cache (key, data, fetched_at) VALUES (?,?,?)", (key, json.dumps(data), now()))

    def results(self, uid):
        row = self.q("SELECT data FROM results WHERE user_id = ?", (uid,), one=True)
        return json.loads(row["data"]) if row else None

    def save_results(self, uid, data):
        self.x("REPLACE INTO results (user_id, data, created) VALUES (?,?,?)", (uid, json.dumps(data), now()))

    # saved jobs
    def save_job(self, uid, job, score):
        self.x("INSERT OR IGNORE INTO saved_jobs (user_id, job_id, job, score, status, saved_at, updated_at) VALUES (?,?,?,?,?,?,?)",
               (uid, job.get("id") or job.get("title"), json.dumps(job), score, "saved", now(), now()))

    def saved_jobs(self, uid):
        rows = self.q("SELECT * FROM saved_jobs WHERE user_id = ? ORDER BY updated_at DESC", (uid,))
        return [dict(r, job=json.loads(r["job"])) for r in rows]

    def saved_job(self, uid, sid):
        r = self.q("SELECT * FROM saved_jobs WHERE user_id = ? AND id = ?", (uid, sid), one=True)
        return dict(r, job=json.loads(r["job"])) if r else None

    def set_job_status(self, uid, sid, status):
        self.x("UPDATE saved_jobs SET status = ?, updated_at = ? WHERE user_id = ? AND id = ?", (status, now(), uid, sid))

    def delete_job(self, uid, sid):
        self.x("DELETE FROM saved_jobs WHERE user_id = ? AND id = ?", (uid, sid))

    # progress, fundamentals, readiness
    def done_keys(self, uid):
        return {r["key"]: r["done_at"] for r in self.q("SELECT key, done_at FROM progress WHERE user_id = ?", (uid,))}

    def toggle_progress(self, uid, key):
        if self.q("SELECT 1 FROM progress WHERE user_id = ? AND key = ?", (uid, key), one=True):
            self.x("DELETE FROM progress WHERE user_id = ? AND key = ?", (uid, key))
            return False
        self.x("INSERT INTO progress (user_id, key, done_at) VALUES (?,?,?)", (uid, key, now()))
        return True

    def fundamentals(self, uid):
        return {r["topic"]: r["level"] for r in self.q("SELECT topic, level FROM fundamentals WHERE user_id = ?", (uid,))}

    def set_fundamental(self, uid, topic, level):
        self.x("REPLACE INTO fundamentals (user_id, topic, level) VALUES (?,?,?)", (uid, topic, level))

    def record_readiness(self, uid, role_id, score):
        self.x("REPLACE INTO readiness_history (user_id, role_id, score, day) VALUES (?,?,?,?)",
               (uid, role_id, score, dt.date.today().isoformat()))

    def readiness_history(self, uid):
        return self.q("SELECT role_id, score, day FROM readiness_history WHERE user_id = ? ORDER BY day", (uid,))

    # settings
    def settings(self, uid):
        row = self.q("SELECT data FROM settings WHERE user_id = ?", (uid,), one=True)
        return json.loads(row["data"]) if row else {}

    def save_settings(self, uid, data):
        self.x("REPLACE INTO settings (user_id, data) VALUES (?,?)", (uid, json.dumps(data)))


def get_db():
    if "db" not in g:
        g.db = DB()
    return g.db


def close_db(_exc=None):
    db = g.pop("db", None)
    if db is not None:
        db.conn.close()
