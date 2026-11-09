"""All website pages. Each route gathers data from the core modules and renders a template."""
import functools
import hashlib
import json

from flask import Blueprint, abort, flash, g, redirect, render_template, request, url_for

from core import advisor, fake_job, jobs_api, llm, market, reviewer
from core.matcher import analyze_job, explain, near_misses, rank_jobs
from core.parser import ResumeReadError, analyze_resume
from core.roles import ROLE_BY_ID, ROLES, predict_roles
from core.skillset import effective_skills
from nlp import models
from nlp.skills import category_of
from web.auth import login_required
from web.db import get_db

bp = Blueprint("main", __name__)
JOB_TYPES = ["Any", "Full-time", "Internship"]
STATUSES = ["saved", "applied", "interview", "offer", "rejected"]
MAX_JD_CHARS = 20000


# ---------------------------------------------------------------- helpers
def current_profile():
    db = get_db()
    p = db.profile(g.user["id"])
    if not p:
        return None
    extra = {s: "beginner" for s in p.get("confirmed", [])}
    for key in db.done_keys(g.user["id"]):
        if key.startswith("skill:"):
            extra[key[6:]] = "intermediate"     # completing a plan item counts as working knowledge
    p["extra_skills"] = extra
    return p


def needs_profile(view):
    @functools.wraps(view)
    def wrapped(*args, **kwargs):
        g.profile = current_profile()
        if g.profile is None:
            flash("Upload your resume first.", "info")
            return redirect(url_for("main.profile"))
        return view(*args, **kwargs)
    return login_required(wrapped)


def role_of_job(analysis):
    """Assign a posting to the closest role by skill overlap (used for market insights)."""
    skills = set(analysis["must"]) | set(analysis["nice"])
    best, best_s = None, 0.0
    for r in ROLES:
        rs = set(r["must"]) | set(r["nice"])
        s = len(skills & rs) / (len(skills | rs) or 1)
        if s > best_s:
            best, best_s = r["id"], s
    return best if best_s >= 0.12 else None


def get_results(refresh=False):
    db, p = get_db(), g.profile
    settings = db.settings(g.user["id"])
    targets = p.get("targets") or [r["id"] for r in predict_roles(p, 3)]
    fingerprint = hashlib.md5(json.dumps([p["version"], targets, p.get("location"), p.get("job_type"),
                                          bool(settings.get("jsearch_key")), bool(settings.get("adzuna_key")),
                                          fake_job.model_available()]).encode()).hexdigest()
    stored = db.results(g.user["id"])
    if stored and stored.get("fingerprint") == fingerprint and not refresh:
        return stored
    queries = [ROLE_BY_ID[t]["query"] for t in targets if t in ROLE_BY_ID]
    jt = p.get("job_type") if p.get("job_type") in ("Full-time", "Internship") else None
    jobs, notes = jobs_api.search(queries, p.get("location", ""), jt, settings, db, refresh)
    ranked = rank_jobs(p, jobs)
    for r in ranked:
        r["risk"] = fake_job.check(r["job"])
        r["risk_level"] = r["risk"]["level"]
        r["role_id"] = role_of_job(r["analysis"])
    data = {"fingerprint": fingerprint, "items": ranked, "notes": notes, "targets": targets,
            "live": not any("sample" in n.lower() for n in notes), "near": near_misses(ranked)}
    db.save_results(g.user["id"], data)
    return data


def demand_for(role_id, results):
    jobs = [r["job"] for r in results["items"] if r.get("role_id") == role_id and r["risk_level"] != "high"]
    scope = "role"
    if len(jobs) < 3:
        jobs = [r["job"] for r in results["items"] if r["risk_level"] != "high"]
        scope = "all"
    return market.skill_demand(jobs), jobs, scope


def pick_role(p, arg):
    targets = p.get("targets") or [r["id"] for r in predict_roles(p, 3)]
    return arg if arg in ROLE_BY_ID else targets[0], targets


# ---------------------------------------------------------------- pages
@bp.route("/")
def home():
    return render_template("home.html")


@bp.route("/profile")
@login_required
def profile():
    p = current_profile()
    if not p:
        return render_template("profile.html", p=None)
    skills, learning = effective_skills(p)
    grouped = {}
    for s, lvl in sorted(skills.items()):
        info = (p.get("skills") or {}).get(s, {"source": "confirmed" if s in p.get("confirmed", []) else "tracker"})
        grouped.setdefault(category_of(s), []).append({"name": s, "level": lvl, **info})
    roles = predict_roles(p, 6)
    return render_template("profile.html", p=p, grouped=grouped, learning=sorted(learning), roles=roles,
                           job_types=JOB_TYPES, all_roles=ROLES, resumes=get_db().resumes(g.user["id"]),
                           nlp_status=models.status())


@bp.route("/profile/upload", methods=["POST"])
@login_required
def upload():
    f = request.files.get("resume")
    if not f or not f.filename:
        flash("Choose a resume file to upload.", "error")
        return redirect(url_for("main.profile"))
    try:
        analysis = analyze_resume(f.read(), f.filename)
    except ResumeReadError as e:
        flash(str(e), "error")
        return redirect(url_for("main.profile"))
    except Exception as e:
        flash(f"Could not read that file ({e}). Try saving it again as PDF or DOCX.", "error")
        return redirect(url_for("main.profile"))
    db = get_db()
    old = db.profile(g.user["id"])
    analysis["confirmed"] = (old or {}).get("confirmed", [])
    quality = reviewer.review(analysis)["score"]
    rid = db.add_resume(g.user["id"], f.filename, analysis, quality)
    targets = (old or {}).get("targets") or [r["id"] for r in predict_roles({**analysis, "extra_skills": {}}, 3)]
    db.save_profile(g.user["id"], analysis, targets=targets, resume_id=rid)
    flash("Resume analysed. Check your profile, then explore matching jobs.", "ok")
    return redirect(url_for("main.profile"))


@bp.route("/profile/targets", methods=["POST"])
@login_required
def save_targets():
    targets = [t for t in request.form.getlist("targets") if t in ROLE_BY_ID][:3]
    if not targets:
        flash("Pick at least one target role.", "error")
        return redirect(url_for("main.profile"))
    p = get_db().profile(g.user["id"])
    jt = request.form.get("job_type", "Any")
    get_db().save_profile(g.user["id"], p, targets=targets, location=request.form.get("location", "").strip()[:60],
                          job_type=jt if jt in JOB_TYPES else "Any")
    flash("Targets saved. Jobs and preparation plans now follow these roles.", "ok")
    return redirect(url_for("main.jobs"))


@bp.route("/profile/confirm", methods=["POST"])
@login_required
def confirm_skill():
    p = get_db().profile(g.user["id"])
    term, action = request.form.get("term", ""), request.form.get("action", "confirm")
    if p and term in p.get("possible", []):
        p["possible"].remove(term)
        if action == "confirm":
            p.setdefault("confirmed", []).append(term.lower())
        get_db().save_profile(g.user["id"], p)
    return redirect(url_for("main.profile") + "#possible")


@bp.route("/jobs")
@needs_profile
def jobs():
    res = get_results(refresh=request.args.get("refresh") == "1")
    min_score = request.args.get("min", 0, type=int) or 0     # bad input like ?min=abc becomes 0, not a crash
    f = {"min": max(0, min(100, min_score)), "type": request.args.get("type", "Any"),
         "loc": request.args.get("loc", "").strip().lower(), "risky": request.args.get("risky", "hide")}
    items = []
    for i, r in enumerate(res["items"]):
        j = r["job"]
        if r["score"] < f["min"]:
            continue
        if f["type"] != "Any" and f["type"].lower() not in (j.get("job_type") or "").lower():
            continue
        if f["loc"] and f["loc"] not in (j.get("location") or "").lower():
            continue
        if f["risky"] == "hide" and r["risk_level"] == "high":
            continue
        items.append((i, r))
    hidden = sum(1 for r in res["items"] if r["risk_level"] == "high") if f["risky"] == "hide" else 0
    links = [(ROLE_BY_ID[t]["title"], jobs_api.search_links(ROLE_BY_ID[t]["title"], g.profile.get("location")))
             for t in res["targets"] if t in ROLE_BY_ID]
    saved = {s["job_id"] for s in get_db().saved_jobs(g.user["id"])}
    return render_template("jobs.html", items=items, res=res, f=f, job_types=JOB_TYPES, links=links,
                           hidden=hidden, saved=saved, p=g.profile)


@bp.route("/jobs/save", methods=["POST"])
@needs_profile
def save_job():
    res = get_results()
    i = request.form.get("i", type=int)
    if i is not None and 0 <= i < len(res["items"]):
        r = res["items"][i]
        get_db().save_job(g.user["id"], r["job"], r["score"])
        flash(f"Saved \u201c{r['job']['title']}\u201d to your tracker.", "ok")
    return redirect(request.referrer or url_for("main.jobs"))


def _render_match(job, r=None, source="results", index=None):
    p = g.profile
    if r is None:       # saved or pasted job: score it now
        r = rank_jobs(p, [job])[0]
        r["risk"] = fake_job.check(job)
    rows = explain(p, job)
    counts = {k: sum(1 for x in rows if x["status"] == k) for k in ("strong", "partial", "missing")}
    near = get_results().get("near", [])
    return render_template("match.html", r=r, job=job, rows=rows, counts=counts, near=near, source=source, index=index)


@bp.route("/match/<int:i>")
@needs_profile
def match(i):
    res = get_results()
    if not 0 <= i < len(res["items"]):
        abort(404)
    r = res["items"][i]
    return _render_match(r["job"], r, index=i)


@bp.route("/match/saved/<int:sid>")
@needs_profile
def match_saved(sid):
    s = get_db().saved_job(g.user["id"], sid)
    if not s:
        abort(404)
    return _render_match(s["job"], source="saved")


@bp.route("/match/custom", methods=["POST"])
@needs_profile
def match_custom():
    text = request.form.get("jd", "").strip()[:MAX_JD_CHARS]   # very long pastes would make the NLP slow
    if len(text.split()) < 15:
        flash("Paste the full job description (at least a few sentences).", "error")
        return redirect(url_for("main.jobs"))
    job = {"id": "custom", "title": request.form.get("title", "").strip() or "Pasted job description", "company": "",
           "location": "", "description": text, "source": "Pasted by you", "url": ""}
    return _render_match(job, source="custom")


@bp.route("/market")
@needs_profile
def market_page():
    res = get_results()
    role_id, targets = pick_role(g.profile, request.args.get("role"))
    demand, jobs, scope = demand_for(role_id, res)
    skills, _ = effective_skills(g.profile)
    top = [(s, round(v * 100), s in skills) for s, v in list(demand.items())[:15]]
    return render_template("market.html", role=ROLE_BY_ID[role_id], targets=targets, roles=ROLE_BY_ID, top=top,
                           emerging=market.emerging_skills(jobs), salary=market.salary_summary(jobs), n=len(jobs),
                           scope=scope, live=res["live"], lpa=market.lpa,
                           missing_hot=[t for t in top if not t[2] and t[1] >= 25][:5])


@bp.route("/feedback", methods=["GET"])
@needs_profile
def feedback(compare=None, ai=None, ai_error=None, ai_job=None):
    rv = reviewer.review(g.profile)
    res = get_results()
    top_jobs = [(i, r["job"]["title"], r["job"].get("company", "")) for i, r in enumerate(res["items"][:15])
                if r["risk_level"] != "high"]
    return render_template("feedback.html", rv=rv, resumes=get_db().resumes(g.user["id"]), compare=compare,
                           ai=ai, ai_error=ai_error, ai_job=ai_job, top_jobs=top_jobs,
                           has_key=bool(get_db().settings(g.user["id"]).get("gemini_key")),
                           pre=request.args.get("job", type=int))


@bp.route("/feedback/compare", methods=["POST"])
@needs_profile
def feedback_compare():
    db = get_db()
    old = db.resume(g.user["id"], request.form.get("old", type=int))
    new = db.resume(g.user["id"], request.form.get("new", type=int))
    if not old or not new:
        flash("Pick two resume versions to compare.", "error")
        return redirect(url_for("main.feedback"))
    cmp = reviewer.compare(old, new)
    jobs = [r["job"] for r in get_results()["items"] if r["risk_level"] != "high"][:15]
    avg = lambda prof: round(sum(r["score"] for r in rank_jobs({**prof, "extra_skills": {}}, jobs, rerank=False)) / max(len(jobs), 1), 1)
    cmp["old_match"], cmp["new_match"] = avg(old), avg(new)
    return feedback(compare=cmp)


@bp.route("/feedback/ai", methods=["POST"])
@needs_profile
def feedback_ai():
    settings = get_db().settings(g.user["id"])
    res = get_results()
    i = request.form.get("job", type=int)
    pasted = request.form.get("jd", "").strip()
    if pasted:
        job_text, label = pasted, "your pasted job description"
    elif i is not None and 0 <= i < len(res["items"]):
        j = res["items"][i]["job"]
        job_text, label = f"{j['title']} at {j.get('company', '')}\n{j['description']}", j["title"]
    else:
        flash("Pick a job or paste a job description.", "error")
        return redirect(url_for("main.feedback"))
    try:
        ai = llm.tailor(g.profile["text"], job_text, settings.get("gemini_key"), settings.get("gemini_model") or None)
        return feedback(ai=ai, ai_job=label)
    except llm.LLMError as e:
        return feedback(ai_error=str(e), ai_job=label)


@bp.route("/prepare")
@needs_profile
def prepare():
    db, p = get_db(), g.profile
    role_id, targets = pick_role(p, request.args.get("role"))
    hours = max(1, min(10, request.args.get("hours", 2, type=int)))
    weeks = max(1, min(16, request.args.get("weeks", 8, type=int)))
    res = get_results()
    demand, _, _ = demand_for(role_id, res)
    near_sk = {s for n in res.get("near", []) for s in n["skills"]}
    rv = reviewer.review(p)
    self_levels = db.fundamentals(g.user["id"])
    ready = advisor.readiness(p, role_id, demand, rv["score"], self_levels)
    db.record_readiness(g.user["id"], role_id, ready["score"])
    done = db.done_keys(g.user["id"])
    plan = advisor.study_plan(p, role_id, hours, weeks, demand, near_sk, frozenset(done))
    missing = {gp["skill"] for gp in plan["gaps"]}
    return render_template("prepare.html", role=ROLE_BY_ID[role_id], targets=targets, roles=ROLE_BY_ID, ready=ready,
                           plan=plan, hours=hours, weeks=weeks, interview=advisor.interview_topics(role_id),
                           projects=advisor.project_ideas(role_id, missing), levels=advisor.SELF_LEVELS)


@bp.route("/prepare/fundamentals", methods=["POST"])
@needs_profile
def save_fundamentals():
    for topic in advisor.FUNDAMENTALS:
        lvl = request.form.get(topic, type=int)
        if lvl is not None and 0 <= lvl <= 3:
            get_db().set_fundamental(g.user["id"], topic, lvl)
    flash("Self-assessment saved.", "ok")
    return redirect(request.referrer or url_for("main.prepare"))


@bp.route("/progress/toggle", methods=["POST"])
@needs_profile
def toggle_progress():
    key = request.form.get("key", "")[:200]
    if key:
        get_db().toggle_progress(g.user["id"], key)
        if key.startswith("skill:"):    # skills changed: bump profile version so matches refresh
            get_db().save_profile(g.user["id"], get_db().profile(g.user["id"]))
    return redirect((request.referrer or url_for("main.prepare")).split("#")[0] + "#plan")


@bp.route("/tracker")
@needs_profile
def tracker():
    db = get_db()
    hist = {}
    for row in db.readiness_history(g.user["id"]):
        hist.setdefault(row["role_id"], []).append((row["day"], row["score"]))
    charts = []
    for rid, pts in hist.items():
        if rid not in ROLE_BY_ID:
            continue
        n = len(pts)
        coords = [(40 + (i * 520 / max(n - 1, 1)), 170 - s * 1.5) for i, (_, s) in enumerate(pts)]
        charts.append({"role": ROLE_BY_ID[rid]["title"], "points": pts,
                       "path": " ".join(f"{x:.0f},{y:.0f}" for x, y in coords), "coords": coords})
    done = sorted(db.done_keys(g.user["id"]).items(), key=lambda x: x[1], reverse=True)
    return render_template("tracker.html", saved=db.saved_jobs(g.user["id"]), statuses=STATUSES, charts=charts, done=done)


@bp.route("/tracker/status", methods=["POST"])
@needs_profile
def tracker_status():
    st = request.form.get("status")
    if st in STATUSES:
        get_db().set_job_status(g.user["id"], request.form.get("id", type=int), st)
    return redirect(url_for("main.tracker"))


@bp.route("/tracker/delete", methods=["POST"])
@needs_profile
def tracker_delete():
    get_db().delete_job(g.user["id"], request.form.get("id", type=int))
    return redirect(url_for("main.tracker"))


@bp.route("/settings", methods=["GET", "POST"])
@login_required
def settings():
    db = get_db()
    s = db.settings(g.user["id"])
    if request.method == "POST":
        for k in ("jsearch_key", "adzuna_id", "adzuna_key", "gemini_key", "gemini_model"):
            v = request.form.get(k, "").strip()
            if v or request.form.get(f"clear_{k}"):
                s[k] = v
        s["jsearch_provider"] = request.form.get("jsearch_provider", "openwebninja")
        db.save_settings(g.user["id"], s)
        flash("Settings saved.", "ok")
        return redirect(url_for("main.settings"))
    masked = {k: (v[:4] + "\u2026" + v[-3:] if len(v) > 10 else "set") for k, v in s.items() if v and k.endswith(("key", "id"))}
    return render_template("settings.html", s=s, masked=masked, nlp_status=models.status(),
                           fake_model=fake_job.model_available())
