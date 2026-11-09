"""Sign up, log in, log out. Passwords are stored as salted hashes (never plain text)."""
import functools
import re

from flask import Blueprint, flash, g, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

from web.db import get_db

bp = Blueprint("auth", __name__)


def login_required(view):
    @functools.wraps(view)
    def wrapped(*args, **kwargs):
        if g.get("user") is None:
            return redirect(url_for("auth.login", next=request.path))
        return view(*args, **kwargs)
    return wrapped


def is_safe_next(nxt):
    """Only allow redirects to pages on this site. Browsers treat '\\' like '/', so '/\\evil.com'
    would go to another website just like '//evil.com' (an "open redirect")."""
    return nxt.startswith("/") and not nxt.startswith("//") and "\\" not in nxt


def load_user():
    uid = session.get("user_id")
    g.user = get_db().user(uid) if uid else None


@bp.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        pw = request.form.get("password", "")
        error = None
        if not name or not email or not pw:
            error = "Fill in your name, email and a password."
        elif not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
            error = "Enter a valid email address."
        elif len(pw) < 6:
            error = "Use a password with at least 6 characters."
        elif get_db().user_by_email(email):
            error = "An account with this email already exists. Log in instead."
        if error:
            flash(error, "error")
        else:
            uid = get_db().create_user(name, email, generate_password_hash(pw))
            session.clear()
            session["user_id"] = uid
            flash(f"Welcome, {name}! Start by uploading your resume.", "ok")
            return redirect(url_for("main.profile"))
    return render_template("signup.html")


@bp.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user = get_db().user_by_email(request.form.get("email", "").strip().lower())
        if user is None or not check_password_hash(user["password_hash"], request.form.get("password", "")):
            flash("Email or password is incorrect.", "error")
        else:
            session.clear()
            session["user_id"] = user["id"]
            nxt = request.args.get("next", "")
            return redirect(nxt if is_safe_next(nxt) else url_for("main.profile"))
    return render_template("login.html")


@bp.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return redirect(url_for("main.home"))
