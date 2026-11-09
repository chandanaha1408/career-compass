"""
Career Compass - AI-based resume screening and job matching for students.
Run:  python app.py   then open http://localhost:5000
"""
import logging
import secrets

from flask import Flask, abort, render_template, request, session
from markupsafe import Markup

import config
from core.market import lpa
from web import auth, routes
from web.db import close_db


def create_app():
    app = Flask(__name__, template_folder="web/templates", static_folder="web/static")
    app.config.update(SECRET_KEY=config.SECRET_KEY, MAX_CONTENT_LENGTH=config.MAX_UPLOAD_MB * 1024 * 1024,
                      SESSION_COOKIE_SAMESITE="Lax", SESSION_COOKIE_HTTPONLY=True)
    app.register_blueprint(auth.bp)
    app.register_blueprint(routes.bp)
    app.before_request(auth.load_user)
    app.teardown_appcontext(close_db)

    def csrf_token():
        if "csrf" not in session:
            session["csrf"] = secrets.token_hex(16)
        return session["csrf"]

    @app.before_request
    def check_csrf():
        """Every form carries a secret token so other websites can't submit forms as you."""
        if request.method == "POST" and request.form.get("csrf_token") != session.get("csrf"):
            abort(400, "Form expired. Go back, refresh the page and try again.")

    def csrf_field():
        return Markup(f'<input type="hidden" name="csrf_token" value="{csrf_token()}">')

    app.jinja_env.globals.update(csrf_token=csrf_token, csrf_field=csrf_field, zip=zip)
    app.jinja_env.filters["lpa"] = lpa

    @app.errorhandler(404)
    def not_found(e):
        return render_template("error.html", code=404, message="That page doesn't exist."), 404

    @app.errorhandler(400)
    def bad_request(e):
        return render_template("error.html", code=400, message=getattr(e, "description", "Bad request")), 400

    @app.errorhandler(413)
    def too_large(e):
        return render_template("error.html", code=413, message=f"Files must be under {config.MAX_UPLOAD_MB} MB."), 413

    return app


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    print("Loading NLP models (first run downloads them, this can take a minute)...")
    from nlp import models
    for name, ok in models.status().items():
        print(f"  {'OK ' if ok else '-- '} {name}{'' if ok else ' (fallback in use)'}")
    print("\nCareer Compass is running at http://localhost:5000\n")
    create_app().run(host="127.0.0.1", port=5000, debug=False)
