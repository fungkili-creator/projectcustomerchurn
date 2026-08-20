from pathlib import Path
import sqlite3
import psycopg2
import psycopg2.extras
from psycopg2 import pool
from flask import Flask, render_template, redirect, url_for, request, flash
from flask_wtf import CSRFProtect
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from model_service import ChurnModelService
import os
import secrets
from dotenv import load_dotenv 
## remarks this app.py better to run in python 3.12, make sure the workspace environment
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "customer_churn_data.csv"
DATABASE_URL = os.environ.get("DATABASE_URL")
pg_pool = None
if DATABASE_URL:
    # change from simle to thtead by Ki 8/20/2026 Point 1
    # amendment due to real environment handling
    pg_pool = pool.ThreadedConnectionPool(
        1, 10, DATABASE_URL, 
        cursor_factory=psycopg2.extras.RealDictCursor,
        keepalives=1,
        keepalives_idle=30,
        keepalives_interval=10,
        keepalives_count=5,)
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN")

app = Flask(__name__)

# SECRET_KEY must be set in the environment (e.g. Render's dashboard).
# No hardcoded fallback: a guessable secret key lets an attacker forge
# session cookies and flash-message signatures.
try:
    app.secret_key = os.environ["SECRET_KEY"]
except KeyError as exc:
    raise RuntimeError(
        "SECRET_KEY environment variable is not set. "
        "Set it in your deployment environment before starting the app."
    ) from exc

csrf = CSRFProtect(app)
@app.after_request
def set_security_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response


limiter = Limiter(get_remote_address, app=app, default_limits=["200 per hour"])

model_service = ChurnModelService(DATA_PATH)


def get_db():
    # Use Postgres in production (Render sets DATABASE_URL). Fall back to the
    # local SQLite file when it isn't set, so `python app.py` still works
    # without a Postgres instance on hand (matches README's local dev flow).
    if DATABASE_URL:
        conn = pg_pool.getconn()
        return conn, "postgres"
    conn = sqlite3.connect(BASE_DIR / "database.db")
    conn.row_factory = sqlite3.Row
    return conn, "sqlite"

def release_db(conn, db_type, discard=False):
    if db_type == "postgres":
        if discard:
            try:
                pg_pool.putconn(conn, close=True)
            except Exception:
                pass
        else:
            try:
                conn.rollback()
                pg_pool.putconn(conn)
            except Exception:
                pg_pool.putconn(conn, close=True)
    else:
        conn.close()

def init_db():
    conn, db_type = get_db()
# change from simle to thtead by Ki 8/20/2026 Point 2
    try: 
        if db_type == "postgres":
            with conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS predictions (
                            id SERIAL PRIMARY KEY,
                            age INTEGER NOT NULL,
                            gender TEXT NOT NULL,
                            tenure INTEGER NOT NULL,
                            monthly_charges REAL NOT NULL,
                            total_charges REAL NOT NULL,
                            contract_type TEXT NOT NULL,
                            internet_service TEXT NOT NULL,
                            tech_support TEXT NOT NULL,
                            prediction TEXT NOT NULL,
                            churn_probability REAL NOT NULL,
                            risk_level TEXT NOT NULL,
                            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                        )
                    """)
        else:
            with conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS predictions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        age INTEGER NOT NULL,
                        gender TEXT NOT NULL,
                        tenure INTEGER NOT NULL,
                        monthly_charges REAL NOT NULL,
                        total_charges REAL NOT NULL,
                        contract_type TEXT NOT NULL,
                        internet_service TEXT NOT NULL,
                        tech_support TEXT NOT NULL,
                        prediction TEXT NOT NULL,
                        churn_probability REAL NOT NULL,
                        risk_level TEXT NOT NULL,
                        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                    )
                """)

    finally:
        release_db(conn, db_type)



# Run once at import time, not inside `if __name__ == "__main__"`: gunicorn
# (see Procfile) imports this module directly and never executes that
# block, so relying on it there meant the predictions table was never
# created in production. A failure here is logged rather than raised so a
# transient DB hiccup at boot doesn't crash the whole worker.
try:
    init_db()
except Exception:
    app.logger.exception("Failed to initialize the predictions table at startup")

def post_fork(server, worker):
    global pg_pool
    import app
    if app.pg_pool:
        app.pg_pool.closeall()
    app.pg_pool = app.pool.ThreadedConnectionPool(1, 10, app.DATABASE_URL, cursor_factory=app.psycopg2.extras.RealDictCursor)

def validate_form(form):
    values = {
        "age": form.get("age", "").strip(),
        "gender": form.get("gender", "").strip(),
        "tenure": form.get("tenure", "").strip(),
        "monthly_charges": form.get("monthly_charges", "").strip(),
        "total_charges": form.get("total_charges", "").strip(),
        "contract_type": form.get("contract_type", "").strip(),
        "internet_service": form.get("internet_service", "").strip(),
        "tech_support": form.get("tech_support", "").strip(),
    }

    required = [k for k, v in values.items() if v == ""]
    if required:
        raise ValueError("Please complete every field before generating a prediction.")

    age = int(values["age"])
    tenure = int(values["tenure"])
    monthly = float(values["monthly_charges"])
    total = float(values["total_charges"])

    if not 18 <= age <= 120:
        raise ValueError("Age must be between 18 and 120.")
    if tenure < 0:
        raise ValueError("Tenure cannot be negative.")
    if monthly < 0 or total < 0:
        raise ValueError("Charges cannot be negative.")
    if values["gender"] not in {"Male", "Female"}:
        raise ValueError("Gender must match the categories used by the trained model: Male or Female.")
    if values["tech_support"] not in {"Yes", "No"}:
        raise ValueError("Tech Support must be Yes or No.")

    return values


@app.context_processor
def inject_model_status():
    return {
        "model_ready": model_service.ready,
        "model_error": model_service.error,
        "selected_model_name": "Tuned K-Nearest Neighbours",
    }


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/analysis")
def analysis():
    return redirect(url_for("analysis_overview"))


@app.route("/analysis/overview")
## change code by using postgres
def analysis_overview():

    conn, db_type = get_db()
    # change from simle to thtead by Ki 8/20/2026 Point 2
    try: 
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) AS total FROM predictions")
        total_predictions = cur.fetchone()["total"]

        cur.execute("SELECT COUNT(*) AS high FROM predictions WHERE risk_level = 'HIGH'")
        high_risk = cur.fetchone()["high"]

        cur.close()
    finally:
        release_db(conn, db_type)

    return render_template(
        "analysis/overview.html",
        total_predictions=total_predictions,
        high_risk=high_risk,
    )


@app.route("/analysis/predictions", methods=["GET", "POST"])
@limiter.limit("20 per minute", methods=["POST"])
def analysis_predictions():
    result = None
    form_data = {}
# change from simle to thtead by Ki 8/20/2026 Point 2 @second try

    if request.method == "POST":
        try:
            form_data = validate_form(request.form)
            prediction = model_service.predict(form_data)

            conn, db_type = get_db()
            discard_conn = False
            try: 
                cursor = conn.cursor()
                if db_type == "postgres":
                    cursor.execute("""
                        INSERT INTO predictions (
                            age, gender, tenure, monthly_charges, total_charges,
                            contract_type, internet_service, tech_support,
                            prediction, churn_probability, risk_level
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        RETURNING id;
                    """, (
                        int(form_data["age"]), form_data["gender"], int(form_data["tenure"]),
                        float(form_data["monthly_charges"]), float(form_data["total_charges"]),
                        form_data["contract_type"], form_data["internet_service"], form_data["tech_support"],
                        prediction["prediction"], prediction["probability"], prediction["risk_level"],
                    ))
                    prediction_id = cursor.fetchone()["id"]
                else:
                    cursor.execute("""
                        INSERT INTO predictions (
                            age, gender, tenure, monthly_charges, total_charges,
                            contract_type, internet_service, tech_support,
                            prediction, churn_probability, risk_level
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        int(form_data["age"]), form_data["gender"], int(form_data["tenure"]),
                        float(form_data["monthly_charges"]), float(form_data["total_charges"]),
                        form_data["contract_type"], form_data["internet_service"], form_data["tech_support"],
                        prediction["prediction"], prediction["probability"], prediction["risk_level"],
                    ))
                    prediction_id = cursor.lastrowid

                conn.commit() # may review on the position from Ki
                cursor.close()
            except psycopg2.OperationalError:
                discard_conn = True
                raise
            finally:
                release_db(conn, db_type, discard=discard_conn)
               
            return redirect(url_for("analysis_predictions", result_id=prediction_id))
        except ValueError as exc:
            # Validation errors are safe and useful to show the user directly.
            flash(str(exc), "error")
        except psycopg2.OperationalError:
            # Stale/dropped DB connection (e.g. SSL EOF). release_db() below
            # already discards this connection instead of pooling it, so a
            # retry from the user should get a fresh, working connection.
            app.logger.exception("Database connection error while generating a prediction")
            flash("Connection hiccup — please try submitting again.", "error")
        except Exception:
            # Anything unexpected (DB errors, model errors, etc.) is logged
            # server-side only, so internals are never exposed to the client.
            app.logger.exception("Unexpected error while generating a prediction")
            flash("Something went wrong processing your request. Please try again.", "error")
            

    result_id = request.args.get("result_id", type=int)
    if result_id:
        conn, db_type = get_db()
        # change from simle to thtead by Ki 8/20/2026 Point 2
        try:
            
            if db_type == "postgres":
                cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                cursor.execute("SELECT * FROM predictions WHERE id = %s", (result_id,))
        #        row = cursor.fetchone()
        #        if row:
        #            row = dict(row)
            else:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM predictions WHERE id = ?", (result_id,))
            # change due to a bit duplicated
            row = cursor.fetchone()
            if row:
                row = dict(row)

                cursor.close()
        
        finally:
            release_db(conn, db_type)

        if row:
            row["confidence_percent"] = round(float(row["churn_probability"]) * 100, 1)
            reasons = []
            if row["contract_type"] == "Month-to-Month": reasons.append("month-to-month contract")
            if float(row["monthly_charges"]) >= 80: reasons.append("higher monthly charges")
            if int(row["tenure"]) <= 12: reasons.append("short customer tenure")
            if row["tech_support"] == "No": reasons.append("no tech support")
            row["root_cause"] = ", ".join(reasons[:3]) if reasons else "combined customer profile signals"
            result = row
            form_data = {
                "age": row["age"], "gender": row["gender"], "tenure": row["tenure"],
                "monthly_charges": row["monthly_charges"], "total_charges": row["total_charges"],
                "contract_type": row["contract_type"], "internet_service": row["internet_service"],
                "tech_support": row["tech_support"],
            }

    return render_template(
        "analysis/predictions.html",
        result=result,
        form_data=form_data,
        model_params={"metric": "manhattan", "n_neighbors": 11, "p": 1, "weights": "distance"},
    )


@app.route("/analysis/predictions/reset", methods=["POST"])
def reset_prediction():
    return redirect(url_for("analysis_predictions"))


@app.route("/analysis/history/clear", methods=["POST"])
@limiter.limit("5 per minute")
def clear_prediction_history():
    # Destructive action — require a shared admin token so an anonymous
    # visitor can't wipe the prediction history for everyone.
    if not ADMIN_TOKEN or not secrets.compare_digest(request.form.get("admin_token",""), ADMIN_TOKEN):
        flash("Not authorized to clear prediction history.", "error")
        return redirect(url_for("analysis_predictions"))

    conn, db_type = get_db()
    # change from simle to thtead by Ki 8/20/2026 Point 2
    discard_conn = False
    try: 
        cursor = conn.cursor()
        cursor.execute("DELETE FROM predictions")
        conn.commit()
        cursor.close()
    except psycopg2.OperationalError:
        discard_conn = True
        app.logger.exception("Database connection error while clearing history")
        flash("Connection hiccup — please try again.", "error")
        return redirect(url_for("analysis_predictions"))
    finally:
        release_db(conn, db_type, discard=discard_conn)
    flash("Prediction history was cleared.", "success")
    return redirect(url_for("analysis_predictions"))


@app.route("/model")
def model():
    return render_template("model.html")


@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    # init_db() already ran at import time above. Debug mode is opt-in via
    # env var so it can never be accidentally left on in a deployed build.
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=5000, debug=debug_mode)
