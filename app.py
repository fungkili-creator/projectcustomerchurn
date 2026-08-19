from pathlib import Path
import sqlite3
import psycopg2
import psycopg2.extras
from flask import Flask, render_template, redirect, url_for, request, flash
from flask_wtf import CSRFProtect
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from model_service import ChurnModelService
import os

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "database.db"
DATA_PATH = BASE_DIR / "customer_churn_data.csv"
DATABASE_URL = os.environ.get("DATABASE_URL")
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
limiter = Limiter(get_remote_address, app=app, default_limits=["200 per hour"])

model_service = ChurnModelService(DATA_PATH)


def get_db():
    # add if case for the newly create of DATABASE_URL environment for postgresql
    if DATABASE_URL:
        url = DATABASE_URL.replace("postgres://", "postgresql://", 1)
        conn = psycopg2.connect(url)
        return conn, "postgres"
    else:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn, "sqlite"


def init_db():
    conn, db_type = get_db()
    # with get_db() as conn: replce this line
##    with conn:
    cursor = conn.cursor()
    if db_type == "postgres":
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id SERIAL PRIMARY KEY,
                age INTEGER NOT NULL,
                gender VARCHAR(20) NOT NULL,
                tenure INTEGER NOT NULL,
                monthly_charges NUMERIC(10,2) NOT NULL,
                total_charges NUMERIC(10,2) NOT NULL,
                contract_type VARCHAR(50) NOT NULL,
                internet_service VARCHAR(50) NOT NULL,
                tech_support VARCHAR(20) NOT NULL,
                prediction VARCHAR(10) NOT NULL,
                churn_probability NUMERIC(5,4) NOT NULL,
                risk_level VARCHAR(20) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
    else:
        cursor.execute("""
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
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
        """)
    conn.commit()
    cursor.close()
    conn.close()


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
##    with get_db() as conn:
##        total_predictions = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
##        high_risk = conn.execute("SELECT COUNT(*) FROM predictions WHERE risk_level = 'HIGH'").fetchone()[0]
##    return render_template(
##        "analysis/overview.html",
##        total_predictions=total_predictions,
##        high_risk=high_risk,
    conn, db_type = get_db()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM predictions")
    total_predictions = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM predictions WHERE risk_level = 'HIGH'")
    high_risk = cursor.fetchone()[0]
    
    cursor.close()
    conn.close()
    
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

    if request.method == "POST":
        try:
            form_data = validate_form(request.form)
            prediction = model_service.predict(form_data)

            conn, db_type = get_db()
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
                prediction_id = cursor.fetchone()[0]
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

            conn.commit()
            cursor.close()
            conn.close()

            return redirect(url_for("analysis_predictions", result_id=prediction_id))
        except ValueError as exc:
            # Validation errors are safe and useful to show the user directly.
            flash(str(exc), "error")
        except Exception:
            # Anything unexpected (DB errors, model errors, etc.) is logged
            # server-side only, so internals are never exposed to the client.
            app.logger.exception("Unexpected error while generating a prediction")
            flash("Something went wrong processing your request. Please try again.", "error")

    result_id = request.args.get("result_id", type=int)
    if result_id:
        conn, db_type = get_db()
        if db_type == "postgres":
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cursor.execute("SELECT * FROM predictions WHERE id = %s", (result_id,))
            row = cursor.fetchone()
            if row:
                row = dict(row)
        else:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM predictions WHERE id = ?", (result_id,))
            row = cursor.fetchone()
            if row:
                row = dict(row)

        cursor.close()
        conn.close()

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
    if not ADMIN_TOKEN or request.form.get("admin_token") != ADMIN_TOKEN:
        flash("Not authorized to clear prediction history.", "error")
        return redirect(url_for("analysis_predictions"))

    conn, db_type = get_db()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM predictions")
    conn.commit()
    cursor.close()
    conn.close()

    flash("Prediction history was cleared.", "success")
    return redirect(url_for("analysis_predictions"))


@app.route("/model")
def model():
    return render_template("model.html")


@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    init_db()
    app.run(host="127.0.0.1", port=5000, debug=True)
