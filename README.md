# ChurnGuard AI — Flask Integration

**Live app:** https://churnguard-ai-w4jg.onrender.com/

## Folder structure

```text
churnguard/
├── app.py
├── model_service.py
├── customer_churn_data.csv   # add your original project dataset here
├── model_cache.joblib         # generated locally; gitignored
├── database.db                 # local SQLite fallback only; gitignored
├── requirements.txt
├── Procfile
├── .env.example                # copy to .env for local dev, fill in real values
├── .gitignore
├── static/
│   └── images/
└── templates/
    ├── index.html
    ├── about.html
    ├── model.html
    └── analysis/
        ├── overview.html
        └── predictions.html
```

## Environment variables

Copy `.env.example` to `.env` and fill in real values. Never commit `.env`.

| Variable | Required | Notes |
|---|---|---|
| `SECRET_KEY` | Always | Flask session/flash signing key. App refuses to start without it. Use a long random value — generate with `python -c "import secrets; print(secrets.token_hex(32))"`. |
| `DATABASE_URL` | Production only | Postgres connection string (Render provides this). If unset, the app falls back to a local SQLite file (`database.db`) so you can develop without a Postgres instance. |
| `ADMIN_TOKEN` | If using `/analysis/history/clear` | Required to authorize clearing prediction history. Generate independently — don't derive it from the database host name or any part of `DATABASE_URL`. |

## Run locally

```bash
python -m venv myvenv
myvenv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env    # then fill in SECRET_KEY at minimum
python app.py
```

Open `http://127.0.0.1:5000`. Without `DATABASE_URL` set, this uses local SQLite (`database.db`) automatically.

## Production deployment (Render)

The app runs under gunicorn in production (see `Procfile`):

```
web: gunicorn app:app --workers 3 --threads 2 --timeout 60
```

Set `SECRET_KEY`, `DATABASE_URL`, and `ADMIN_TOKEN` as environment variables in Render's dashboard — the app reads them via `python-dotenv`/`os.environ`, never from source. `DATABASE_URL` should point at the managed Postgres instance provisioned on Render; the app auto-creates the `predictions` table on startup either way (SQLite locally, Postgres in production). See `Steps_of_Deployment.docx` for the full step-by-step walkthrough.

## Prediction flow

1. User enters customer data in `/analysis/predictions`.
2. Flask validates all fields (`validate_form`).
3. `model_service.py` applies the same encoding and `StandardScaler` logic used in the Phase 4 notebook.
4. Tuned KNN runs with Manhattan distance, `n_neighbors=11`, `p=1`, and distance weighting.
5. The prediction and churn probability are inserted into the active database (Postgres in production, SQLite locally).
6. The page reloads the stored record and renders the risk result and probability bar.
7. Reset clears the current form/result. Clear Database History (admin-token protected) deletes all stored prediction records.

## Important

The uploaded notebook did not contain the original `customer_churn_data.csv` or a serialized fitted `.joblib` model. The app retrains the tuned KNN from `customer_churn_data.csv` on startup if no valid cache is found; make sure that CSV is present beside `app.py`. `model_cache.joblib` is gitignored since its cache key (file size + mtime) is invalidated by most fresh checkouts anyway, so committing it doesn't save a retrain in practice.

The existing HTML uses Tailwind CDN and Google-hosted fonts, which requires the deployed environment to have outbound internet access (Render's default is fine). For a fully offline frontend, bundle Tailwind CSS/fonts into `static/` and replace those CDN references with local files.
