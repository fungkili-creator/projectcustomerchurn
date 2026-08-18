# ChurnGuard AI — Offline Flask Integration

## Folder structure

```text
churnguard_offline/
├── app.py
├── model_service.py
├── customer_churn_data.csv   # add your original project dataset here
├── database.db               # created automatically
├── requirements.txt
├── static/
│   └── images/               # copy your existing image folder here
└── templates/
    ├── index.html
    ├── about.html
    ├── model.html
    └── analysis/
        ├── overview.html
        └── predictions.html
```

## Run locally

```bash
python -m venv myvenv
myvenv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000`.

## Prediction flow

1. User enters customer data in `/analysis/predictions`.
2. Flask validates all fields.
3. The deployment service applies the same encoding and `StandardScaler` logic used in the Phase 4 notebook.
4. Tuned KNN runs with Manhattan distance, `n_neighbors=11`, `p=1`, and distance weighting.
5. The prediction and churn probability are inserted into SQLite `database.db`.
6. The page reloads the stored record and renders the risk result and probability bar.
7. Reset clears the current form/result. Clear Database History deletes all stored prediction records.

## Important

The uploaded notebook did not contain the original `customer_churn_data.csv` or a serialized fitted `.joblib` model. The app therefore retrains the selected tuned KNN locally from `customer_churn_data.csv` when Flask starts. Add the same CSV used by the notebook beside `app.py` before running predictions.

The existing HTML uses Tailwind CDN and Google-hosted fonts. The Flask app/model/database run locally, but for a **fully no-internet visual frontend**, bundle Tailwind CSS/fonts into `static/` and replace those CDN references with local files.
"# projectcustomerchurn" 
