from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import joblib

RANDOM_STATE = 42
MODEL_CACHE = Path(__file__).resolve().parent / "model_cache.joblib"

MODEL_PARAMS = {
    "metric": "manhattan",
    "n_neighbors": 11,
    "p": 1,
    "weights": "distance",
}

class ChurnModelService:
    def __init__(self, data_path: str | Path):
        self.data_path = Path(data_path)
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.ready = False
        self.error = None
        self.train_rows = 0
        self.test_rows = 0
        self._load_and_train()

    @staticmethod
    def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        required_columns = [
            "CustomerID", "Age", "Gender", "Tenure", "MonthlyCharges",
            "TotalCharges", "ContractType", "InternetService",
            "TechSupport", "Churn"
        ]
        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Dataset is missing required columns: {missing}")

        model_df = df.copy()
        object_columns = model_df.select_dtypes(include=["object", "string"]).columns
        for column in object_columns:
            model_df[column] = model_df[column].replace(r"^\s*$", np.nan, regex=True)

        model_df["InternetService"] = model_df["InternetService"].fillna("No Internet Service")

        numeric_columns = model_df.select_dtypes(include=["number"]).columns
        for column in numeric_columns:
            model_df[column] = model_df[column].fillna(model_df[column].median())

        categorical_columns = model_df.select_dtypes(include=["object", "string"]).columns
        for column in categorical_columns:
            if model_df[column].isna().any():
                model_df[column] = model_df[column].fillna(model_df[column].mode().iloc[0])

        return model_df.drop_duplicates().reset_index(drop=True)

    @staticmethod
    def _encode(dataframe: pd.DataFrame) -> pd.DataFrame:
        encoded = dataframe.copy()
        encoded["Gender"] = encoded["Gender"].map({"Male": 1, "Female": 0})
        encoded["TechSupport"] = encoded["TechSupport"].map({"Yes": 1, "No": 0})
        if "Churn" in encoded.columns:
            encoded["Churn"] = encoded["Churn"].map({"Yes": 1, "No": 0})

        encoded = pd.get_dummies(
            encoded,
            columns=["ContractType", "InternetService"],
            drop_first=False,
        )
        if "CustomerID" in encoded.columns:
            encoded = encoded.drop(columns=["CustomerID"])
        return encoded

    def _data_fingerprint(self) -> tuple:
        # Cheap way to detect "the CSV changed" without hashing file contents:
        # size + mtime is enough to invalidate the cache when the dataset
        # is replaced, while still being fast to check on every worker boot.
        stat = self.data_path.stat()
        return (stat.st_size, stat.st_mtime)

    def _load_and_train(self):
        try:
            if not self.data_path.exists():
                raise FileNotFoundError(
                    f"{self.data_path.name} was not found. Place it beside app.py."
                )

            fingerprint = self._data_fingerprint()
            if MODEL_CACHE.exists():
                try:
                    cached = joblib.load(MODEL_CACHE)
                    if cached.get("fingerprint") == fingerprint and cached.get("params") == MODEL_PARAMS:
                        self.model = cached["model"]
                        self.scaler = cached["scaler"]
                        self.feature_columns = cached["feature_columns"]
                        self.train_rows = cached["train_rows"]
                        self.test_rows = cached["test_rows"]
                        self.ready = True
                        self.error = None
                        return
                except Exception:
                    # Corrupt or incompatible cache file — fall through and retrain.
                    pass

            df = pd.read_csv(self.data_path)
            model_df = self._clean_dataframe(df)
            train_df, test_df = train_test_split(
                model_df,
                test_size=0.20,
                random_state=RANDOM_STATE,
                stratify=model_df["Churn"],
            )

            train_encoded = self._encode(train_df)
            test_encoded = self._encode(test_df)

            X_train = train_encoded.drop(columns=["Churn"])
            y_train = train_encoded["Churn"]
            X_test = test_encoded.drop(columns=["Churn"]).reindex(columns=X_train.columns, fill_value=0)

            self.feature_columns = list(X_train.columns)
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            self.scaler.transform(X_test)  # validates deployment feature alignment

            self.model = KNeighborsClassifier(**MODEL_PARAMS)
            self.model.fit(X_train_scaled, y_train)

            self.train_rows = len(train_df)
            self.test_rows = len(test_df)
            self.ready = True
            self.error = None

            try:
                joblib.dump({
                    "model": self.model,
                    "scaler": self.scaler,
                    "feature_columns": self.feature_columns,
                    "train_rows": self.train_rows,
                    "test_rows": self.test_rows,
                    "fingerprint": fingerprint,
                    "params": MODEL_PARAMS,
                }, MODEL_CACHE)
            except Exception:
                # Caching is a nice-to-have; never let a failed write to
                # disk (e.g. read-only filesystem) break the app.
                pass
        except Exception as exc:
            self.ready = False
            self.error = str(exc)

    def predict(self, values: dict) -> dict:
        if not self.ready:
            raise RuntimeError(self.error or "Model is not ready.")

        row = pd.DataFrame([{
            "CustomerID": "WEB-INPUT",
            "Age": float(values["age"]),
            "Gender": values["gender"],
            "Tenure": float(values["tenure"]),
            "MonthlyCharges": float(values["monthly_charges"]),
            "TotalCharges": float(values["total_charges"]),
            "ContractType": values["contract_type"],
            "InternetService": values["internet_service"],
            "TechSupport": values["tech_support"],
        }])

        encoded = self._encode(row).reindex(columns=self.feature_columns, fill_value=0)
        scaled = self.scaler.transform(encoded)
        prediction = int(self.model.predict(scaled)[0])
        probabilities = self.model.predict_proba(scaled)[0]
        class_index = list(self.model.classes_).index(1)
        churn_probability = float(probabilities[class_index])

        if churn_probability >= 0.70:
            risk_level = "HIGH"
        elif churn_probability >= 0.40:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        reasons = []
        if values["contract_type"] == "Month-to-Month":
            reasons.append("month-to-month contract")
        if float(values["monthly_charges"]) >= 80:
            reasons.append("higher monthly charges")
        if float(values["tenure"]) <= 12:
            reasons.append("short customer tenure")
        if values["tech_support"] == "No":
            reasons.append("no tech support")
        root_cause = ", ".join(reasons[:3]) if reasons else "combined customer profile signals"

        return {
            "prediction": "Yes" if prediction == 1 else "No",
            "probability": churn_probability,
            "confidence_percent": round(churn_probability * 100, 1),
            "risk_level": risk_level,
            "root_cause": root_cause,
            "model_name": "Tuned K-Nearest Neighbours",
            "model_params": MODEL_PARAMS,
        }
