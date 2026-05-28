"""Streamlit app to predict Titanic ticket using a saved random forest pipeline."""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "rf_ticket_model.pkl"
LEGACY_MODEL_PATH = BASE_DIR / "ticket_model_small.pkl"
DATA_PATH = BASE_DIR / "Titanic-Dataset.csv"


def _read_dataset():
    if not DATA_PATH.exists():
        return None

    df = pd.read_csv(DATA_PATH)
    required_columns = {"Survived", "Sex", "Age", "Pclass", "Fare"}
    if not required_columns.issubset(df.columns):
        return None

    df = df[["Survived", "Sex", "Age", "Pclass", "Fare"]].copy()
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df["Fare"] = pd.to_numeric(df["Fare"], errors="coerce")
    df["Pclass"] = pd.to_numeric(df["Pclass"], errors="coerce")
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    df["Pclass"] = df["Pclass"].fillna(df["Pclass"].mode().iloc[0])
    df["Sex"] = df["Sex"].astype(str).str.lower().str.strip()
    return df.dropna(subset=["Survived", "Sex", "Age", "Pclass", "Fare"])


def _build_pipeline():
    numeric_features = ["Age", "Pclass", "Fare"]
    categorical_features = ["Sex"]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


@st.cache_resource
def load_artifacts():
    for path in (MODEL_PATH, LEGACY_MODEL_PATH):
        if path.exists():
            try:
                return joblib.load(path)
            except Exception as e:
                bad_path = path.with_suffix(path.suffix + ".broken")
                try:
                    path.rename(bad_path)
                except Exception:
                    pass
                print(f"Failed to load existing artifacts at {path}; moved to {bad_path}. Error: {e}")

    df = _read_dataset()
    if df is None or df.empty:
        return None

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["Survived"].astype(int))
    X = df[["Sex", "Age", "Pclass", "Fare"]]

    model = _build_pipeline()
    model.fit(X, y)

    artifacts = {"model": model, "label_encoder": label_encoder}
    try:
        joblib.dump(artifacts, MODEL_PATH)
    except Exception as e:
        print(f"Failed to save artifacts to {MODEL_PATH}: {e}")
    return artifacts


def build_input_df(sex, age, pclass, fare):
    return pd.DataFrame(
        {
            "Sex": [sex],
            "Age": [age],
            "Pclass": [pclass],
            "Fare": [fare],
        }
    )


st.set_page_config(page_title="Titanic Ticket Predictor", layout="centered")
st.title("Titanic Ticket Predictor")
st.write("Predict ticket based on Sex, Age, Pclass, and Fare.")

artifacts = load_artifacts()
if artifacts is None:
    st.error("Could not load or train a Titanic model from Titanic-Dataset.csv.")
    st.stop()

model = artifacts["model"]
label_encoder = artifacts["label_encoder"]

col1, col2 = st.columns(2)
with col1:
    sex = st.selectbox("Sex", ["female", "male"])
    pclass = st.selectbox("Pclass", [1, 2, 3])
with col2:
    age = st.number_input("Age", min_value=0.0, max_value=100.0, value=30.0, step=1.0)
    fare = st.number_input("Fare", min_value=0.0, value=32.0, step=1.0)

if st.button("Predict"):
    input_df = build_input_df(sex, age, pclass, fare)
    pred_label = model.predict(input_df)[0]
    survival_flag = label_encoder.inverse_transform([pred_label])[0]
    survival_text = "Survived" if int(survival_flag) == 1 else "Did not survive"
    st.success(f"Predicted Outcome: {survival_text}")
