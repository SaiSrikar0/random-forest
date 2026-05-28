"""Streamlit app to predict laptop price using a saved random forest pipeline."""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "rf_laptop_price_model.pkl"
DATA_PATH = BASE_DIR / "data.csv"


def _read_dataset():
    if not DATA_PATH.exists():
        return None

    df = pd.read_csv(DATA_PATH)
    required_columns = {"brand", "CPU", "Ram", "ROM", "price"}
    if not required_columns.issubset(df.columns):
        return None

    df = df[["brand", "CPU", "Ram", "ROM", "price"]].copy()
    df["Ram_GB"] = df["Ram"].astype(str).str.extract(r"(\d+)").astype(float)
    df["ROM_GB"] = df["ROM"].astype(str).str.extract(r"(\d+)").astype(float)
    return df.drop(columns=["Ram", "ROM"])


def _build_pipeline():
    numeric_features = ["Ram_GB", "ROM_GB"]
    categorical_features = ["brand", "CPU"]

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
                RandomForestRegressor(
                    n_estimators=200,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


@st.cache_resource
def load_model():
    if MODEL_PATH.exists():
        try:
            return joblib.load(MODEL_PATH)
        except Exception as e:
            # Rename broken model so we don't keep failing to unpickle it
            bad_path = MODEL_PATH.with_suffix(MODEL_PATH.suffix + ".broken")
            try:
                MODEL_PATH.rename(bad_path)
            except Exception:
                pass
            print(f"Failed to load existing model; moved to {bad_path}. Error: {e}")

    df = _read_dataset()
    if df is None or df.empty:
        return None

    model = _build_pipeline()
    model.fit(df[["brand", "CPU", "Ram_GB", "ROM_GB"]], df["price"])
    try:
        joblib.dump(model, MODEL_PATH)
    except Exception as e:
        print(f"Failed to save trained model to {MODEL_PATH}: {e}")
    return model


def load_options():
    df = _read_dataset()
    if df is None or df.empty:
        return ["HP", "Dell", "Lenovo"], ["Intel Core i5", "AMD Ryzen 5"]

    brands = sorted(df["brand"].dropna().astype(str).unique().tolist())
    cpus = sorted(df["CPU"].dropna().astype(str).unique().tolist())
    return brands, cpus


def build_input_df(brand, cpu, ram_gb, rom_gb):
    return pd.DataFrame(
        {
            "brand": [brand],
            "CPU": [cpu],
            "Ram_GB": [ram_gb],
            "ROM_GB": [rom_gb],
        }
    )


st.set_page_config(page_title="Laptop Price Predictor", layout="centered")
st.title("Laptop Price Predictor")
st.write("Predict laptop price using brand, CPU, RAM, and ROM.")

model = load_model()
if model is None:
    st.error("Could not load or train a laptop pricing model from data.csv.")
    st.stop()

brands, cpus = load_options()

col1, col2 = st.columns(2)
with col1:
    brand = st.selectbox("Brand", brands)
    ram_gb = st.number_input("RAM (GB)", min_value=2, max_value=128, value=8, step=1)
with col2:
    cpu = st.selectbox("CPU", cpus)
    rom_gb = st.number_input("ROM (GB)", min_value=64, max_value=4096, value=512, step=64)

if st.button("Predict"):
    input_df = build_input_df(brand, cpu, ram_gb, rom_gb)
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Price: {prediction:.0f}")
