"""Streamlit app to predict laptop price using a saved random forest pipeline."""
import os

import joblib
import pandas as pd
import streamlit as st


MODEL_PATH = "rf_laptop_price_model.pkl"
DATA_PATH = "data.csv"


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


def load_options():
    if not os.path.exists(DATA_PATH):
        return ["HP", "Dell", "Lenovo"], ["Intel Core i5", "AMD Ryzen 5"]
    df = pd.read_csv(DATA_PATH)
    brands = sorted(df["brand"].dropna().unique().tolist())
    cpus = sorted(df["CPU"].dropna().unique().tolist())
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
    st.error(
        "Model file not found. Run rf_laptop_regression.ipynb to train and save rf_laptop_price_model.pkl."
    )
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
