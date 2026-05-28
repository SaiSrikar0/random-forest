"""Streamlit app to predict Titanic ticket using a saved random forest pipeline."""
import os

import joblib
import pandas as pd
import streamlit as st


MODEL_PATH = "rf_ticket_model.pkl"


@st.cache_resource
def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


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
    st.error("Model file not found. Run rf_ticket_classification.ipynb to train and save rf_ticket_model.pkl.")
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
    ticket = label_encoder.inverse_transform([pred_label])[0]
    st.success(f"Predicted Ticket: {ticket}")
