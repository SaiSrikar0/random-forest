# Random Forest Streamlit Apps

This repository contains two Streamlit apps:

- `random_forest_regressor/App.py` — Laptop price predictor. Trains from `data.csv` and saves `rf_laptop_price_model.pkl` if missing.
- `random_forest_classifier/App.py` — Titanic survival predictor. Trains from `Titanic-Dataset.csv` and saves `rf_ticket_model.pkl` if missing.

Deployment notes:

1. Streamlit Cloud will install dependencies from the repository root `requirements.txt`.
2. Configure each app to point to the correct entry file in Streamlit Cloud (choose the folder and `App.py`).
3. On first run, the apps will train from the included CSVs if a model pickle is not present — this avoids deployment-time failures caused by missing artifact files.

Local run:

```bash
# From repo root
pip install -r requirements.txt

# Run laptop app
cd random_forest_regressor
streamlit run App.py

# Run classifier app
cd ../random_forest_classifier
streamlit run App.py
```

If you deploy with Streamlit Cloud, push the branch that the app is configured to track and hit "Redeploy" in the app dashboard.
