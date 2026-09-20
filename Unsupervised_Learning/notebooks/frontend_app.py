import requests
import streamlit as st

st.title("Customer Persona Segment Predictor")
st.write("Enter customer details to predict their segment cluster.")

with st.form("segmentation_form"):
    annual_income_k = st.number_input("Annual Income ($k)", min_value=0.0, value=50.0, step=1.0)
    spending_score = st.number_input("Spending Score (1-100)", min_value=1, max_value=100, value=50, step=1)
    submitted = st.form_submit_button("Predict")

if submitted:
    payload = {
        "annual_income_k": annual_income_k,
        "spending_score": int(spending_score),
    }

    try:
        response = requests.post("http://localhost:8000/predict", json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()

        st.success(f"Cluster {result['cluster']}: {result['cluster_label']}")
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
