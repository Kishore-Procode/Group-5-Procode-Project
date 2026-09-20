import requests
import streamlit as st

st.title("Loan Approval Predictor")
st.write("Enter customer details to predict whether the loan will be approved.")

with st.form("loan_form"):
    income = st.number_input("Income", min_value=0.0, value=50000.0, step=1000.0)
    credit_score = st.number_input("Credit Score", min_value=0, max_value=850, value=700, step=1)
    loan_amount = st.number_input("Loan Amount", min_value=0.0, value=25000.0, step=1000.0)
    employment_years = st.number_input("Employment Years", min_value=0, value=5, step=1)
    submitted = st.form_submit_button("Predict")

if submitted:
    payload = {
        "income": income,
        "credit_score": int(credit_score),
        "loan_amount": loan_amount,
        "employment_years": int(employment_years),
    }

    try:
        response = requests.post("http://localhost:8000/predict", json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()

        if result["prediction"] == 1:
            st.success(f"Approved — confidence: {result['approval_probability']:.2%}")
        else:
            st.error(f"Rejected — confidence: {result['approval_probability']:.2%}")
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
