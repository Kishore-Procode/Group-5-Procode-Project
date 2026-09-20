import streamlit as st
import requests

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Customer Persona Segmentation",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --------------------------------------------------
# Custom CSS
# --------------------------------------------------
st.markdown("""
<style>

    /* Main background */
    .stApp {
        background: #f7f9fc;
    }

    /* Remove default top padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* Header */
    .hero {
        background: linear-gradient(135deg, #2563eb, #4f46e5);
        padding: 2.5rem 3rem;
        border-radius: 20px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(37, 99, 235, 0.15);
    }

    .hero h1 {
        font-size: 2.3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .hero p {
        font-size: 1.05rem;
        opacity: 0.9;
        margin-bottom: 0;
    }

    /* Section cards */
    .card {
        background: white;
        padding: 2rem;
        border-radius: 18px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 5px 18px rgba(0, 0, 0, 0.04);
        margin-bottom: 1.5rem;
    }

    .card-title {
        font-size: 1.25rem;
        font-weight: 650;
        color: #111827;
        margin-bottom: 0.4rem;
    }

    .card-description {
        color: #6b7280;
        font-size: 0.92rem;
        margin-bottom: 1.5rem;
    }

    /* Input labels */
    label {
        font-weight: 600 !important;
        color: #374151 !important;
    }

    /* Input boxes */
    div[data-baseweb="input"] {
        border-radius: 10px;
    }

    /* Button */
    .stButton > button {
        width: 100%;
        height: 48px;
        border-radius: 10px;
        border: none;
        background: #2563eb;
        color: white;
        font-size: 1rem;
        font-weight: 600;
        transition: 0.2s ease;
    }

    .stButton > button:hover {
        background: #1d4ed8;
        border: none;
        color: white;
    }

    /* Result box */
    .result-box {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        border-radius: 16px;
        padding: 1.5rem;
        margin-top: 1rem;
    }

    .result-title {
        color: #1e40af;
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .result-text {
        color: #1f2937;
        font-size: 1rem;
    }

    /* Info cards */
    .info-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 1.3rem;
        height: 100%;
    }

    .info-icon {
        font-size: 1.5rem;
        margin-bottom: 0.5rem;
    }

    .info-title {
        font-weight: 650;
        color: #111827;
        margin-bottom: 0.3rem;
    }

    .info-text {
        color: #6b7280;
        font-size: 0.88rem;
        line-height: 1.5;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #9ca3af;
        font-size: 0.82rem;
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid #e5e7eb;
    }

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# Hero section
# --------------------------------------------------
st.markdown("""
<div class="hero">
    <h1>Customer Persona Segmentation</h1>
    <p>
        Analyze customer behavior and identify their persona
        using income and spending patterns.
    </p>
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# Main input section
# --------------------------------------------------
st.markdown("""
<div class="card">
    <div class="card-title">Customer Information</div>
    <div class="card-description">
        Enter the customer's financial and spending details
        to generate a persona prediction.
    </div>
</div>
""", unsafe_allow_html=True)


col1, col2 = st.columns(2)

with col1:
    annual_income = st.number_input(
        "Annual Income ($K)",
        min_value=0.0,
        value=50.0,
        step=1.0,
        help="Enter the customer's annual income in thousands of dollars."
    )

with col2:
    spending_score = st.number_input(
        "Spending Score",
        min_value=0.0,
        max_value=100.0,
        value=50.0,
        step=1.0,
        help="Enter a spending score between 0 and 100."
    )


st.write("")

# --------------------------------------------------
# Prediction button
# --------------------------------------------------
predict = st.button("Predict Customer Persona")


# --------------------------------------------------
# API request
# --------------------------------------------------
if predict:

    data = {
        "annual_income_k": annual_income,
        "spending_score": spending_score
    }

    with st.spinner("Analyzing customer profile..."):

        try:
            response = requests.post(
                "http://127.0.0.1:8000/predict",
                json=data,
                timeout=10
            )

            # Check HTTP status
            response.raise_for_status()

            result = response.json()

            # --------------------------------------------------
            # Result
            # --------------------------------------------------
            st.markdown("""
            <div class="result-box">
                <div class="result-title">Prediction Result</div>
                <div class="result-text">
                    Customer persona successfully identified.
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.json(result)

        except requests.exceptions.ConnectionError:

            st.error(
                "Unable to connect to the prediction server. "
                "Please make sure the FastAPI backend is running."
            )

        except requests.exceptions.Timeout:

            st.error(
                "The prediction server took too long to respond. "
                "Please try again."
            )

        except requests.exceptions.HTTPError as e:

            st.error(
                f"The prediction server returned an error: {e}"
            )

        except ValueError:

            st.error(
                "The server returned an invalid response."
            )

        except Exception as e:

            st.error(
                f"Something went wrong: {e}"
            )


# --------------------------------------------------
# Information section
# --------------------------------------------------
st.write("")
st.write("")

info1, info2, info3 = st.columns(3)

with info1:
    st.markdown("""
    <div class="info-card">
        <div class="info-icon">💰</div>
        <div class="info-title">Annual Income</div>
        <div class="info-text">
            The customer's annual income, measured in thousands
            of dollars.
        </div>
    </div>
    """, unsafe_allow_html=True)

with info2:
    st.markdown("""
    <div class="info-card">
        <div class="info-icon">🛍️</div>
        <div class="info-title">Spending Score</div>
        <div class="info-text">
            A score from 0 to 100 representing the customer's
            spending behavior.
        </div>
    </div>
    """, unsafe_allow_html=True)

with info3:
    st.markdown("""
    <div class="info-card">
        <div class="info-icon">🎯</div>
        <div class="info-title">Persona Prediction</div>
        <div class="info-text">
            The model analyzes the provided details and returns
            the predicted customer persona.
        </div>
    </div>
    """, unsafe_allow_html=True)


# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("""
<div class="footer">
    Customer Persona Segmentation · Machine Learning Application
</div>
""", unsafe_allow_html=True)