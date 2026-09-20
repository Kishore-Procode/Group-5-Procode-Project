from pathlib import Path

import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Request

FEATURE_COLUMNS = ["annual_income_k", "spending_score"]

CLUSTER_DETAILS = {
    0: {
        "cluster_name": "High-Value Customers",
        "cluster_label": "High Income, High Spending (Target)",
        "description": "Customers with high annual income and high spending scores. Represents strong purchasing power and high engagement.",
        "recommendation": "Target with premium product launches, loyalty rewards, VIP events, and exclusive personalized offers.",
    },
    1: {
        "cluster_name": "High-Engagement Customers",
        "cluster_label": "Low/Moderate Income, High Spending (Careless)",
        "description": "Customers with lower to moderate income but high spending scores. Highly active and loyal buyers.",
        "recommendation": "Offer value bundles, discount incentives, flexible payment plans, and referral bonuses.",
    },
    2: {
        "cluster_name": "Low-Engagement High-Income Customers",
        "cluster_label": "High Income, Low Spending (Cautious)",
        "description": "Customers with high income but low spending scores. High potential but currently untapped revenue stream.",
        "recommendation": "Re-engage through tailored marketing campaigns, premium product trials, and trust-building incentives.",
    },
}


def load_models():
    base_dir = Path(__file__).parent.parent
    kmeans_candidates = [
        base_dir / "models" / "kmeans_model.pkl",
        Path(__file__).with_name("kmeans_model.pkl"),
        Path.cwd() / "Unsupervised_Learning" / "models" / "kmeans_model.pkl",
        Path.cwd() / "models" / "kmeans_model.pkl",
    ]
    scaler_candidates = [
        base_dir / "models" / "scaler.pkl",
        Path(__file__).with_name("scaler.pkl"),
        Path.cwd() / "Unsupervised_Learning" / "models" / "scaler.pkl",
        Path.cwd() / "models" / "scaler.pkl",
    ]

    kmeans = None
    scaler = None

    for path in kmeans_candidates:
        if path.exists():
            kmeans = joblib.load(path)
            break

    for path in scaler_candidates:
        if path.exists():
            scaler = joblib.load(path)
            break

    if kmeans is None:
        raise FileNotFoundError("kmeans_model.pkl not found. Place it in Unsupervised_Learning/models/ or near this script.")
    if scaler is None:
        raise FileNotFoundError("scaler.pkl not found. Place it in Unsupervised_Learning/models/ or near this script.")

    return kmeans, scaler


kmeans, scaler = load_models()
app = FastAPI(title="Customer Segmentation API")


def predict_record(payload):
    missing = [field for field in FEATURE_COLUMNS if field not in payload]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing fields: {missing}")

    df = pd.DataFrame([payload], columns=FEATURE_COLUMNS)
    scaled_data = scaler.transform(df)
    cluster = int(kmeans.predict(scaled_data)[0])
    details = CLUSTER_DETAILS.get(cluster, {
        "cluster_name": f"Cluster {cluster}",
        "cluster_label": f"Cluster {cluster}",
        "description": "No description available.",
        "recommendation": "No recommendation available.",
    })

    return {
        "cluster": cluster,
        "cluster_name": details["cluster_name"],
        "cluster_label": details["cluster_label"],
        "description": details["description"],
        "recommendation": details["recommendation"],
        "annual_income_k": float(payload["annual_income_k"]),
        "spending_score": float(payload["spending_score"]),
    }


@app.get("/")
def root():
    return {"message": "Customer segmentation model API is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/clusters")
def get_clusters():
    return {"clusters": CLUSTER_DETAILS}


@app.post("/predict")
async def predict(request: Request):
    try:
        payload = await request.json()
        return predict_record(payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/predict_batch")
async def predict_batch(request: Request):
    try:
        payload = await request.json()
        records = payload.get("records", [])
        if not isinstance(records, list):
            raise HTTPException(status_code=400, detail="'records' must be a list")
        return {"results": [predict_record(item) for item in records]}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
