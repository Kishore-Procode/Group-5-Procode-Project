from pathlib import Path

import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Request

FEATURE_COLUMNS = ["income", "credit_score", "loan_amount", "employment_years"]


def load_model():
    candidates = [
        Path(__file__).with_name("loan_approval_model.pkl"),
        Path.cwd() / "loan_approval_model.pkl",
        Path.cwd() / "Supervised_Learning" / "notebooks" / "loan_approval_model.pkl",
        Path.cwd().parent / "Supervised_Learning" / "notebooks" / "loan_approval_model.pkl",
    ]

    for path in candidates:
        if path.exists():
            return joblib.load(path)

    raise FileNotFoundError("Model file not found. Place loan_approval_model.pkl near this script or in the project root.")


model = load_model()
app = FastAPI(title="Loan Approval API")


def predict_record(payload):
    missing = [field for field in FEATURE_COLUMNS if field not in payload]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing fields: {missing}")

    df = pd.DataFrame([payload], columns=FEATURE_COLUMNS)
    prediction = int(model.predict(df)[0])
    probability = float(model.predict_proba(df)[0][1])

    return {
        "prediction": prediction,
        "approval_probability": round(probability, 4),
        "status": "Approved" if prediction == 1 else "Rejected",
    }


@app.get("/")
def root():
    return {"message": "Loan approval model API is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


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
