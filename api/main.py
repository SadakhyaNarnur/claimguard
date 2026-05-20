"""FastAPI fraud-detection service."""
from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

ARTIFACTS = Path("model/artifacts")
clf = joblib.load(ARTIFACTS / "model.joblib")
enc = joblib.load(ARTIFACTS / "encoder.joblib")

app = FastAPI(title="ClaimGuard", version="1.0.0")
app.mount("/ui", StaticFiles(directory="ui"), name="ui")


class Claim(BaseModel):
    claim_amount: float
    days_to_report: int
    num_prior_claims: int
    policy_age_years: float
    claim_type: str  # auto | home | health | life


@app.get("/")
def root():
    return FileResponse("ui/index.html")


@app.post("/predict")
def predict(claim: Claim):
    row = pd.DataFrame([claim.model_dump()])
    row["claim_type"] = enc.transform(row["claim_type"])
    prob = float(clf.predict_proba(row.values)[0, 1])
    label = "FRAUD" if prob >= 0.2 else "LEGIT"
    return {"label": label, "fraud_probability": round(prob, 4)}


@app.get("/health")
def health():
    return {"status": "ok"}
