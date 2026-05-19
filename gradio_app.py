"""Gradio UI — alternative to the FastAPI frontend."""
import gradio as gr
import joblib
import pandas as pd
from pathlib import Path

ARTIFACTS = Path("model/artifacts")
clf = joblib.load(ARTIFACTS / "model.joblib")
enc = joblib.load(ARTIFACTS / "encoder.joblib")


def predict(claim_amount, days_to_report, num_prior_claims, policy_age_years, claim_type):
    row = pd.DataFrame([{
        "claim_amount": claim_amount,
        "days_to_report": days_to_report,
        "num_prior_claims": num_prior_claims,
        "policy_age_years": policy_age_years,
        "claim_type": enc.transform([claim_type])[0],
    }])
    prob = float(clf.predict_proba(row)[0, 1])
    label = "FRAUD" if prob >= 0.5 else "LEGIT"
    return f"{label} — {prob:.1%} fraud probability"


demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="Claim Amount ($)", value=8500),
        gr.Number(label="Days to Report", value=2, precision=0),
        gr.Number(label="Prior Claims", value=1, precision=0),
        gr.Number(label="Policy Age (years)", value=5),
        gr.Dropdown(["auto", "home", "health", "life"], label="Claim Type", value="auto"),
    ],
    outputs=gr.Text(label="Result"),
    title="ClaimGuard — Fraud Detector",
)

if __name__ == "__main__":
    demo.launch()
