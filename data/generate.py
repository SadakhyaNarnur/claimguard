"""Generate synthetic insurance claims data."""
import numpy as np
import pandas as pd

np.random.seed(42)
N = 5000

claim_amounts = np.random.exponential(scale=5000, size=N)
days_to_report = np.random.poisson(lam=3, size=N)
num_prior_claims = np.random.poisson(lam=1, size=N)
policy_age_years = np.random.uniform(0.5, 20, size=N)
claim_type = np.random.choice(["auto", "home", "health", "life"], size=N, p=[0.4, 0.3, 0.2, 0.1])

# Fraud probability influenced by features
fraud_score = (
    0.3 * (claim_amounts > 15000).astype(float)
    + 0.2 * (days_to_report > 10).astype(float)
    + 0.25 * (num_prior_claims > 2).astype(float)
    + 0.15 * (policy_age_years < 1).astype(float)
    + 0.1 * (claim_type == "auto").astype(float)
    + np.random.uniform(0, 0.2, size=N)
)
is_fraud = (fraud_score > 0.5).astype(int)

df = pd.DataFrame({
    "claim_amount": claim_amounts.round(2),
    "days_to_report": days_to_report,
    "num_prior_claims": num_prior_claims,
    "policy_age_years": policy_age_years.round(2),
    "claim_type": claim_type,
    "is_fraud": is_fraud,
})

df.to_csv("data/claims.csv", index=False)
print(f"Generated {N} claims — fraud rate: {is_fraud.mean():.1%}")
