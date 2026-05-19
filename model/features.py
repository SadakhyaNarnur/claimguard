"""Feature engineering shared by train and inference."""
import pandas as pd
from sklearn.preprocessing import LabelEncoder

CATEGORICAL = ["claim_type"]
NUMERIC = ["claim_amount", "days_to_report", "num_prior_claims", "policy_age_years"]
FEATURES = NUMERIC + CATEGORICAL


def encode(df: pd.DataFrame, encoder: LabelEncoder | None = None) -> tuple[pd.DataFrame, LabelEncoder]:
    df = df.copy()
    enc = encoder or LabelEncoder()
    if encoder is None:
        df["claim_type"] = enc.fit_transform(df["claim_type"])
    else:
        df["claim_type"] = enc.transform(df["claim_type"])
    return df[FEATURES], enc
