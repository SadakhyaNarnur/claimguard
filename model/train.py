"""
ClaimGuard — Model Training
Trains a RandomForest classifier on aggregated fraud-ring features.
Saves model.joblib and encoder.joblib to model/artifacts/.
"""

import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from model.features import encode, FEATURES

DATA_PATH      = ROOT / "data" / "claims.csv"
ARTIFACTS_PATH = ROOT / "model" / "artifacts"


def main():
    print("── ClaimGuard Training ─────────────────────────────────────")

    print("\n[1/4] Loading and engineering features ...")
    df = pd.read_csv(DATA_PATH)
    print(f"      Loaded {len(df):,} claims  "
          f"({df.is_fraud.sum():,} fraud, "
          f"{(df.is_fraud == 0).sum():,} legit)")

    X_df, enc = encode(df)
    X = X_df.values
    y = df["is_fraud"].values
    print(f"      Features: {FEATURES}")

    print("\n[2/4] Splitting 80 / 20 (stratified) ...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"      Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    print("\n[3/4] 5-fold cross-validation ...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train,
                                cv=cv, scoring="roc_auc", n_jobs=-1)
    print(f"      CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    print("\n[4/4] Training final model on full training set ...")
    model.fit(X_train, y_train)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    auc     = roc_auc_score(y_test, y_proba)

    print(f"\n── Test-set results ────────────────────────────────────────")
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))
    print(f"   ROC-AUC : {auc:.4f}")

    print("\n── Feature importance ──────────────────────────────────────")
    importances = sorted(
        zip(FEATURES, model.feature_importances_),
        key=lambda x: x[1], reverse=True
    )
    for feat, imp in importances:
        bar = "█" * int(imp * 40)
        print(f"  {feat:<35} {bar} {imp:.4f}")

    ARTIFACTS_PATH.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, ARTIFACTS_PATH / "model.joblib")
    joblib.dump(enc,   ARTIFACTS_PATH / "encoder.joblib")
    print(f"\n✅ Artifacts saved → {ARTIFACTS_PATH}")


if __name__ == "__main__":
    main()
