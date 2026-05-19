"""Train a fraud-detection classifier and save artifacts."""
import joblib
import pandas as pd
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

from model.features import encode

ARTIFACTS = Path("model/artifacts")

def main():
    df = pd.read_csv("data/claims.csv")
    X, enc = encode(df)
    y = df["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    clf = GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
    clf.fit(X_train, y_train)

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, ARTIFACTS / "model.joblib")
    joblib.dump(enc, ARTIFACTS / "encoder.joblib")

    from model.evaluate import report
    report(clf, X_test, y_test)

if __name__ == "__main__":
    main()
