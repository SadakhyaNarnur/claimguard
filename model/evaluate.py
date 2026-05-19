"""Evaluation utilities."""
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score


def report(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    print(f"\nROC-AUC: {auc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["legit", "fraud"]))
