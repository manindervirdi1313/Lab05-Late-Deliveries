import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

VOUCHER_COST = 4
MISSED_LATE_COST = 15


def order_cost(y_true, flagged):
    y_true = np.asarray(y_true)
    flagged = np.asarray(flagged, dtype=bool)
    vouchers = int(flagged.sum())
    missed_late = int(((y_true == 1) & ~flagged).sum())
    return VOUCHER_COST * vouchers + MISSED_LATE_COST * missed_late


def choose_threshold(y_true, probabilities):
    thresholds = np.round(np.arange(0.05, 1.00, 0.05), 2)
    results = []

    for threshold in thresholds:
        flagged = probabilities >= threshold
        results.append(
            {
                "threshold": float(threshold),
                "cost": int(order_cost(y_true, flagged)),
            }
        )

    best = min(results, key=lambda result: result["cost"])
    return best["threshold"], results


def classification_metrics(y_true, probabilities, threshold):
    predictions = probabilities >= threshold
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
    }
