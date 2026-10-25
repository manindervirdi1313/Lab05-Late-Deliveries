import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.metrics import confusion_matrix

from latedeliveries.data import make_orders
from latedeliveries.evaluation import (
    classification_metrics,
    choose_threshold,
    order_cost,
)
from latedeliveries.pipeline import (
    CATEGORICAL_FEATURES,
    FEATURES,
    RANDOM_STATE,
    TARGET,
    build_pipeline,
)

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}


def main():
    MODELS.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)

    data = make_orders()
    X = data[FEATURES]
    y = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    models = {
        "dummy": DummyClassifier(strategy="prior"),
        "logistic_regression_balanced": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=150, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            random_state=RANDOM_STATE
        ),
    }

    rows = []
    for name, estimator in models.items():
        pipeline = build_pipeline(estimator)
        scores = cross_validate(
            pipeline, X_train, y_train, cv=cv, scoring=SCORING, n_jobs=-1
        )
        row = {"model": name}
        for metric in SCORING:
            row[f"{metric}_mean"] = float(scores[f"test_{metric}"].mean())
            row[f"{metric}_std"] = float(scores[f"test_{metric}"].std())
        rows.append(row)

    pd.DataFrame(rows).to_csv(REPORTS / "cv_results.csv", index=False)

    tuning_pipeline = build_pipeline(
        LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
    )
    search = GridSearchCV(
        tuning_pipeline,
        {
            "model__C": [0.1, 1.0, 10.0],
            "model__class_weight": [None, "balanced"],
        },
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)
    best_pipeline = search.best_estimator_

    oof_probabilities = cross_val_predict(
        best_pipeline,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]
    threshold, threshold_results = choose_threshold(y_train, oof_probabilities)

    # The test set is used only here, after model and threshold selection.
    best_pipeline.fit(X_train, y_train)
    test_probabilities = best_pipeline.predict_proba(X_test)[:, 1]
    test_predictions = test_probabilities >= threshold
    metrics = classification_metrics(y_test, test_probabilities, threshold)

    costs = {
        "flag_nobody": int(order_cost(y_test, np.zeros(len(y_test), dtype=bool))),
        "flag_everybody": int(order_cost(y_test, np.ones(len(y_test), dtype=bool))),
        "model": int(order_cost(y_test, test_predictions)),
    }
    matrix = confusion_matrix(y_test, test_predictions, labels=[0, 1]).tolist()

    metrics_output = {
        "threshold": threshold,
        "test_metrics": metrics,
        "confusion_matrix_labels": ["actual_on_time", "actual_late"],
        "confusion_matrix": matrix,
        "policy_costs": costs,
        "threshold_costs": threshold_results,
        "test_size": int(len(y_test)),
        "late_prevalence": float(y.mean()),
        "tuned_best_cv_roc_auc": float(search.best_score_),
        "best_params": search.best_params_,
    }
    (REPORTS / "metrics.json").write_text(
        json.dumps(metrics_output, indent=2), encoding="utf-8"
    )

    joblib.dump(best_pipeline, MODELS / "late_orders.joblib")
    model_info = {
        "sklearn_version": sklearn.__version__,
        "random_state": RANDOM_STATE,
        "threshold": threshold,
        "numeric_features": [
            feature for feature in FEATURES if feature not in CATEGORICAL_FEATURES
        ],
        "categorical_features": CATEGORICAL_FEATURES,
        "excluded_columns": {
            "order_id": "Identifier, not a predictive feature",
            "customer_rating": "Not known when the order is placed",
        },
        "best_params": search.best_params_,
        "test_metrics": metrics,
        "test_confusion_matrix": matrix,
        "policy_costs": costs,
    }
    (MODELS / "model_info.json").write_text(
        json.dumps(model_info, indent=2), encoding="utf-8"
    )

    logging.info("Best tuning settings: %s", search.best_params_)
    logging.info("Chosen threshold: %.2f", threshold)
    logging.info("Test metrics: %s", metrics)
    logging.info("Test policy costs: %s", costs)


if __name__ == "__main__":
    main()
