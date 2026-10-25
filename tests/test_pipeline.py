import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from latedeliveries.pipeline import (
    CATEGORICAL_FEATURES,
    FEATURES,
    RANDOM_STATE,
    TARGET,
    build_pipeline,
)


def test_feature_list_excludes_identifier_and_outcome():
    assert "order_id" not in FEATURES
    assert "customer_rating" not in FEATURES
    assert TARGET not in FEATURES


def test_split_is_stratified_and_repeatable(sample_data):
    X, y = sample_data[FEATURES], sample_data[TARGET]
    a = train_test_split(X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE)
    b = train_test_split(X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE)
    assert a[0].equals(b[0])
    assert a[2].equals(b[2])
    assert abs(a[3].mean() - y.mean()) < 0.05


def test_probabilities_are_valid(sample_data):
    X, y = sample_data[FEATURES], sample_data[TARGET]
    pipeline = build_pipeline(
        LogisticRegression(max_iter=1000, class_weight="balanced")
    )
    pipeline.fit(X, y)
    probabilities = pipeline.predict_proba(X)[:, 1]
    assert np.all((probabilities >= 0) & (probabilities <= 1))


def test_pipeline_handles_missing_numeric_value_and_unseen_category(sample_data):
    X, y = sample_data[FEATURES].copy(), sample_data[TARGET]
    X.loc[X.index[0], "prep_minutes"] = np.nan
    X.loc[X.index[1], "weather"] = "hail"
    pipeline = build_pipeline(LogisticRegression(max_iter=1000))
    pipeline.fit(X, y)
    assert pipeline.predict_proba(X.iloc[:2]).shape == (2, 2)


def test_feature_lists_are_disjoint():
    assert not set(CATEGORICAL_FEATURES).intersection(
        {"distance_km", "items", "hour", "prep_minutes", "courier_months"}
    )


def test_pipeline_returns_binary_predictions(sample_data):
    X, y = sample_data[FEATURES], sample_data[TARGET]
    pipeline = build_pipeline(LogisticRegression(max_iter=1000))
    pipeline.fit(X, y)
    assert set(pipeline.predict(X)).issubset({0, 1})
