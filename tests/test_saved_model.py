import joblib
from sklearn.linear_model import LogisticRegression

from latedeliveries.pipeline import FEATURES, build_pipeline


def test_saved_model_predictions_match_after_reload(sample_data, tmp_path):
    X, y = sample_data[FEATURES], sample_data["late"]
    model = build_pipeline(LogisticRegression(max_iter=1000))
    model.fit(X, y)

    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    reloaded = joblib.load(path)

    assert (model.predict(X.iloc[:10]) == reloaded.predict(X.iloc[:10])).all()
