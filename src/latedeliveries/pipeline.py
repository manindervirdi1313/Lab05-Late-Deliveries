from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = [
    "distance_km",
    "items",
    "hour",
    "prep_minutes",
    "courier_months",
]

CATEGORICAL_FEATURES = [
    "weather",
    "day_type",
    "vehicle",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "late"
RANDOM_STATE = 42


def build_pipeline(model):
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    preprocessing = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )

    return Pipeline(
        [
            ("preprocessing", preprocessing),
            ("model", model),
        ]
    )
