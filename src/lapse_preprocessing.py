"""Part B.Q5 — Leakage-free feature preprocessing."""

import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .lapse_prediction import (
    FEATURE_COLUMNS,
    run_lapse_dataset,
)


# ---------------------------------------------------------
# B.Q5.1 — Define numeric and categorical features
# ---------------------------------------------------------

NUMERIC_FEATURES = [
    "recency",
    "frequency",
    "monetary",
    "avg_discount_pct",
    "cancellation_rate",
    "avg_delivery_minutes",
    "age",
    "tenure_days",
]

CATEGORICAL_FEATURES = [
    "favourite_category",
    "city",
    "gender",
]


# ---------------------------------------------------------
# B.Q5.2 — Create an unfitted preprocessor
# ---------------------------------------------------------

def build_preprocessor():
    """Build scaling and encoding without fitting on data."""

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler(),
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                OneHotEncoder(
                    drop="first",
                    handle_unknown="ignore",
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


# ---------------------------------------------------------
# B.Q5.3 — Create a model pipeline for B.Q6
# ---------------------------------------------------------

def build_model_pipeline(model):
    """
    Create a fresh pipeline for leakage-free model training.

    GridSearchCV will fit preprocessing separately within
    each training fold when this pipeline is passed to it.
    """

    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", model),
        ]
    )


# ---------------------------------------------------------
# B.Q5.4 — Validate all required features
# ---------------------------------------------------------

def validate_feature_columns(dataset, X_train, X_test):
    """Confirm feature coverage and zero missing values."""

    expected = set(FEATURE_COLUMNS)

    actual = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    assert len(actual) == 11, (
        f"Expected 11 features, got {len(actual)}"
    )

    assert len(set(actual)) == len(actual), (
        "Duplicate feature names found"
    )

    assert set(actual) == expected, (
        "Numeric/categorical features do not match B.Q4"
    )

    assert set(X_train.columns) == expected
    assert set(X_test.columns) == expected

    # Check the full cohort.
    missing_counts = dataset[FEATURE_COLUMNS].isna().sum()

    assert (missing_counts == 0).all(), (
        "Missing feature values in cohort:\n"
        f"{missing_counts[missing_counts > 0]}"
    )

    # Check training and test separately.
    assert not X_train.isna().any().any(), (
        "Training features contain missing values"
    )

    assert not X_test.isna().any().any(), (
        "Test features contain missing values"
    )

    print("PASS | All 11 required features assigned")
    print("PASS | No missing values in any feature column")


# ---------------------------------------------------------
# B.Q5.5 — Validate fitted scaler and encoder
# ---------------------------------------------------------

def validate_fitted_preprocessor(
    dataset,
    X_train,
    preprocessor,
):
    """Confirm preprocessing was learned from training rows."""

    scaler = preprocessor.named_transformers_["numeric"]
    encoder = preprocessor.named_transformers_["categorical"]

    # Every numeric mean must match the training data.
    expected_means = (
        X_train[NUMERIC_FEATURES]
        .mean()
        .to_numpy()
    )

    assert np.allclose(
        scaler.mean_,
        expected_means,
    ), "Scaler was not fitted on the training rows"

    assert encoder.drop == "first"
    assert encoder.handle_unknown == "ignore"

    # Verify that categorical columns were encoded.
    encoded_names = encoder.get_feature_names_out(
        CATEGORICAL_FEATURES
    )

    for column in CATEGORICAL_FEATURES:
        assert any(
            name.startswith(f"{column}_")
            for name in encoded_names
        ), f"Missing one-hot encoding for {column}"

    print("PASS | StandardScaler fitted on training rows")
    print("PASS | OneHotEncoder fitted on training rows")
    print("PASS | City, gender and category one-hot encoded")

    # Assignment: print training and full-cohort
    # frequency means.
    frequency_index = NUMERIC_FEATURES.index("frequency")

    scaler_frequency_mean = scaler.mean_[frequency_index]
    cohort_frequency_mean = dataset["frequency"].mean()

    print("\n=== Frequency Mean Comparison ===")
    print(
        "Fitted scaler frequency mean: "
        f"{scaler_frequency_mean:.4f}"
    )
    print(
        "Full-cohort frequency mean:   "
        f"{cohort_frequency_mean:.4f}"
    )

    print(
        "Explanation: The means may differ because "
        "the scaler learns from training customers only, "
        "while the cohort mean includes both training "
        "and test customers."
    )


# ---------------------------------------------------------
# B.Q5.6 — Run preprocessing and validate output
# ---------------------------------------------------------

def run_lapse_preprocessing():
    """Run B.Q4, fit B.Q5 preprocessing and validate."""

    dataset, X_train, X_test, y_train, y_test = (
        run_lapse_dataset()
    )

    validate_feature_columns(
        dataset,
        X_train,
        X_test,
    )

    # Create a new, unfitted preprocessor.
    preprocessor = build_preprocessor()

    # Fit ONLY on training rows.
    X_train_transformed = preprocessor.fit_transform(
        X_train
    )

    # Transform test rows using training-fitted parameters.
    X_test_transformed = preprocessor.transform(
        X_test
    )

    validate_fitted_preprocessor(
        dataset,
        X_train,
        preprocessor,
    )

    assert (
        X_train_transformed.shape[0] == len(X_train)
    ), "Training row count changed"

    assert (
        X_test_transformed.shape[0] == len(X_test)
    ), "Test row count changed"

    assert (
        X_train_transformed.shape[1]
        == X_test_transformed.shape[1]
    ), "Training/test encoded feature counts differ"

    print("\n=== B.Q5: Preprocessing Summary ===")
    print(
        f"Original feature count: {len(FEATURE_COLUMNS)}"
    )
    print(
        f"Transformed training shape: "
        f"{X_train_transformed.shape}"
    )
    print(
        f"Transformed test shape: "
        f"{X_test_transformed.shape}"
    )

    print("\nB.Q5 ALL CHECKPOINTS PASSED")

    return (
        dataset,
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor,
    )


if __name__ == "__main__":
    run_lapse_preprocessing()
