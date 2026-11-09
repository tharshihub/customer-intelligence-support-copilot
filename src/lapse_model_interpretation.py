
"""
Part B.Q8 — Model Interpretation, Persistence and Retention

(a) Report top 5 Random Forest feature importances and
    3 Logistic Regression coefficient signs, each with
    a one-line business interpretation.

(b) Save selected model to outputs/lapse_model.joblib,
    reload it, and verify identical test predictions.

(c) Rank Premium test-set customers by predicted lapse
    probability, save top 10 to retention_list.csv,
    and report how many truly lapsed.

Run:
    python -m src.lapse_model_interpretation
"""

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import f1_score

from .config import OUTPUT_DIR
from .lapse_model_training import run_lapse_model_training


MODEL_PATH = OUTPUT_DIR / "lapse_model.joblib"
RETENTION_PATH = OUTPUT_DIR / "retention_list.csv"
SEGMENTS_PATH = OUTPUT_DIR / "customer_segments.csv"


# =========================================================
# B.Q8(a) — RANDOM FOREST FEATURE IMPORTANCE
# =========================================================

BUSINESS_MEANINGS = {
    "recency": "Days since the last completed order",
    "frequency": "Historical completed-order frequency",
    "monetary": "Historical customer spending",
    "avg_discount_pct": "Historical discount usage",
    "cancellation_rate": "Historical cancellation behaviour",
    "avg_delivery_minutes": "Historical delivery experience",
    "age": "Customer age",
    "tenure_days": "Length of customer relationship",
    "favourite_category": "Customer's preferred product category",
    "city": "Customer's location",
    "gender": "Recorded customer gender",
}


def get_business_description(encoded_feature):
    """Map an encoded feature to its business meaning."""

    clean_name = encoded_feature.split("__", 1)[-1]

    # Match exact numeric names or one-hot categorical names.
    for feature, description in BUSINESS_MEANINGS.items():
        if (
            clean_name == feature
            or clean_name.startswith(feature + "_")
        ):
            return description

    return "Customer characteristic"


def report_forest_importances(forest_pipeline):
    """Report top 5 feature importances with business readings."""

    print("\n" + "=" * 60)
    print("B.Q8(a) — TOP FIVE RANDOM FOREST FEATURES")
    print("=" * 60)

    preprocessor = forest_pipeline.named_steps["preprocessor"]
    forest = forest_pipeline.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out()
    importances = forest.feature_importances_

    assert len(feature_names) == len(importances)
    assert np.isclose(importances.sum(), 1.0)

    importance_table = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values(
        "importance",
        ascending=False,
    ).reset_index(drop=True)

    top_five = importance_table.head(5).copy()

    print("\nTop five features:")
    print(top_five.to_string(index=False))

    print("\nOne-line business interpretations:")

    for rank, row in enumerate(
        top_five.itertuples(index=False),
        start=1,
    ):
        description = get_business_description(row.feature)

        print(
            f"{rank}. {row.feature} "
            f"(importance={row.importance:.4f}): "
            f"{description} contributes to predicting "
            f"customer lapse."
        )

    print(
        "\nNote: Random Forest importance indicates "
        "predictive usefulness, not whether a feature "
        "increases or decreases lapse risk."
    )

    print("PASS | Five Random Forest features interpreted")

    return top_five


# =========================================================
# B.Q8(a) — LOGISTIC REGRESSION COEFFICIENTS
# =========================================================

def report_logistic_coefficients(logistic_pipeline):
    """Report 3 coefficient signs and business interpretations."""

    print("\n" + "=" * 60)
    print("B.Q8(a) — LOGISTIC REGRESSION COEFFICIENTS")
    print("=" * 60)

    preprocessor = logistic_pipeline.named_steps["preprocessor"]
    logistic = logistic_pipeline.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out()
    coefficients = logistic.coef_[0]

    assert len(feature_names) == len(coefficients)
    assert list(logistic.classes_) == [0, 1]

    coefficient_table = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefficients,
    })

    selected_features = [
        "numeric__recency",
        "numeric__frequency",
        "numeric__monetary",
    ]

    selected_rows = []

    print("\nThree business-relevant numeric features:")

    for feature in selected_features:

        matched = coefficient_table.loc[
            coefficient_table["feature"] == feature
        ]

        assert len(matched) == 1, (
            f"Missing Logistic Regression feature: {feature}"
        )

        coefficient = float(
            matched.iloc[0]["coefficient"]
        )

        description = get_business_description(feature)

        if coefficient > 0:
            sign = "Positive"
            interpretation = (
                "Higher values are associated with "
                "higher predicted odds of lapsing."
            )
        elif coefficient < 0:
            sign = "Negative"
            interpretation = (
                "Higher values are associated with "
                "lower predicted odds of lapsing."
            )
        else:
            sign = "Zero"
            interpretation = (
                "The feature has no fitted linear "
                "contribution to lapse log-odds."
            )

        print(
            f"\nFeature: {feature}\n"
            f"Coefficient: {coefficient:+.6f}\n"
            f"Sign: {sign}\n"
            f"Business reading: {description}. "
            f"{interpretation}"
        )

        selected_rows.append({
            "feature": feature,
            "coefficient": coefficient,
            "sign": sign,
        })

    print(
        "\nNote: Numeric features are standardized. "
        "Coefficient signs indicate associations, "
        "holding other model features constant."
    )

    print("PASS | Three Logistic Regression coefficients interpreted")

    return pd.DataFrame(selected_rows)


# =========================================================
# B.Q8(b) — SAVE AND RELOAD SELECTED MODEL
# =========================================================

def save_and_verify_model(selected_model, X_test):
    """Save fitted pipeline and verify identical predictions."""

    print("\n" + "=" * 60)
    print("B.Q8(b) — MODEL SAVE AND RELOAD")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    original_predictions = selected_model.predict(X_test)
    original_probabilities = selected_model.predict_proba(
        X_test
    )

    joblib.dump(selected_model, MODEL_PATH)

    reloaded_model = joblib.load(MODEL_PATH)

    reloaded_predictions = reloaded_model.predict(X_test)
    reloaded_probabilities = reloaded_model.predict_proba(
        X_test
    )

    # Exact prediction equality required by the assignment.
    np.testing.assert_array_equal(
        original_predictions,
        reloaded_predictions,
    )

    # Additional verification of probabilities.
    np.testing.assert_allclose(
        original_probabilities,
        reloaded_probabilities,
        rtol=0,
        atol=0,
    )

    assert MODEL_PATH.is_file()

    print(f"Saved model: {MODEL_PATH}")
    print(f"Verified test customers: {len(X_test)}")
    print("PASS | Original and reloaded predictions match exactly")
    print("PASS | Original and reloaded probabilities match exactly")

    return reloaded_model


# =========================================================
# B.Q8(c) — PREMIUM RETENTION RECOMMENDATIONS
# =========================================================

def create_retention_list(
    dataset,
    X_test,
    y_test,
    selected_model,
):
    """Create top-10 Premium test-customer retention list."""

    print("\n" + "=" * 60)
    print("B.Q8(c) — PREMIUM CUSTOMER RETENTION LIST")
    print("=" * 60)

    assert SEGMENTS_PATH.is_file(), (
        f"Missing B.Q3 segment file: {SEGMENTS_PATH}"
    )

    segments = pd.read_csv(SEGMENTS_PATH)

    required_columns = {
        "customer_id",
        "segment_id",
        "segment_name",
        "is_premium",
    }

    assert required_columns.issubset(segments.columns)
    assert segments["customer_id"].is_unique
    assert dataset["customer_id"].is_unique

    # B.Q4 split retains original dataset row indices.
    assert X_test.index.equals(y_test.index)
    assert X_test.index.isin(dataset.index).all()

    # Construct test-customer IDs and actual lapse labels.
    test_customers = pd.DataFrame({
        "customer_id": dataset.loc[
            X_test.index,
            "customer_id",
        ].to_numpy(),
        "actually_lapsed": y_test.to_numpy(),
    })

    # Get probability corresponding to lapsed = 1.
    class_labels = list(selected_model.classes_)
    assert 1 in class_labels

    lapsed_column = class_labels.index(1)

    test_customers["lapse_probability"] = (
        selected_model.predict_proba(X_test)[
            :, lapsed_column
        ]
    )

    # Handle true/false values loaded from CSV.
    premium_flags = (
        segments["is_premium"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1"])
    )

    premium_segments = segments.loc[
        premium_flags,
        ["customer_id"],
    ].copy()

    assert not premium_segments.empty, (
        "No Premium customers found in B.Q3 segments"
    )

    # Keep only test customers belonging to Premium.
    premium_test = test_customers.merge(
        premium_segments,
        on="customer_id",
        how="inner",
        validate="one_to_one",
    )

    assert not premium_test.empty, (
        "No Premium customers in the test set"
    )

    # Rank highest lapse probability first.
    premium_test = premium_test.sort_values(
        ["lapse_probability", "customer_id"],
        ascending=[False, True],
    ).reset_index(drop=True)

    # Top 10, or all if fewer than 10.
    retention_list = premium_test[
        [
            "customer_id",
            "lapse_probability",
            "actually_lapsed",
        ]
    ].head(10).copy()

    assert len(retention_list) == min(
        10,
        len(premium_test),
    )

    assert retention_list["customer_id"].is_unique
    assert retention_list["lapse_probability"].between(
        0, 1
    ).all()

    assert (
        retention_list["lapse_probability"]
        .is_monotonic_decreasing
    )

    # Save the exact required CSV columns.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    retention_list.to_csv(
        RETENTION_PATH,
        index=False,
    )

    # Count truly lapsed among recommended customers.
    actual_lapsed_count = int(
        retention_list["actually_lapsed"].sum()
    )

    print(f"\nPremium test customers: {len(premium_test)}")
    print(f"Customers recommended: {len(retention_list)}")

    print("\nTop retention recommendations:")
    print(
        retention_list.to_string(
            index=False,
            formatters={
                "lapse_probability": "{:.4f}".format,
            },
        )
    )

    print(
        "\nActually lapsed among recommendations: "
        f"{actual_lapsed_count}/{len(retention_list)}"
    )

    print(f"Saved CSV: {RETENTION_PATH}")

    # Verify saved CSV structure.
    saved_csv = pd.read_csv(RETENTION_PATH)

    assert list(saved_csv.columns) == [
        "customer_id",
        "lapse_probability",
        "actually_lapsed",
    ]

    assert len(saved_csv) == len(retention_list)

    print("PASS | Only Premium test customers selected")
    print("PASS | Ranked by predicted lapse probability")
    print("PASS | Top 10 (or fewer) saved")
    print("PASS | CSV columns match assignment")
    print("PASS | Actual lapse count reported")

    limitation = (
        "The Premium retention list is illustrative and "
        "not a leakage-free evaluation because Premium "
        "membership was derived from whole-window RFM "
        "data that includes the September lapse-label "
        "period; the lapse model's own features use "
        "only pre-cutoff information."
    )

    print("\nLIMITATION:")
    print(limitation)

    return retention_list, actual_lapsed_count


# =========================================================
# B.Q8 — MAIN RUNNER
# =========================================================

def run_model_interpretation():

    print("\n" + "=" * 60)
    print("PART B.Q8 — MODEL INTERPRETATION AND RETENTION")
    print("=" * 60)

    # Reuse B.Q6 training and model selection.
    results = run_lapse_model_training()

    dataset = results["dataset"]
    X_test = results["X_test"]
    y_test = results["y_test"]

    logistic_pipeline = (
        results["logistic_search"].best_estimator_
    )

    forest_pipeline = (
        results["forest_search"].best_estimator_
    )

    selected_name = results["selected_name"]
    selected_model = results["selected_model"]

    print(f"\nSelected B.Q6 model: {selected_name}")

    # B.Q8(a)
    top_features = report_forest_importances(
        forest_pipeline
    )

    coefficients = report_logistic_coefficients(
        logistic_pipeline
    )

    # B.Q8(b)
    reloaded_model = save_and_verify_model(
        selected_model,
        X_test,
    )

    # B.Q8(c)
    retention_list, actual_lapsed_count = (
        create_retention_list(
            dataset,
            X_test,
            y_test,
            reloaded_model,
        )
    )

    # Verify selected model still passes B.Q6 threshold.
    predictions = reloaded_model.predict(X_test)

    selected_f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    assert selected_f1 >= 0.55

    assert MODEL_PATH.exists()
    assert RETENTION_PATH.exists()

    print("\n" + "=" * 60)
    print("B.Q8 VALIDATION SUMMARY")
    print("=" * 60)

    print("PASS | Top 5 Random Forest features interpreted")
    print("PASS | 3 Logistic Regression coefficients interpreted")
    print("PASS | Selected model saved to joblib")
    print("PASS | Reloaded predictions match exactly")
    print("PASS | Premium test customers ranked correctly")
    print("PASS | Retention CSV saved with exact columns")
    print("PASS | Actual lapse count reported")
    print("PASS | Selected model F1 >= 0.55")
    print("PASS | Retention-list limitation documented")

    print(f"\nSelected model test F1: {selected_f1:.4f}")
    print(
        "Actually lapsed among recommended Premium customers: "
        f"{actual_lapsed_count}/{len(retention_list)}"
    )

    print("\nB.Q8 ALL CHECKPOINTS PASSED")
    print("PART B ALL QUESTIONS COMPLETED")

    return {
        "selected_model": reloaded_model,
        "top_features": top_features,
        "coefficients": coefficients,
        "retention_list": retention_list,
        "actual_lapsed_count": actual_lapsed_count,
    }


if __name__ == "__main__":
    run_model_interpretation()
