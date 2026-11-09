
"""
Part B.Q7 — Training-Only Class Balancing

Requirements:
(a) Keep all original training rows and oversample
    lapsed training rows with replacement, random_state=42.
(b) Refit the best Logistic Regression from B.Q6.
(c) Compare precision, recall and F1 on the same
    untouched test set.
(d) Identify the largest metric change and explain
    the implications for the retention team.

Run:
    python -m src.lapse_class_balancing
"""

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

from .lapse_prediction import run_lapse_dataset
from .lapse_preprocessing import build_model_pipeline


# Best Logistic Regression parameter from our B.Q6 logs.
BEST_LOGISTIC_C = 100
RANDOM_STATE = 42


# =========================================================
# B.Q7(a) — OVERSAMPLE TRAINING SET ONLY
# =========================================================

def oversample_training_data(X_train, y_train):
    """Duplicate lapsed training rows until classes are equal."""

    print("\n" + "=" * 60)
    print("B.Q7(a) — TRAINING-ONLY OVERSAMPLING")
    print("=" * 60)

    X = X_train.reset_index(drop=True)
    y = y_train.reset_index(drop=True)

    original_counts = y.value_counts().sort_index()

    print("\nBefore oversampling:")
    print(original_counts.to_string())

    assert set(original_counts.index) == {0, 1}
    assert original_counts[0] > original_counts[1], (
        "Expected lapsed class (1) to be the minority"
    )

    majority_indices = np.flatnonzero(y.to_numpy() == 0)
    minority_indices = np.flatnonzero(y.to_numpy() == 1)

    target_count = len(majority_indices)
    duplicate_count = target_count - len(minority_indices)

    # Draw only from lapsed training rows.
    rng = np.random.RandomState(RANDOM_STATE)

    duplicate_indices = rng.choice(
        minority_indices,
        size=duplicate_count,
        replace=True,
    )

    # Preserve every original row and append duplicates.
    balanced_indices = np.concatenate([
        np.arange(len(y)),
        duplicate_indices,
    ])

    X_balanced = (
        X.iloc[balanced_indices]
        .reset_index(drop=True)
    )

    y_balanced = (
        y.iloc[balanced_indices]
        .reset_index(drop=True)
    )

    balanced_counts = y_balanced.value_counts().sort_index()

    print("\nAfter oversampling:")
    print(balanced_counts.to_string())

    print(f"\nOriginal training rows: {len(X_train)}")
    print(f"Duplicated lapsed rows: {duplicate_count}")
    print(f"Balanced training rows: {len(X_balanced)}")

    # Verify both classes are balanced.
    assert balanced_counts[0] == balanced_counts[1]

    # Verify all original training rows were retained.
    pd.testing.assert_frame_equal(
        X_balanced.iloc[:len(X)].reset_index(drop=True),
        X,
    )

    pd.testing.assert_series_equal(
        y_balanced.iloc[:len(y)].reset_index(drop=True),
        y,
    )

    # Verify duplicated rows belong to the lapsed class.
    assert (
        y_balanced.iloc[len(y):] == 1
    ).all()

    assert len(X_balanced) == len(y_balanced)
    assert not X_balanced.isna().any().any()

    print("\nPASS | All original training rows retained")
    print("PASS | Only lapsed training rows duplicated")
    print("PASS | Sampling with replacement and random_state=42")
    print("PASS | Training class counts are equal")

    return X_balanced, y_balanced


# =========================================================
# B.Q7(b) — REFIT BEST LOGISTIC REGRESSION
# =========================================================

def fit_logistic_model(X_train, y_train, model_name):
    """Fit Logistic Regression with preprocessing pipeline."""

    pipeline = build_model_pipeline(
        LogisticRegression(
            C=BEST_LOGISTIC_C,
            max_iter=1000,
        )
    )

    pipeline.fit(X_train, y_train)

    print(
        f"PASS | Fitted {model_name} "
        f"using C={BEST_LOGISTIC_C}"
    )

    return pipeline


# =========================================================
# B.Q7(c) — EVALUATE ON UNCHANGED TEST SET
# =========================================================

def evaluate_model(model_name, model, X_test, y_test):
    """Evaluate predictions for the lapsed class (1)."""

    predictions = model.predict(X_test)

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    )

    tn, fp, fn, tp = matrix.ravel()

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    print(f"\n=== {model_name} ===")
    print("Confusion matrix (actual rows, predicted columns):")
    print(matrix)

    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1:        {metrics['f1']:.4f}")
    print(f"FP: {fp} | FN: {fn}")

    assert matrix.sum() == len(y_test)

    return metrics


# =========================================================
# B.Q7(d) — COMPARE METRICS AND EXPLAIN TRADEOFF
# =========================================================

def compare_balancing_results(original, balanced):
    """Identify the largest metric change and its meaning."""

    print("\n" + "=" * 60)
    print("B.Q7(d) — BEFORE VS AFTER BALANCING")
    print("=" * 60)

    comparison = pd.DataFrame([
        original,
        balanced,
    ])

    print(
        comparison[
            ["model", "precision", "recall", "f1"]
        ].to_string(index=False)
    )

    metric_changes = {
        metric: balanced[metric] - original[metric]
        for metric in ["precision", "recall", "f1"]
    }

    print("\nMetric changes after oversampling:")

    for metric, change in metric_changes.items():
        print(f"{metric.capitalize():10s}: {change:+.4f}")

    largest_change = max(
        metric_changes,
        key=lambda metric: abs(metric_changes[metric]),
    )

    largest_value = metric_changes[largest_change]

    print(
        "\nMetric with largest absolute change: "
        f"{largest_change.capitalize()} "
        f"({largest_value:+.4f})"
    )

    print("\n=== Retention Team Interpretation ===")

    if metric_changes["recall"] > 0:
        print(
            "Recall increased: more genuinely lapsing "
            "customers are identified, reducing missed "
            "retention opportunities (false negatives)."
        )
    elif metric_changes["recall"] < 0:
        print(
            "Recall decreased: more genuinely lapsing "
            "customers are missed."
        )
    else:
        print("Recall remained unchanged.")

    if metric_changes["precision"] < 0:
        print(
            "Precision decreased: a larger share of "
            "targeted customers may actually be loyal, "
            "increasing unnecessary campaign costs."
        )
    elif metric_changes["precision"] > 0:
        print(
            "Precision increased: a larger share of "
            "targeted customers genuinely lapsed."
        )
    else:
        print("Precision remained unchanged.")

    if largest_change == "recall":
        print(
            "Largest impact: the retention team can "
            "reach more at-risk customers, but should "
            "monitor unnecessary outreach."
        )
    elif largest_change == "precision":
        print(
            "Largest impact: the proportion of useful "
            "retention targets changed most."
        )
    else:
        print(
            "Largest impact: the overall balance "
            "between precision and recall changed most."
        )

    return comparison, metric_changes


# =========================================================
# B.Q7 — MAIN RUNNER
# =========================================================

def run_class_balancing():

    print("\n" + "=" * 60)
    print("PART B.Q7 — CLASS BALANCING")
    print("=" * 60)

    dataset, X_train, X_test, y_train, y_test = (
        run_lapse_dataset()
    )

    # Preserve the original test data for verification.
    X_test_original = X_test.copy(deep=True)
    y_test_original = y_test.copy(deep=True)

    # B.Q7(a): Oversample training data only.
    X_balanced, y_balanced = oversample_training_data(
        X_train,
        y_train,
    )

    # B.Q7(b): Train original and balanced versions
    # using the same best Logistic Regression C.
    original_model = fit_logistic_model(
        X_train,
        y_train,
        "Original Logistic Regression",
    )

    balanced_model = fit_logistic_model(
        X_balanced,
        y_balanced,
        "Oversampled Logistic Regression",
    )

    # B.Q7(c): Evaluate both on the same test set.
    original_result = evaluate_model(
        "Original Logistic Regression",
        original_model,
        X_test,
        y_test,
    )

    balanced_result = evaluate_model(
        "Oversampled Logistic Regression",
        balanced_model,
        X_test,
        y_test,
    )

    # B.Q7(d): Report the metric changes and implications.
    comparison, metric_changes = compare_balancing_results(
        original_result,
        balanced_result,
    )

    # Confirm the test set was not modified.
    pd.testing.assert_frame_equal(
        X_test,
        X_test_original,
    )

    pd.testing.assert_series_equal(
        y_test,
        y_test_original,
    )

    assert len(X_test) == 232
    assert len(y_test) == 232

    # Compare original model against validated B.Q6
    # result. Allow a tiny floating-point tolerance.
    assert np.isclose(
        original_result["precision"],
        0.644444,
        atol=0.001,
    ), "Original precision differs from B.Q6"

    assert np.isclose(
        original_result["recall"],
        0.651685,
        atol=0.001,
    ), "Original recall differs from B.Q6"

    assert np.isclose(
        original_result["f1"],
        0.648045,
        atol=0.001,
    ), "Original F1 differs from B.Q6"

    print("\n" + "=" * 60)
    print("B.Q7 VALIDATION SUMMARY")
    print("=" * 60)

    print("PASS | Training-only oversampling")
    print("PASS | All original training rows retained")
    print("PASS | Lapsed rows duplicated with replacement")
    print("PASS | random_state=42")
    print("PASS | Both training classes balanced")
    print(f"PASS | Best Logistic Regression C={BEST_LOGISTIC_C}")
    print("PASS | Same untouched test set used")
    print("PASS | Original model matches B.Q6 metrics")
    print("PASS | Precision, recall and F1 compared")
    print("PASS | Largest metric change identified")
    print("PASS | Retention team interpretation reported")

    print("\nB.Q7 ALL CHECKPOINTS PASSED")

    return {
        "dataset": dataset,
        "original_model": original_model,
        "balanced_model": balanced_model,
        "original_result": original_result,
        "balanced_result": balanced_result,
        "comparison": comparison,
        "metric_changes": metric_changes,
    }


if __name__ == "__main__":
    run_class_balancing()
