
"""
Part B.Q6 — Lapse Prediction: Model Training and Evaluation

(a) Majority-class baseline
(b) Logistic Regression hyperparameter tuning
(c) Random Forest hyperparameter tuning
    and both models' best CV results
(d) Untouched test evaluation and model selection

Run:
    python -m src.lapse_model_training
"""

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

from .lapse_prediction import run_lapse_dataset
from .lapse_preprocessing import (
    FEATURE_COLUMNS,
    build_model_pipeline,
)


# =========================================================
# B.Q6(a) — REPORT MAJORITY-CLASS BASELINE
# =========================================================

def report_baseline(y_test):
    """
    Predict class 0 (not lapsed) for every test customer.

    Baseline is reported without fitting a model.
    Positive class: lapsed = 1.
    """

    predictions = np.zeros(len(y_test), dtype=int)

    metrics = {
        "model": "Majority-Class Baseline",
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_test, predictions, zero_division=0
        ),
    }

    print("\n" + "=" * 60)
    print("B.Q6(a) — MAJORITY-CLASS BASELINE")
    print("=" * 60)
    print("Prediction: Every customer is not lapsed (0)")
    print(f"Accuracy:          {metrics['accuracy']:.4f}")
    print(f"Lapsed precision:  {metrics['precision']:.4f}")
    print(f"Lapsed recall:     {metrics['recall']:.4f}")
    print(f"Lapsed F1:         {metrics['f1']:.4f}")

    print(
        "Interpretation: The baseline predicts no lapses, "
        "so its lapsed-class recall and F1 are zero."
    )

    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0

    return metrics


# =========================================================
# B.Q6(b) — TUNE LOGISTIC REGRESSION
# =========================================================

def tune_logistic_regression(X_train, y_train):
    """
    LogisticRegression(max_iter=1000)

    C: 0.01, 0.1, 1, 10, 100
    CV: 5 folds
    Scoring: F1 for lapsed class (1)

    5 combinations x 5 folds = 25 CV fits.
    """

    pipeline = build_model_pipeline(
        LogisticRegression(max_iter=1000)
    )

    search = GridSearchCV(
        estimator=pipeline,
        param_grid={
            "model__C": [0.01, 0.1, 1, 10, 100],
        },
        cv=5,
        scoring="f1",
        n_jobs=-1,
        refit=True,
        error_score="raise",
    )

    print("\n" + "=" * 60)
    print("B.Q6(b) — LOGISTIC REGRESSION GRID SEARCH")
    print("=" * 60)
    print("C values: [0.01, 0.1, 1, 10, 100]")
    print("Cross-validation: 5 folds")
    print("Scoring: f1")
    print("Expected CV fits: 25")

    # Raw training features: preprocessing is inside
    # the pipeline and refitted in every CV training fold.
    search.fit(X_train, y_train)

    print(f"Best C: {search.best_params_['model__C']}")
    print(f"Best CV F1: {search.best_score_:.4f}")

    return search


# =========================================================
# B.Q6(c) — TUNE RANDOM FOREST
# =========================================================

def tune_random_forest(X_train, y_train):
    """
    RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    max_depth: 3, 5, 8
    min_samples_leaf: 5, 10, 20
    CV: 5 folds
    Scoring: F1 for lapsed class (1)

    3 x 3 x 5 = 45 CV fits.
    """

    pipeline = build_model_pipeline(
        RandomForestClassifier(
            n_estimators=100,
            random_state=42,
        )
    )

    search = GridSearchCV(
        estimator=pipeline,
        param_grid={
            "model__max_depth": [3, 5, 8],
            "model__min_samples_leaf": [5, 10, 20],
        },
        cv=5,
        scoring="f1",
        n_jobs=-1,
        refit=True,
        error_score="raise",
    )

    print("\n" + "=" * 60)
    print("B.Q6(c) — RANDOM FOREST GRID SEARCH")
    print("=" * 60)
    print("max_depth: [3, 5, 8]")
    print("min_samples_leaf: [5, 10, 20]")
    print("n_estimators: 100")
    print("random_state: 42")
    print("Cross-validation: 5 folds")
    print("Scoring: f1")
    print("Expected CV fits: 45")

    search.fit(X_train, y_train)

    print(
        "Best max_depth: "
        f"{search.best_params_['model__max_depth']}"
    )
    print(
        "Best min_samples_leaf: "
        f"{search.best_params_['model__min_samples_leaf']}"
    )
    print(f"Best CV F1: {search.best_score_:.4f}")

    return search


def report_tuning_results(logistic_search, forest_search):
    """
    B.Q6(c): Report BOTH best parameter sets
    and BOTH best cross-validated F1 scores.
    """

    print("\n" + "=" * 60)
    print("B.Q6(c) — BOTH MODELS: BEST CV RESULTS")
    print("=" * 60)

    print("\nLogistic Regression:")
    print(f"  Best parameters: {logistic_search.best_params_}")
    print(f"  Best CV F1:      {logistic_search.best_score_:.4f}")

    print("\nRandom Forest:")
    print(f"  Best parameters: {forest_search.best_params_}")
    print(f"  Best CV F1:      {forest_search.best_score_:.4f}")


# =========================================================
# B.Q6(d) — EVALUATE ON UNTOUCHED TEST SET
# =========================================================

def evaluate_model(name, fitted_model, X_test, y_test):
    """
    Evaluate a tuned model on the untouched test set.

    Confusion matrix:
                  Predicted 0   Predicted 1
    Actual 0          TN            FP
    Actual 1          FN            TP
    """

    predictions = fitted_model.predict(X_test)

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    )

    tn, fp, fn, tp = matrix.ravel()

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_test, predictions, zero_division=0
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    print("\n" + "-" * 60)
    print(f"B.Q6(d) — {name.upper()}: TEST RESULTS")
    print("-" * 60)

    print("Confusion matrix [rows=actual, columns=predicted]:")
    print(matrix)

    print(f"True negatives (TN):  {tn}")
    print(f"False positives (FP): {fp}")
    print(f"False negatives (FN): {fn}")
    print(f"True positives (TP):  {tp}")

    print(f"Accuracy:          {metrics['accuracy']:.4f}")
    print(f"Lapsed precision:  {metrics['precision']:.4f}")
    print(f"Lapsed recall:     {metrics['recall']:.4f}")
    print(f"Lapsed F1:         {metrics['f1']:.4f}")

    assert matrix.sum() == len(y_test)

    return metrics


# =========================================================
# B.Q6(d) — SELECT AND JUSTIFY MODEL
# =========================================================

def select_and_justify_model(
    logistic_search,
    forest_search,
    logistic_metrics,
    forest_metrics,
):
    """
    Choose the model using training cross-validation F1.

    Test-set results are for final evaluation, not
    hyperparameter tuning or model selection.
    """

    if forest_search.best_score_ > logistic_search.best_score_:
        selected_name = "Random Forest"
        selected_model = forest_search.best_estimator_
        selected_metrics = forest_metrics
    else:
        selected_name = "Logistic Regression"
        selected_model = logistic_search.best_estimator_
        selected_metrics = logistic_metrics

    print("\n" + "=" * 60)
    print("B.Q6(d) — SELECTED MODEL AND JUSTIFICATION")
    print("=" * 60)

    print(f"Selected model: {selected_name}")
    print(f"Selected test F1: {selected_metrics['f1']:.4f}")

    justification = (
        f"We selected {selected_name} because it achieved "
        "the higher five-fold cross-validated F1 score "
        "on the training data. "
        f"On the untouched test set, its lapsed-class "
        f"precision was {selected_metrics['precision']:.4f}, "
        f"recall was {selected_metrics['recall']:.4f}, "
        f"and F1 was {selected_metrics['f1']:.4f}. "
        "Missing a genuinely lapsing customer (false negative) "
        "can result in lost revenue and a missed retention "
        "opportunity. "
        "Wrongly targeting a loyal customer (false positive) "
        "can waste discounts and marketing resources. "
        "We therefore consider recall and precision together, "
        "while recognising that the relative business costs "
        "should guide future threshold adjustments."
    )

    print("\nJustification:")
    print(justification)

    if selected_metrics["f1"] >= 0.55:
        print("\nPASS | Selected test F1 >= 0.55")
    else:
        print(
            "\nWARNING | Selected test F1 < 0.55; "
            "review dataset and feature preparation."
        )

    return selected_name, selected_model, justification


# =========================================================
# MAIN RUNNER — B.Q6(a), (b), (c), (d)
# =========================================================

def run_lapse_model_training():

    print("\n" + "=" * 60)
    print("PART B.Q6 — MODEL TRAINING AND EVALUATION")
    print("=" * 60)

    # Reuse the same B.Q4 historical feature dataset
    # and stratified train/test split.
    dataset, X_train, X_test, y_train, y_test = (
        run_lapse_dataset()
    )

    # Confirm no missing values in any model feature.
    assert not X_train.isna().any().any()
    assert not X_test.isna().any().any()

    # Confirm the original split remains unchanged.
    assert len(X_train) == 924
    assert len(X_test) == 232

    # (a) Majority-class baseline.
    baseline_metrics = report_baseline(y_test)

    # (b) Logistic Regression tuning.
    logistic_search = tune_logistic_regression(
        X_train, y_train
    )

    # (c) Random Forest tuning.
    forest_search = tune_random_forest(
        X_train, y_train
    )

    # (c) Report BOTH best parameter sets and CV F1.
    report_tuning_results(
        logistic_search,
        forest_search,
    )

    # (d) Only now evaluate on the untouched test set.
    logistic_metrics = evaluate_model(
        "Logistic Regression",
        logistic_search.best_estimator_,
        X_test,
        y_test,
    )

    forest_metrics = evaluate_model(
        "Random Forest",
        forest_search.best_estimator_,
        X_test,
        y_test,
    )

    # Display a compact comparison.
    comparison = pd.DataFrame([
        baseline_metrics,
        logistic_metrics,
        forest_metrics,
    ])

    print("\n" + "=" * 60)
    print("B.Q6 — FINAL MODEL COMPARISON")
    print("=" * 60)

    print(
        comparison[
            ["model", "accuracy", "precision", "recall", "f1"]
        ].to_string(index=False)
    )

    # (d) Select model using training CV performance.
    selected_name, selected_model, justification = (
        select_and_justify_model(
            logistic_search,
            forest_search,
            logistic_metrics,
            forest_metrics,
        )
    )

    print("\n" + "=" * 60)
    print("B.Q6 VALIDATION SUMMARY")
    print("=" * 60)
    print("PASS | (a) Majority-class baseline reported")
    print("PASS | (b) Logistic Regression tuned with 5-fold CV")
    print("PASS | (c) Random Forest tuned with 45 CV fits")
    print("PASS | (c) Both best parameter sets and CV F1 reported")
    print("PASS | (d) Both models evaluated on test data")
    print("PASS | (d) Confusion matrices and metrics reported")
    print("PASS | (d) Model selected and justified")
    print("PASS | Preprocessing occurs inside CV pipelines")

    if (
        selected_model.predict(X_test).shape[0]
        == len(y_test)
    ):
        print("PASS | Selected model produces test predictions")

    print("\nB.Q6 EXECUTION COMPLETED")

    return {
        "dataset": dataset,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "logistic_search": logistic_search,
        "forest_search": forest_search,
        "baseline_metrics": baseline_metrics,
        "logistic_metrics": logistic_metrics,
        "forest_metrics": forest_metrics,
        "selected_name": selected_name,
        "selected_model": selected_model,
        "justification": justification,
    }


if __name__ == "__main__":
    run_lapse_model_training()
