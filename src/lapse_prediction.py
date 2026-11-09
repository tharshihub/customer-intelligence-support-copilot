
"""Part B.Q4 — Customer lapse cohort, labels and features."""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from .config import CLEAN_DIR


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

CUTOFF = pd.Timestamp("2025-09-01 00:00:00")
LABEL_END = pd.Timestamp("2025-10-01 00:00:00")

FEATURE_COLUMNS = [
    "recency",
    "frequency",
    "monetary",
    "avg_discount_pct",
    "cancellation_rate",
    "avg_delivery_minutes",
    "favourite_category",
    "city",
    "age",
    "gender",
    "tenure_days",
]


# ------------------------------------------------------------
# B.Q4.1 — Load cleaned data
# ------------------------------------------------------------

def require_columns(df, required, table_name):
    """Check that the cleaned data has the required columns."""

    missing = set(required) - set(df.columns)

    if missing:
        raise ValueError(
            f"{table_name} missing columns: {sorted(missing)}"
        )


def load_lapse_data():
    """Load cleaned customers, orders and products."""

    customers = pd.read_csv(CLEAN_DIR / "customers.csv")
    orders = pd.read_csv(CLEAN_DIR / "orders.csv")
    products = pd.read_csv(CLEAN_DIR / "products.csv")

    require_columns(
        customers,
        [
            "customer_id",
            "signup_date",
            "city",
            "age",
            "gender",
        ],
        "customers",
    )

    require_columns(
        orders,
        [
            "customer_id",
            "order_id",
            "order_ts",
            "status",
            "order_amount",
            "discount_pct",
            "delivery_minutes",
            "product_id",
        ],
        "orders",
    )

    require_columns(
        products,
        ["product_id", "category"],
        "products",
    )

    customers["signup_date"] = pd.to_datetime(
        customers["signup_date"],
        errors="raise",
    )

    orders["order_ts"] = pd.to_datetime(
        orders["order_ts"],
        errors="raise",
    )

    assert customers["customer_id"].is_unique
    assert orders["order_id"].is_unique
    assert products["product_id"].is_unique

    return customers, orders, products


# ------------------------------------------------------------
# B.Q4.2 — Build cohort, labels and historical features
# ------------------------------------------------------------

def build_lapse_dataset(customers, orders, products):
    """Build features before cutoff and labels from September."""

    # Only orders before September 1 can generate features.
    historical = orders.loc[
        orders["order_ts"] < CUTOFF
    ].copy()

    completed = historical.loc[
        historical["status"].eq("completed")
    ].copy()

    assert (historical["order_ts"] < CUTOFF).all()
    assert (completed["order_ts"] < CUTOFF).all()

    # Cohort: customers with >= 1 completed historical order.
    cohort_ids = set(completed["customer_id"])

    # Target label: completed orders during September.
    # These orders are used ONLY for the label.
    september_completed_ids = set(
        orders.loc[
            (orders["order_ts"] >= CUTOFF)
            & (orders["order_ts"] < LABEL_END)
            & (orders["status"] == "completed"),
            "customer_id",
        ]
    )

    # --------------------------------------------------------
    # Historical RFM and completed-order averages
    # --------------------------------------------------------

    features = (
        completed.groupby("customer_id")
        .agg(
            last_order=("order_ts", "max"),
            frequency=("order_id", "nunique"),
            monetary=("order_amount", "sum"),
            avg_discount_pct=("discount_pct", "mean"),
            avg_delivery_minutes=("delivery_minutes", "mean"),
        )
        .reset_index()
    )

    features["recency"] = (
        CUTOFF - features["last_order"]
    ).dt.days

    features = features.drop(columns="last_order")

    # --------------------------------------------------------
    # Cancellation rate from final-status historical orders
    # --------------------------------------------------------

    final_orders = historical.loc[
        historical["status"].isin(
            ["completed", "cancelled"]
        )
    ].copy()

    final_orders["is_cancelled"] = (
        final_orders["status"] == "cancelled"
    ).astype(int)

    cancellation = (
        final_orders.groupby("customer_id")
        .agg(
            cancellation_rate=("is_cancelled", "mean")
        )
        .reset_index()
    )

    features = features.merge(
        cancellation,
        on="customer_id",
        how="left",
        validate="one_to_one",
    )

    # --------------------------------------------------------
    # Favourite category from historical completed orders
    # --------------------------------------------------------

    categorized = completed[
        ["customer_id", "product_id"]
    ].merge(
        products[["product_id", "category"]],
        on="product_id",
        how="left",
        validate="many_to_one",
    )

    assert categorized["category"].notna().all(), (
        "Missing category for a completed order"
    )

    category_counts = (
        categorized.groupby(
            ["customer_id", "category"]
        )
        .size()
        .reset_index(name="category_count")
    )

    # Highest count wins; alphabetical tie-break.
    favourite = (
        category_counts.sort_values(
            [
                "customer_id",
                "category_count",
                "category",
            ],
            ascending=[True, False, True],
        )
        .drop_duplicates("customer_id")
        [["customer_id", "category"]]
        .rename(
            columns={"category": "favourite_category"}
        )
    )

    features = features.merge(
        favourite,
        on="customer_id",
        how="left",
        validate="one_to_one",
    )

    # --------------------------------------------------------
    # Customer attributes and tenure at cutoff
    # --------------------------------------------------------

    customer_info = customers[
        [
            "customer_id",
            "city",
            "age",
            "gender",
            "signup_date",
        ]
    ].copy()

    customer_info["tenure_days"] = (
        CUTOFF - customer_info["signup_date"]
    ).dt.days

    customer_info = customer_info.drop(
        columns="signup_date"
    )

    features = features.merge(
        customer_info,
        on="customer_id",
        how="left",
        validate="one_to_one",
    )

    # --------------------------------------------------------
    # Target: 1 = lapsed, 0 = not lapsed
    # --------------------------------------------------------

    features["lapsed"] = (
        ~features["customer_id"].isin(
            september_completed_ids
        )
    ).astype(int)

    assert set(features["customer_id"]) == cohort_ids

    return features[
        ["customer_id"] + FEATURE_COLUMNS + ["lapsed"]
    ]


# ------------------------------------------------------------
# B.Q4.3 — Validate cohort, features and labels
# ------------------------------------------------------------

def validate_lapse_dataset(dataset):
    """Validate assignment reference values and leakage rules."""

    assert dataset["customer_id"].is_unique

    assert len(dataset) == 1156, (
        f"Expected 1156 customers, got {len(dataset)}"
    )

    assert dataset["lapsed"].sum() == 443, (
        "Expected 443 lapsed customers, got "
        f"{dataset['lapsed'].sum()}"
    )

    assert abs(dataset["lapsed"].mean() - 0.3832) < 0.001

    assert len(FEATURE_COLUMNS) == 11

    assert all(
        col in dataset.columns
        for col in FEATURE_COLUMNS
    )

    assert not dataset.isna().any().any(), (
        "Lapse dataset contains missing values"
    )

    assert dataset["lapsed"].isin([0, 1]).all()

    assert (dataset["recency"] >= 0).all()
    assert (dataset["frequency"] >= 1).all()
    assert (dataset["monetary"] >= 0).all()
    assert dataset["cancellation_rate"].between(0, 1).all()
    assert (dataset["tenure_days"] >= 0).all()

    # B.Q3 segments must never become model features.
    forbidden = {
        "segment_id",
        "segment_name",
        "is_premium",
    }

    assert not forbidden.intersection(dataset.columns)

    print("PASS | Cohort: 1156 customers")
    print("PASS | Lapsed: 443 customers")
    print("PASS | All 11 required features")
    print("PASS | No missing values")
    print("PASS | No B.Q3 segment leakage")


# ------------------------------------------------------------
# B.Q4.4 — Stratified train/test split
# ------------------------------------------------------------

def split_lapse_dataset(dataset):
    """Create the required 80/20 stratified split."""

    X = dataset[FEATURE_COLUMNS].copy()
    y = dataset["lapsed"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    assert len(X_train) == 924
    assert len(X_test) == 232

    assert set(X_train.index).isdisjoint(
        set(X_test.index)
    )

    forbidden = {
        "segment_id",
        "segment_name",
        "is_premium",
        "lapsed",
    }

    assert not forbidden.intersection(X_train.columns)
    assert not forbidden.intersection(X_test.columns)

    print("PASS | Stratified 80/20 train/test split")

    return X_train, X_test, y_train, y_test


# ------------------------------------------------------------
# B.Q4.5 — Always-not-lapsed baseline
# ------------------------------------------------------------

def evaluate_baseline(y_test):
    """Evaluate predicting 'not lapsed' for every customer."""

    predictions = np.zeros(
        len(y_test),
        dtype=int,
    )

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    print("\n=== Majority-Class Baseline ===")
    print(f"Accuracy:  {accuracy:.1%}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"Lapsed F1: {f1:.3f}")

    assert abs(accuracy - 0.616) <= 0.01
    assert f1 == 0.0

    print("PASS | Always-not-lapsed baseline")

    return accuracy, precision, recall, f1


# ------------------------------------------------------------
# B.Q4 — Main runner
# ------------------------------------------------------------

def run_lapse_dataset():
    """Run all B.Q4 steps and return data for B.Q5."""

    customers, orders, products = load_lapse_data()

    dataset = build_lapse_dataset(
        customers,
        orders,
        products,
    )

    validate_lapse_dataset(dataset)

    X_train, X_test, y_train, y_test = (
        split_lapse_dataset(dataset)
    )

    evaluate_baseline(y_test)

    print("\n=== B.Q4: Lapse Dataset Summary ===")
    print(f"Cohort size: {len(dataset)}")
    print(f"Lapsed: {dataset['lapsed'].sum()}")
    print(f"Lapse rate: {dataset['lapsed'].mean():.2%}")
    print(f"Training rows: {len(X_train)}")
    print(f"Test rows: {len(X_test)}")
    print(f"Features: {len(FEATURE_COLUMNS)}")

    print("\nB.Q4 ALL CHECKPOINTS PASSED")

    return dataset, X_train, X_test, y_train, y_test


if __name__ == "__main__":
    run_lapse_dataset()
