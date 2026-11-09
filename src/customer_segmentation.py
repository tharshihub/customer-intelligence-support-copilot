
"""Part B: Customer segmentation using RFM analysis."""

import pandas as pd
from sklearn.decomposition import PCA

from .config import CLEAN_DIR, OUTPUT_DIR, ensure_directories

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

SNAPSHOT_DATE = pd.Timestamp("2025-10-01 00:00:00")


def build_rfm_table(orders: pd.DataFrame) -> pd.DataFrame:
    """Calculate RFM metrics for customers with completed orders."""

    completed = orders.loc[orders["status"].eq("completed")].copy()

    completed["order_ts"] = pd.to_datetime(
        completed["order_ts"], errors="raise"
    )

    # Aggregate completed-order behaviour per customer.
    rfm = (
        completed.groupby("customer_id")
        .agg(
            last_order_ts=("order_ts", "max"),
            frequency=("order_id", "nunique"),
            monetary=("order_amount", "sum"),
        )
        .reset_index()
    )

    # Recency is measured in whole days from the fixed snapshot.
    rfm["recency"] = (
        SNAPSHOT_DATE - rfm["last_order_ts"]
    ).dt.days

    rfm = rfm[
        ["customer_id", "recency", "frequency", "monetary"]
    ].copy()

    return rfm


def validate_rfm_table(rfm: pd.DataFrame) -> None:
    """Validate RFM results against assignment checkpoints."""

    assert len(rfm) == 1175, "Unexpected customer count"
    assert rfm["customer_id"].is_unique
    assert rfm["recency"].median() == 19
    assert rfm["recency"].min() == 0
    assert rfm["recency"].max() == 180
    assert round(rfm["frequency"].mean(), 2) == 9.46
    assert rfm["frequency"].max() == 33
    assert round(rfm["monetary"].mean(), 2) == 2552.06
    assert round(rfm["monetary"].max(), 2) == 14766.55

    print("B.Q1 ALL CHECKPOINTS PASSED")


def run_rfm_analysis() -> pd.DataFrame:
    """Load cleaned orders, calculate RFM and validate results."""

    orders = pd.read_csv(CLEAN_DIR / "orders.csv")

    rfm = build_rfm_table(orders)

    print("\n=== B.Q1: RFM Customer Analysis ===")
    print(f"Customers: {len(rfm)}")
    print(f"Median recency: {rfm['recency'].median():.0f} days")
    print(f"Recency range: {rfm['recency'].min()}–{rfm['recency'].max()} days")
    print(f"Mean frequency: {rfm['frequency'].mean():.2f}")
    print(f"Maximum frequency: {rfm['frequency'].max()}")
    print(f"Mean monetary: INR {rfm['monetary'].mean():,.2f}")
    print(f"Maximum monetary: INR {rfm['monetary'].max():,.2f}")

    validate_rfm_table(rfm)

    return rfm

# Part B.Q2 — K-means clustering

RFM_FEATURES = ["recency", "frequency", "monetary"]

KMEANS_REFERENCE = {
    2: (1820.8, 0.465),
    3: (1268.2, 0.445),
    4: (743.1, 0.526),
    5: (617.4, 0.481),
    6: (503.2, 0.482),
    7: (408.5, 0.416),
    8: (366.5, 0.412),
}


def evaluate_kmeans(rfm: pd.DataFrame) -> tuple:
    """Standardize RFM and evaluate K=2 through K=8."""

    features = rfm[RFM_FEATURES].copy()

    assert not features.isna().any().any(), (
        "RFM features contain missing values"
    )

    scaler = StandardScaler()
    scaled_rfm = scaler.fit_transform(features)

    results = []

    for k in range(2, 9):
        model = KMeans(
            n_clusters=k,
            n_init=10,
            random_state=42,
        )

        labels = model.fit_predict(scaled_rfm)

        results.append(
            {
                "k": k,
                "inertia": model.inertia_,
                "silhouette": silhouette_score(
                    scaled_rfm, labels
                ),
            }
        )

    metrics = pd.DataFrame(results)

    return metrics, scaled_rfm, scaler


def validate_kmeans(metrics: pd.DataFrame) -> None:
    """Check clustering metrics against reference tolerances."""

    assert len(metrics) == 7
    assert metrics["k"].tolist() == list(range(2, 9))

    for row in metrics.itertuples(index=False):
        expected_inertia, expected_silhouette = (
            KMEANS_REFERENCE[row.k]
        )

        inertia_error = (
            abs(row.inertia - expected_inertia)
            / expected_inertia
        )

        silhouette_error = abs(
            row.silhouette - expected_silhouette
        )

        assert inertia_error <= 0.05, (
            f"K={row.k}: inertia differs by "
            f"{inertia_error:.2%}"
        )

        assert silhouette_error <= 0.05, (
            f"K={row.k}: silhouette mismatch"
        )

    print("B.Q2 ALL CHECKPOINTS PASSED")


def run_kmeans_analysis(rfm: pd.DataFrame) -> tuple:
    """Evaluate clustering and choose the final K."""

    metrics, scaled_rfm, scaler = evaluate_kmeans(rfm)

    print("\n=== B.Q2: K-means Evaluation ===")
    print(metrics.round(3).to_string(index=False))

    validate_kmeans(metrics)

    # The assignment requires the final K to be
    # chosen from {3, 4, 5}.
    candidates = metrics.loc[
        metrics["k"].isin([3, 4, 5])
    ]

    best_k = int(
        candidates.loc[
            candidates["silhouette"].idxmax(), "k"
        ]
    )

    print(f"\nRecommended K: {best_k}")

    # Fit the final model for use in B.Q3.
    final_model = KMeans(
        n_clusters=best_k,
        n_init=10,
        random_state=42,
    )

    final_model.fit(scaled_rfm)

    return metrics, final_model, scaler


# ============================================================
# PART B.Q3 — SEGMENT PROFILES, BUSINESS NAMES AND PCA
# ============================================================

SEGMENT_COLUMNS = [
    "customer_id",
    "recency",
    "frequency",
    "monetary",
    "segment_id",
    "segment_name",
    "is_premium",
]

PCA_TOLERANCE = 0.02


def profile_segments(
    rfm: pd.DataFrame,
    model: KMeans,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign customers to clusters and calculate RFM profiles."""

    segmented = rfm.copy()

    segmented["segment_id"] = model.labels_

    profile = (
        segmented.groupby("segment_id")
        .agg(
            customers=("customer_id", "count"),
            mean_recency=("recency", "mean"),
            mean_frequency=("frequency", "mean"),
            mean_monetary=("monetary", "mean"),
        )
        .reset_index()
        .sort_values("segment_id")
    )

    return segmented, profile


def name_segments(
    segmented: pd.DataFrame,
    profile: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign meaningful names based on RFM characteristics."""

    profile = profile.copy()

    assert len(profile) == 4, (
        "B.Q3 naming expects K=4"
    )

    # Premium: highest average monetary value.
    premium_id = int(
        profile.loc[
            profile["mean_monetary"].idxmax(),
            "segment_id",
        ]
    )

    remaining = profile.loc[
        profile["segment_id"] != premium_id
    ]

    # Frequent Loyal: highest frequency among remaining.
    loyal_id = int(
        remaining.loc[
            remaining["mean_frequency"].idxmax(),
            "segment_id",
        ]
    )

    remaining = remaining.loc[
        remaining["segment_id"] != loyal_id
    ]

    # At Risk: longest recency among remaining.
    at_risk_id = int(
        remaining.loc[
            remaining["mean_recency"].idxmax(),
            "segment_id",
        ]
    )

    regular_ids = (
        set(profile["segment_id"].astype(int))
        - {premium_id, loyal_id, at_risk_id}
    )

    assert len(regular_ids) == 1
    regular_id = regular_ids.pop()

    names = {
        premium_id: "Premium",
        loyal_id: "Frequent Loyal",
        at_risk_id: "At Risk",
        regular_id: "Regular",
    }

    actions = {
        premium_id: (
            "VIP service and exclusive retention offers"
        ),
        loyal_id: (
            "Loyalty rewards and referral incentives"
        ),
        at_risk_id: (
            "Targeted reactivation campaigns"
        ),
        regular_id: (
            "Personalized recommendations and repeat-order offers"
        ),
    }

    profile["segment_name"] = profile[
        "segment_id"
    ].map(names)

    profile["recommended_action"] = profile[
        "segment_id"
    ].map(actions)

    segmented["segment_name"] = segmented[
        "segment_id"
    ].map(names)

    segmented["is_premium"] = (
        segmented["segment_id"] == premium_id
    )

    return segmented, profile


def analyze_pca(
    scaled_rfm,
) -> tuple[PCA, pd.DataFrame]:
    """Calculate two PCA components from scaled RFM features."""

    pca = PCA(n_components=2)

    pca.fit(scaled_rfm)

    summary = pd.DataFrame(
        {
            "component": ["PC1", "PC2"],
            "explained_variance_ratio": (
                pca.explained_variance_ratio_
            ),
        }
    )

    return pca, summary


def validate_pca(pca: PCA) -> None:
    """Validate PCA against the assignment's ±0.02 tolerance."""

    actual = pca.explained_variance_ratio_
    expected = [0.634, 0.218]

    for index, (value, reference) in enumerate(
        zip(actual, expected),
        start=1,
    ):
        assert abs(value - reference) <= PCA_TOLERANCE, (
            f"PC{index}: expected {reference:.3f} "
            f"±{PCA_TOLERANCE:.2f}, got {value:.4f}"
        )

        print(f"PASS | PC{index}: {value:.4f}")

    total = actual.sum()

    assert abs(total - 0.852) <= PCA_TOLERANCE, (
        "Total explained variance mismatch: "
        f"expected 0.852 ±0.02, got {total:.4f}"
    )

    print(f"PASS | PCA total: {total:.4f}")


def validate_segment_results(
    segmented: pd.DataFrame,
    profile: pd.DataFrame,
    pca: PCA,
) -> None:
    """Validate customer segments, reference profiles and PCA."""

    # 1. Customer count and segment count
    assert len(segmented) == 1175
    assert segmented["customer_id"].is_unique
    assert segmented["segment_id"].nunique() == 4

    # 2. Required columns and missing values
    assert all(
        column in segmented.columns
        for column in SEGMENT_COLUMNS
    )

    assert not segmented[
        SEGMENT_COLUMNS
    ].isna().any().any()

    print("PASS | Customer count and required columns")

    # 3. Approximate segment sizes
    actual_sizes = sorted(
        profile["customers"].tolist()
    )

    expected_sizes = sorted(
        [436, 138, 419, 182]
    )

    # The assignment says approximately, so ±5
    # is our suggested tolerance, not an official one.
    assert all(
        abs(actual - expected) <= 5
        for actual, expected in zip(
            actual_sizes,
            expected_sizes,
        )
    ), (
        f"Segment sizes differ: {actual_sizes}"
    )

    print("PASS | Approximate segment sizes")

    # 4. Premium segment reference checks
    premium = profile.loc[
        profile["mean_monetary"].idxmax()
    ]

    assert abs(premium["customers"] - 138) <= 5
    assert abs(premium["mean_monetary"] - 7130) <= 150
    assert abs(premium["mean_frequency"] - 6.8) <= 0.5

    assert premium["segment_name"] == "Premium"

    premium_id = premium["segment_id"]

    assert (
        segmented["is_premium"].sum()
        == premium["customers"]
    )

    assert (
        segmented.loc[
            segmented["is_premium"],
            "segment_id",
        ] == premium_id
    ).all()

    print("PASS | Premium segment reference profile")

    # 5. Frequent Loyal segment reference checks
    frequent = profile.loc[
        profile["mean_frequency"].idxmax()
    ]

    assert abs(frequent["customers"] - 419) <= 5
    assert abs(frequent["mean_frequency"] - 19.4) <= 0.5
    assert abs(frequent["mean_recency"] - 8.8) <= 1.0

    assert frequent["segment_name"] == "Frequent Loyal"

    print("PASS | Frequent Loyal reference profile")

    # 6. PCA explained variance (official ±0.02 tolerance)
    validate_pca(pca)

    print("\nB.Q3 ALL CHECKPOINTS PASSED")


def run_segment_profiling(
    rfm: pd.DataFrame,
    model: KMeans,
    scaled_rfm,
) -> pd.DataFrame:
    """Run B.Q3 and export the customer segment table."""

    segmented, profile = profile_segments(
        rfm,
        model,
    )

    segmented, profile = name_segments(
        segmented,
        profile,
    )

    pca, pca_summary = analyze_pca(
        scaled_rfm
    )

    print("\n=== B.Q3: Segment Profiles ===")

    print(
        profile.round(2).to_string(index=False)
    )

    print("\n=== PCA Explained Variance ===")

    print(
        pca_summary.round(4).to_string(index=False)
    )

    validate_segment_results(
        segmented,
        profile,
        pca,
    )

    ensure_directories()

    output_path = (
        OUTPUT_DIR / "customer_segments.csv"
    )

    segmented[SEGMENT_COLUMNS].to_csv(
        output_path,
        index=False,
    )

    print(f"\nSaved: {output_path}")

    return segmented

def run_customer_segmentation() -> pd.DataFrame:
    """Run the full customer segmentation analysis."""

    rfm = run_rfm_analysis()
    metrics, model, scaler = run_kmeans_analysis(rfm)

    scaled_rfm = scaler.transform(
        rfm[RFM_FEATURES]
    )

    segmented = run_segment_profiling(
        rfm,
        model,
        scaled_rfm,
    )

    return segmented

if __name__ == "__main__":
    run_customer_segmentation()
    # rfm = run_rfm_analysis()
    # metrics, model, scaler = run_kmeans_analysis(rfm)

    # scaled_rfm = scaler.transform(
    #     rfm[RFM_FEATURES]
    # )

    # run_segment_profiling(
    #     rfm,
    #     model,
    #     scaled_rfm,
    # )