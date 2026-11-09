"""Part A.Q7: Pandas cross-check and exploratory data analysis."""

import sqlite3

import pandas as pd

from .config import CLEAN_DIR, DATABASE_PATH


def check_reference(
    name: str,
    actual,
    expected,
    tolerance: float | None = None,
) -> None:
    """Validate a result against its assignment reference."""
    passed = (
        actual == expected
        if tolerance is None
        else abs(actual - expected) < tolerance
    )

    print(
        f"{'PASS' if passed else 'FAIL'} | {name}: "
        f"actual={actual}, expected={expected}"
    )

    assert passed, (
        f"{name}: expected {expected}, found {actual}"
    )


def run_pandas_cross_check(
    customers: pd.DataFrame,
    products: pd.DataFrame,
    orders: pd.DataFrame,
) -> None:
    """
    A.Q7(a): Reproduce SQL Q2 using pandas.

    Validates:
        - Pivot-table grand total against SQL Q2.
        - Bengaluru completed revenue, mean and order count.
        - Mumbai completed revenue, mean and order count.
    """
    print("\nA.Q7(a) - Pandas cross-check")

   # Select only required columns to prevent duplicate column names.
    completed = orders.loc[
        orders["status"].eq("completed"),
        ["order_id", "customer_id", "product_id", "order_amount"]
    ].copy()
    
    # Join customers to obtain city and products to obtain category.
    joined = (
        completed
        .merge(
            customers[["customer_id", "city"]],
            on="customer_id",
            how="inner",
            validate="many_to_one",
        )
        .merge(
            products[["product_id", "category"]],
            on="product_id",
            how="inner",
            validate="many_to_one",
        )
    )

    # Validate that no completed orders were lost during joining.
    check_reference(
        "Joined completed orders",
        len(joined),
        len(completed),
    )

    # Validate that the required columns exist.
    required_columns = {
        "order_id",
        "customer_id",
        "product_id",
        "order_amount",
        "city",
        "category",
    }

    assert required_columns.issubset(joined.columns), (
        f"Missing columns: {required_columns - set(joined.columns)}"
    )

    print("\nJoined columns:", joined.columns.tolist())
    # Rebuild SQL Q2 using a pivot table.
    # margins=True adds the grand total row and column.
    revenue_pivot = pd.pivot_table(
        joined,
        values="order_amount",
        index="city",
        columns="category",
        aggfunc="sum",
        fill_value=0,
        margins=True,
        margins_name="All",
    )

    print("\nRevenue by city and category (INR):")
    print(revenue_pivot.round(2).to_string())

    pandas_total = float(revenue_pivot.loc["All", "All"])

    # Execute SQL Q2 directly for an independent cross-check.
    sql_q2 = """
        SELECT
            c.city,
            p.category,
            SUM(o.order_amount) AS revenue
        FROM orders AS o
        JOIN customers AS c
            ON o.customer_id = c.customer_id
        JOIN products AS p
            ON o.product_id = p.product_id
        WHERE o.status = 'completed'
        GROUP BY c.city, p.category
    """

    with sqlite3.connect(DATABASE_PATH) as connection:
        sql_result = pd.read_sql_query(
            sql_q2, connection
        )

    sql_total = float(sql_result["revenue"].sum())

    check_reference(
        "Pandas vs SQL revenue",
        pandas_total,
        sql_total,
        tolerance=0.01,
    )

    check_reference(
        "Total completed revenue (INR)",
        round(pandas_total, 2),
        2_998_668.80,
        tolerance=0.01,
    )

    # Named aggregation: sum, mean and count by city.
    city_summary = (
        joined.groupby("city")
        .agg(
            total_revenue=("order_amount", "sum"),
            mean_order_amount=("order_amount", "mean"),
            completed_orders=("order_amount", "count"),
        )
        .sort_index()
    )

    print("\nCity-level completed order summary:")
    print(city_summary.round(2).to_string())

    # Reference checkpoints from the assignment.
    city_references = {
        "Bengaluru": (523_643.40, 276.47, 1894),
        "Mumbai": (479_058.50, 297.00, 1613),
    }

    for city, (revenue, mean, count) in (
        city_references.items()
    ):
        actual = city_summary.loc[city]

        check_reference(
            f"{city} revenue",
            float(actual["total_revenue"]),
            revenue,
            tolerance=0.01,
        )

        check_reference(
            f"{city} mean",
            round(float(actual["mean_order_amount"]), 2),
            mean,
            tolerance=0.01,
        )

        check_reference(
            f"{city} completed orders",
            int(actual["completed_orders"]),
            count,
        )

def run_exploratory_analysis(
    orders: pd.DataFrame,
) -> None:
    """
    A.Q7(b): Analyse completed order amounts.

    Calculates:
        - Mean, median, skewness and quartiles.
        - IQR fences and outlier counts.
        - Correlations among the four required variables.
    """
    print("\nA.Q7(b) - Exploratory Data Analysis")

    completed = orders.loc[
        orders["status"].eq("completed")
    ].copy()

    amounts = completed["order_amount"]

    # Descriptive statistics
    mean = amounts.mean()
    median = amounts.median()
    skewness = amounts.skew()

    q1 = amounts.quantile(0.25)
    q3 = amounts.quantile(0.75)

    # IQR method for detecting outliers.
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr

    lower_outliers = int((amounts < lower_fence).sum())
    upper_outliers = int((amounts > upper_fence).sum())

    statistics = pd.Series({
        "Mean (INR)": mean,
        "Median (INR)": median,
        "Skewness": skewness,
        "Q1 (INR)": q1,
        "Q3 (INR)": q3,
        "IQR (INR)": iqr,
        "Lower fence (INR)": lower_fence,
        "Upper fence (INR)": upper_fence,
        "Lower outliers": lower_outliers,
        "Upper outliers": upper_outliers,
    })

    print("\nCompleted order amount statistics:")
    print(statistics.round(2).to_string())

    # Validate the assignment's EDA reference values.
    check_reference(
        "Mean", round(mean, 2), 269.91,
        tolerance=0.01,
    )
    check_reference(
        "Median", round(median, 2), 135.00,
        tolerance=0.01,
    )
    check_reference(
        "Skewness", round(skewness, 2), 4.61,
        tolerance=0.05,
    )
    check_reference(
        "Q1", round(q1, 2), 64.80,
        tolerance=0.01,
    )
    check_reference(
        "Q3", round(q3, 2), 297.00,
        tolerance=0.01,
    )
    check_reference(
        "Upper fence", round(upper_fence, 2), 645.30,
        tolerance=0.01,
    )
    check_reference(
        "Lower outliers", lower_outliers, 0,
    )
    check_reference(
        "Upper outliers", upper_outliers, 897,
    )

    # Correlation matrix over completed orders only.
    columns = [
        "order_amount",
        "discount_pct",
        "quantity",
        "delivery_minutes",
    ]

    correlation_matrix = completed[columns].corr()

    print("\nCorrelation matrix:")
    print(correlation_matrix.round(3).to_string())

    # Find the strongest absolute correlation between
    # two different variables, excluding the diagonal.
    pairs = []

    for i, first in enumerate(columns):
        for second in columns[i + 1:]:
            correlation = correlation_matrix.loc[first, second]

            if pd.notna(correlation):
                pairs.append((first, second, float(correlation)))

    strongest = max(
        pairs,
        key=lambda pair: abs(pair[2]),
    )

    first, second, value = strongest

    print(
        f"\nStrongest absolute correlation: "
        f"{first} vs {second} = {value:.3f}"
    )

    check_reference(
        "Strongest correlation pair",
        {first, second},
        {"order_amount", "discount_pct"},
    )

    check_reference(
        "Strongest correlation value",
        value,
        -0.095,
        tolerance=0.01,
    )

    check_reference(
        "Order amount vs quantity correlation",
        float(
            correlation_matrix.loc[
                "order_amount", "quantity"
            ]
        ),
        0.087,
        tolerance=0.01,
    )


def run_pandas_analysis() -> None:
    """Execute and validate both parts of A.Q7."""

    customers = pd.read_csv(
        CLEAN_DIR / "customers.csv"
    )
    products = pd.read_csv(
        CLEAN_DIR / "products.csv"
    )
    orders = pd.read_csv(
        CLEAN_DIR / "orders.csv"
    )

    run_pandas_cross_check(customers, products, orders)
    run_exploratory_analysis(orders)

    print("\nA.Q7 ALL CHECKPOINTS PASSED")


if __name__ == "__main__":
    run_pandas_analysis()
