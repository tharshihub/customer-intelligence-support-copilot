
"""Part A.Q6: Execute and validate seven SQL queries."""

import re
import sqlite3

import pandas as pd

from .config import DATABASE_PATH, SQL_DIR

def load_sql_queries() -> dict[str, str]:
    """Load Q1-Q7 from sql/queries.sql."""

    sql_path = SQL_DIR / "queries.sql"
    sql_text = sql_path.read_text(encoding="utf-8")

    sections = re.split(
        r"(?m)^-- Q([1-7]):[^\n]*$",
        sql_text,
    )

    queries = {}

    for index in range(1, len(sections), 2):
        question_number = f"Q{sections[index]}"
        query = sections[index + 1].strip().rstrip(";")

        queries[question_number] = query

    assert len(queries) == 7, (
        f"Expected 7 SQL queries, found {len(queries)}"
    )

    return queries


def validate_sql_results(
    results: dict[str, pd.DataFrame],
) -> None:
    """
    Validate Q1-Q7 SQL results against all Part A.Q6 reference values.

    Raises:
        AssertionError: If an actual result does not match its reference.

    Returns:
        None. Prints the results and a success message when all pass.
    """

    def check_reference(name, actual, expected, tolerance=None):
        """Print and assert a reference checkpoint."""
        if tolerance is None:
            passed = actual == expected
        else:
            passed = abs(actual - expected) < tolerance

        print(
            f"{'PASS' if passed else 'FAIL'} | {name}: "
            f"actual={actual}, expected={expected}"
        )

        assert passed, (
            f"{name}: expected {expected}, found {actual}"
        )

    # ---------------------------------------------------------
    # Q1: WHERE followed by HAVING
    # Reference: 30 customers whose total spending on completed
    # orders above INR 500 exceeds INR 8,000.
    # ---------------------------------------------------------
    print("\nQ1 - WHERE and HAVING")

    q1 = results["Q1"]

    check_reference(
        "Qualifying customers",
        len(q1),
        30,
    )

    # ---------------------------------------------------------
    # Q2: Three-table JOIN
    # References:
    # - 30 city-category rows
    # - Total completed revenue: INR 2,998,668.80
    # - Largest cell: Bengaluru / Electronics
    # - Largest cell revenue: INR 233,909.45
    # ---------------------------------------------------------
    print("\nQ2 - Three-table JOIN")

    q2 = results["Q2"]

    check_reference("City-category rows", len(q2), 30)

    check_reference(
        "Total completed revenue",
        round(q2["revenue"].sum(), 2),
        2_998_668.80,
        tolerance=0.01,
    )

    largest = q2.loc[q2["revenue"].idxmax()]

    check_reference(
        "Highest-revenue city",
        largest["city"],
        "Bengaluru",
    )

    check_reference(
        "Highest-revenue category",
        largest["category"],
        "Electronics",
    )

    check_reference(
        "Highest city-category revenue",
        float(largest["revenue"]),
        233_909.45,
        tolerance=0.01,
    )

    # ---------------------------------------------------------
    # Q3: Scalar subquery
    # Reference: 11 products priced above the average
    # catalogue price.
    # ---------------------------------------------------------
    print("\nQ3 - Scalar subquery")

    q3 = results["Q3"]

    check_reference(
        "Products above average catalogue price",
        len(q3),
        11,
    )

    # ---------------------------------------------------------
    # Q4: LEFT JOIN
    # Reference: 25 customers have no completed orders.
    # ---------------------------------------------------------
    print("\nQ4 - LEFT JOIN")

    q4 = results["Q4"]

    check_reference(
        "Customers without completed orders",
        len(q4),
        25,
    )

    # ---------------------------------------------------------
    # Q5: Common Table Expressions (CTEs)
    # References:
    # - 550 customers above average completed spending
    # - Average customer spending: INR 2,552.06
    # ---------------------------------------------------------
    print("\nQ5 - Common Table Expressions")

    q5 = results["Q5"]

    check_reference(
        "Customers above average spend",
        len(q5),
        550,
    )

    check_reference(
        "Average customer spending",
        float(q5["avg_spend"].iloc[0]),
        2_552.06,
        tolerance=0.01,
    )

    # ---------------------------------------------------------
    # Q6: Ranking window functions
    # References:
    # - Six specified top-spending customers, one per city
    # - Their corresponding completed spending amounts
    # - Exactly 8 rows where RANK() differs from ROW_NUMBER()
    # ---------------------------------------------------------
    print("\nQ6 - Ranking window functions")

    q6 = results["Q6"]

    expected_leaders = {
        "Bengaluru": ("C0082", 12_856.20),
        "Chennai": ("C0930", 10_151.40),
        "Delhi": ("C0291", 13_742.60),
        "Hyderabad": ("C0743", 12_730.05),
        "Mumbai": ("C0693", 14_766.55),
        "Pune": ("C0426", 10_377.10),
    }

    leaders = q6.loc[q6["spend_rank"].eq(1)]

    check_reference(
        "Number of city leaders",
        len(leaders),
        6,
    )

    for city, (expected_customer, expected_spend) in (
        expected_leaders.items()
    ):
        city_leader = leaders.loc[leaders["city"].eq(city)]

        check_reference(
            f"{city} leader count",
            len(city_leader),
            1,
        )

        actual_leader = city_leader.iloc[0]

        check_reference(
            f"{city} top customer",
            actual_leader["customer_id"],
            expected_customer,
        )

        check_reference(
            f"{city} top customer spend",
            float(actual_leader["total_spend"]),
            expected_spend,
            tolerance=0.01,
        )

    # RANK() assigns equal ranks to tied spending values.
    # ROW_NUMBER() assigns a unique sequential number to each row.
    different_rank_rows = q6.loc[
        q6["spend_rank"] != q6["row_number"]
    ]

    check_reference(
        "RANK vs ROW_NUMBER differences",
        len(different_rank_rows),
        8,
    )
    #Tied customer spending values receive the same `RANK()`, whereas `ROW_NUMBER()` assigns a unique sequential position to every customer, causing the two rankings to differ for 8 rows.

    # ---------------------------------------------------------
    # Q7: Running total using a window function
    # References:
    # - 183 daily revenue rows
    # - Final cumulative revenue: INR 2,998,668.80
    # ---------------------------------------------------------
    print("\nQ7 - Daily cumulative revenue")

    q7 = results["Q7"]

    check_reference(
        "Daily revenue rows",
        len(q7),
        183,
    )

    check_reference(
        "Final cumulative revenue",
        float(q7["cumulative_revenue"].iloc[-1]),
        2_998_668.80,
        tolerance=0.01,
    )

    print("\nAll seven SQL query checkpoints passed.")


def run_sql_analysis() -> dict[str, pd.DataFrame]:
    """Execute all SQL queries and validate their results."""

    queries = load_sql_queries()
    results = {}

    with sqlite3.connect(DATABASE_PATH) as connection:
        for question_number, query in queries.items():
            dataframe = pd.read_sql_query(query, connection)
            results[question_number] = dataframe

            print(f"\n{'=' * 50}")
            print(f"{question_number}: {len(dataframe)} rows")
            print(f"{'=' * 50}")
            print(dataframe.head(5).to_string(index=False))

    validate_sql_results(results)

    return results


if __name__ == "__main__":
    run_sql_analysis()
