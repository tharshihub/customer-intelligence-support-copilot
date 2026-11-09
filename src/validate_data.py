
"""Part A.Q5: Validate cleaned CSV files and SQLite storage."""

import sqlite3

import pandas as pd

from .config import CLEAN_DIR, DATABASE_PATH


def validate_cleaned_data() -> None:
    """Validate saved CSV files against assignment checkpoints."""

    customers = pd.read_csv(CLEAN_DIR / "customers.csv")
    products = pd.read_csv(CLEAN_DIR / "products.csv")
    orders = pd.read_csv(CLEAN_DIR / "orders.csv")

    completed = orders.loc[orders["status"].eq("completed")]
    cancelled = orders.loc[orders["status"].eq("cancelled")]

    checkpoints = {
        "Customer rows": (len(customers), 1200),
        "Pune customers": (
            customers["city"].eq("Pune").sum(), 227
        ),
        "Delhi customers": (
            customers["city"].eq("Delhi").sum(), 210
        ),
        "Bengaluru customers": (
            customers["city"].eq("Bengaluru").sum(), 197
        ),
        "Female customers": (
            customers["gender"].eq("female").sum(), 610
        ),
        "Male customers": (
            customers["gender"].eq("male").sum(), 560
        ),
        "Unknown gender customers": (
            customers["gender"].eq("unknown").sum(), 30
        ),
        "Missing customer ages": (
            customers["age"].isna().sum(), 0
        ),
        "Total cleaned orders": (len(orders), 12309),
        "Unique order IDs": (orders["order_id"].nunique(), 12309),
        "Completed orders": (len(completed), 11110),
        "Cancelled orders": (len(cancelled), 1199),
        "Pending orders": (
            orders["status"].eq("pending").sum(), 0
        ),
        "Missing order amounts": (
            orders["order_amount"].isna().sum(), 0
        ),
        "Missing completed delivery times": (
            completed["delivery_minutes"].isna().sum(), 0
        ),
        "Missing cancelled delivery times": (
            cancelled["delivery_minutes"].isna().sum(), 1199
        ),
        "Negative delivery times": (
            orders["delivery_minutes"].lt(0).sum(), 0
        ),
    }

    for name, (actual, expected) in checkpoints.items():
        assert actual == expected, (
            f"{name}: expected {expected}, found {actual}"
        )

    # Validate completed order revenue
    revenue = round(completed["order_amount"].sum(), 2)

    assert abs(revenue - 2_998_668.80) < 0.01, (
        f"Completed revenue mismatch: INR {revenue:,.2f}"
    )

    # Validate order timestamp format
    timestamps = orders["order_ts"].astype(str)

    assert timestamps.str.fullmatch(
        r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}"
    ).all(), "Invalid order_ts format"

    assert pd.to_datetime(
        timestamps, format="%Y-%m-%d %H:%M:%S",
        errors="coerce"
    ).notna().all(), "Unparseable order_ts values"

    # Validate SQLite database and its tables
    with sqlite3.connect(DATABASE_PATH) as connection:
        expected_tables = {
            "customers": customers,
            "products": products,
            "orders": orders,
        }

        actual_tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type = 'table'"
            )
        }

        assert expected_tables.keys() <= actual_tables, (
            "Missing required SQLite tables"
        )

        for table_name, dataframe in expected_tables.items():
            count = connection.execute(
                f"SELECT COUNT(*) FROM {table_name}"
            ).fetchone()[0]

            assert count == len(dataframe), (
                f"{table_name}: CSV has {len(dataframe)} rows, "
                f"SQLite has {count} rows"
            )

        # SQLite must store order_ts as TEXT
        columns = connection.execute(
            "PRAGMA table_info(orders)"
        ).fetchall()

        order_ts_type = next(
            (column[2] for column in columns
             if column[1] == "order_ts"),
            None
        )

        assert order_ts_type == "TEXT", (
            f"Expected order_ts type TEXT, found {order_ts_type}"
        )

        # Check actual timestamp values in SQLite
        invalid_timestamps = connection.execute(
            """
            SELECT COUNT(*)
            FROM orders
            WHERE order_ts IS NULL
               OR strftime('%Y-%m-%d %H:%M:%S', order_ts)
                  IS NULL
               OR order_ts != strftime(
                   '%Y-%m-%d %H:%M:%S', order_ts
               )
            """
        ).fetchone()[0]

        assert invalid_timestamps == 0, (
            f"Invalid SQLite timestamps: {invalid_timestamps}"
        )

    print("ALL CHECKPOINTS PASSED")


if __name__ == "__main__":
    validate_cleaned_data()
