"""Part A Q2: raw data audit."""
from pathlib import Path
import pandas as pd

from .config import RAW_DIR

def audit_table(name, df):
    """Display basic data-quality information."""

    print(f"\n===== {name} =====")

    # Number of rows and columns
    print("Shape:", df.shape)

    # Column data types
    print("\nData types:")
    print(df.dtypes)

    # Detect NaN, empty strings and whitespace-only values
    missing = df.isna() | df.astype("string").apply(
        lambda col: col.str.strip().eq("")
    ).fillna(False)
    
    # Missing-value counts and percentages
    missing_report = pd.DataFrame({
        "missing_count": missing.sum(),
        "missing_percentage": (
            missing.sum() / len(df) * 100
        ).round(2)
    })

    print("\nMissing-value report:")
    print(missing_report.round(2))


def count_blanks(series):
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )


def run_audit():
    """Print raw data checks without modifying source files."""
    customers = pd.read_csv(RAW_DIR / "customers.csv", keep_default_na=False)
    products = pd.read_csv(RAW_DIR / "products.csv", keep_default_na=False)
    orders = pd.read_csv(RAW_DIR / "orders.csv", keep_default_na=False)

    audit_table("CUSTOMERS", customers)
    audit_table("PRODUCTS", products)
    audit_table("ORDERS", orders)

    print("\n===== ORDER AUDIT =====")

    print("Raw order rows:", len(orders))

    print(
        "Exact duplicate rows:",
        orders.duplicated().sum()
    )

    print(
        "Pending orders:",
        orders["status"].eq("pending").sum()
    )

    print(
        "Distinct order IDs:",
        orders["order_id"].nunique()
    )

    # Convert order amounts temporarily for validation
    amount_numeric = pd.to_numeric(
        orders["order_amount"],
        errors="coerce"
    )

    print(
        "Non-numeric order amounts:",
        amount_numeric.isna().sum()
    )

    # Convert delivery times temporarily
    delivery_numeric = pd.to_numeric(
        orders["delivery_minutes"],
        errors="coerce"
    )

    print(
        "Negative delivery times:",
        delivery_numeric.lt(0).sum()
    )


    print("Blank delivery times:", count_blanks(orders["delivery_minutes"]))


    # A.Q2: Specific customer-data checks

    print("\n===== CUSTOMER AUDIT =====")

    # Convert age temporarily to numeric
    age_numeric = pd.to_numeric(
        customers["age"],
        errors="coerce"
    )

    print("Blank ages:", count_blanks(customers["age"]))

    print(
        "Ages outside 10 to 100:",
        ((age_numeric < 10) | (age_numeric > 100)).sum()
    )

    print(
        "Distinct raw city spellings:",
        customers["city"].nunique()
    )

    # Count non-blank gender spellings
    gender_text = customers["gender"].fillna("").astype(str)
    nonblank_gender = gender_text[gender_text.str.strip().ne("")]

    print(
        "Distinct raw gender spellings:",
        nonblank_gender.nunique()
    )

    print(
        "Blank genders:",
        count_blanks(customers["gender"])
    )

if __name__ == "__main__":
    run_audit()
