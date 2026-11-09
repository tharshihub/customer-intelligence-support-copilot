
"""Part A: Customer and order data cleaning."""

from datetime import date

import pandas as pd
import numpy as np

from .config import RAW_DIR, CLEAN_DIR, DATABASE_PATH,  ensure_directories

import sqlite3

def clean_customers(customers: pd.DataFrame) -> pd.DataFrame:
    """Clean customer data according to assignment A.Q3."""

    cleaned = customers.copy()
    # A.Q3(a): Standardise city names
    cleaned["city"] = (
        cleaned["city"].astype(str).str.strip().str.title()
    )

    # A.Q3(b): Standardise gender
    gender = (
        cleaned["gender"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    gender_mapping = {
        "m": "male",
        "male": "male",
        "f": "female",
        "female": "female",
    }

    cleaned["gender"] = (
        gender.map(gender_mapping).fillna("unknown")
    )

    # A.Q3(c): Convert age to numeric
    cleaned["age"] = pd.to_numeric(
        cleaned["age"],
        errors="coerce",
    )

    # Treat ages outside 10-100 as missing
    cleaned.loc[
        ~cleaned["age"].between(10, 100),
        "age",
    ] = np.nan

    # Calculate median age for each city
    city_median_age = (
        cleaned.groupby("city")["age"].transform("median")
    )
    # Fill missing ages using their city's median
    cleaned["age"] = cleaned["age"].fillna(
        city_median_age
    ).round().astype("Int64")

    # A.Q3(d): Convert signup date to datetime
    cleaned["signup_date"] = pd.to_datetime(
        cleaned["signup_date"],
        errors="raise",
    )
    return cleaned


def print_customer_summary(cleaned: pd.DataFrame) -> None:
    """Display A.Q3 customer-cleaning results."""

    print("\n===== A.Q3 CUSTOMER CLEANING =====")

    # A.Q3(a): City checks
    print("\n--- CITY ---")
    print("Unique cities:", cleaned["city"].nunique())
    print("City distribution:")
    print(cleaned["city"].value_counts())

    # A.Q3(b): Gender checks
    print("\n--- GENDER ---")
    print("Unique genders:", cleaned["gender"].nunique())
    print("Gender distribution:")
    print(cleaned["gender"].value_counts())

    # A.Q3(c): Age checks
    print("\n--- AGE ---")
    print("Missing ages:", cleaned["age"].isna().sum())
    print("Minimum age:", cleaned["age"].min())
    print("Maximum age:", cleaned["age"].max())
    print(
        "Ages outside 18–60:",
        (~cleaned["age"].between(18, 60)).sum()
    )

    # A.Q3(d): Signup date checks
    print("\n--- SIGNUP DATE ---")
    print("Missing signup dates:", cleaned["signup_date"].isna().sum())
    print("Earliest signup:", cleaned["signup_date"].min())
    print("Latest signup:", cleaned["signup_date"].max())
    print("Signup date dtype:", cleaned["signup_date"].dtype)

    # Overall checks
    print("\n--- DATASET SUMMARY ---")
    print("Total customers:", len(cleaned))
    print("Duplicate customer IDs:", cleaned["customer_id"].duplicated().sum())
    print("\nData types:")
    print(cleaned.dtypes)


def validate_customers(cleaned: pd.DataFrame) -> None:
    """Validate required A.Q3 reference checkpoints."""

    expected_cities = {
        "Pune": 227,
        "Delhi": 210,
        "Bengaluru": 197,
        "Chennai": 191,
        "Hyderabad": 191,
        "Mumbai": 184,
    }

    expected_genders = {
        "female": 610,
        "male": 560,
        "unknown": 30,
    }

    assert cleaned["city"].value_counts().to_dict() == expected_cities
    assert cleaned["gender"].value_counts().to_dict() == expected_genders
    assert cleaned["age"].notna().all()
    assert cleaned["age"].between(18, 60).all()
    assert pd.api.types.is_datetime64_any_dtype(
        cleaned["signup_date"]
    )

    print("A.Q3 validation passed.")


def run_customer_cleaning() -> pd.DataFrame:
    """Load raw customers, clean, validate and save."""

    customers = pd.read_csv(
        RAW_DIR / "customers.csv",
        keep_default_na=False,
    )

    cleaned = clean_customers(customers)
    print_customer_summary(cleaned)
    validate_customers(cleaned)

    ensure_directories()

    output_path = CLEAN_DIR / "customers.csv"
    cleaned.to_csv(output_path, index=False)

    print(f"Saved cleaned customers to {output_path}")

    return cleaned


def clean_orders(
    orders: pd.DataFrame,
    products: pd.DataFrame,
) -> pd.DataFrame:
    """Clean orders according to assignment A.Q4."""

    cleaned = orders.copy()

    # A.Q4(i): Convert order timestamp to datetime
    cleaned["order_ts"] = pd.to_datetime(
        cleaned["order_ts"],
        errors="raise",
    )

    
    # A.Q4(ii): Remove exact duplicate rows
    duplicate_count = cleaned.duplicated().sum()
    print("Raw orders:", len(cleaned))
    print("A.Q4(ii) — Exact duplicates:", duplicate_count)
    cleaned = cleaned.drop_duplicates().copy()
    print("After duplicate removal:", len(cleaned))

    validate_order_duplicates(cleaned)

    # A.Q4(iii): Keep latest record per order ID
    print("A.Q4(iii) — Duplicate order IDs:", cleaned.duplicated("order_id").sum())
    cleaned = (
        cleaned
        .sort_values("order_ts", kind="stable")
        .drop_duplicates(subset=["order_id"], keep="last")
        .reset_index(drop=True)
    )
    print("After duplicate Order Id removal:", len(cleaned))
    print("Final unique orders:", cleaned["order_id"].nunique())
    validate_latest_orders(cleaned)

    # A.Q4(iv): Repair missing order amounts
    # Convert invalid amounts to NaN
    cleaned["order_amount"] = pd.to_numeric(
        cleaned["order_amount"],
        errors="coerce",
    )
    # Convert calculation inputs to numeric
    cleaned["quantity"] = pd.to_numeric(
        cleaned["quantity"], errors="coerce"
    )

    cleaned["discount_pct"] = pd.to_numeric(
        cleaned["discount_pct"], errors="coerce"
    )

    # Validate product prices without modifying the original DataFrame
    price_by_product = pd.to_numeric(
        products.set_index("product_id")["price_inr"],
        errors="coerce"
    )

    validate_product_prices(cleaned, price_by_product)

    # Identify rows with missing order amounts
    missing_amounts = cleaned["order_amount"].isna()

    print("Missing amounts before:", missing_amounts.sum())
    """
    # Calculate only for missing amounts
    cleaned.loc[missing_amounts, "order_amount"] = (
        cleaned.loc[missing_amounts, "product_id"].map(price_by_product)
        * cleaned.loc[missing_amounts, "quantity"]
        * (1 - cleaned.loc[missing_amounts, "discount_pct"] / 100)
    ).round(2)

    """
    # Calculate replacement amounts for missing records only
    calculated_amounts = (
        cleaned.loc[missing_amounts, "product_id"].map(price_by_product)
        * cleaned.loc[missing_amounts, "quantity"]
        * (1 - cleaned.loc[missing_amounts, "discount_pct"] / 100)
    ).round(2)
    """
    # Debug: Show missing records with calculated amounts
    debug_records = cleaned.loc[
        missing_amounts,
        ["order_id", "product_id", "quantity", "discount_pct", "order_amount"]
    ].copy()
    debug_records["price_inr"] = (
        debug_records["product_id"].map(price_by_product)
    )
    debug_records["quantity_price"] = debug_records["quantity"] * debug_records["price_inr"]
    debug_records["discount_amount"] = debug_records["quantity_price"] * debug_records["discount_pct"] / 100
    debug_records["calculated_amount"] = calculated_amounts
    print("\n--- Missing Order Amounts: Before Repair ---")
    print(debug_records.head().to_string(index=False))
    """
    # Fill missing amounts
    cleaned.loc[missing_amounts, "order_amount"] = calculated_amounts
    print("Missing amounts after:", cleaned["order_amount"].isna().sum())  

    # A.Q4(v): Clean delivery minutes
    print("\n--- A.Q4(v): Clean Delivery Minutes ---")

    # Convert to numeric; invalid entries become NaN
    cleaned["delivery_minutes"] = pd.to_numeric(
        cleaned["delivery_minutes"],
        errors="coerce",
    )

    negative_before = cleaned["delivery_minutes"].lt(0).sum()
    missing_before = cleaned["delivery_minutes"].isna().sum()

    print("Negative delivery times before:", negative_before)
    print("Missing delivery times before:", missing_before)

    # Replace negative delivery times with NaN
    cleaned.loc[
        cleaned["delivery_minutes"] < 0,
        "delivery_minutes"
    ] = float("nan")

    # Map product categories to orders
    category_by_product = products.set_index("product_id")["category"]
    cleaned["category"] = cleaned["product_id"].map(category_by_product)

    # Identify completed and cancelled orders
    is_completed = cleaned["status"].eq("completed")
    is_cancelled = cleaned["status"].eq("cancelled")

    print(
        "Completed orders missing delivery before:",
        cleaned.loc[is_completed, "delivery_minutes"].isna().sum()
    )
    print(
        "Cancelled orders missing delivery before:",
        cleaned.loc[is_cancelled, "delivery_minutes"].isna().sum()
    )

    # Calculate category median using completed orders only
    category_medians = (
        cleaned.loc[is_completed]
        .groupby("category")["delivery_minutes"]
        .transform("median")
    ).round().astype("Int64")

    # Fill missing delivery times for completed orders
    cleaned.loc[is_completed, "delivery_minutes"] = (
        cleaned.loc[is_completed, "delivery_minutes"]
        .fillna(category_medians)
    )

    print(
        "Completed orders missing delivery after:",
        cleaned.loc[is_completed, "delivery_minutes"].isna().sum()
    )
    print(
        "Cancelled orders missing delivery after:",
        cleaned.loc[is_cancelled, "delivery_minutes"].isna().sum()
    )
     
    return cleaned


import sqlite3

from .config import (
    CLEAN_DIR,
    DATABASE_PATH,
    ensure_directories,
)


def save_clean_data(
    customers: pd.DataFrame,
    products: pd.DataFrame,
    orders: pd.DataFrame,
) -> None:
    """Save cleaned datasets to CSV and SQLite."""

    ensure_directories()

    datasets = {
        "customers": customers.copy(),
        "products": products.copy(),
        "orders": orders.copy(),
    }

    # Format order timestamp for storage
    datasets["orders"]["order_ts"] = (
        datasets["orders"]["order_ts"]
        .dt.strftime("%Y-%m-%d %H:%M:%S")
    )

    # Save CSV files
    for table_name, dataframe in datasets.items():
        output_path = CLEAN_DIR / f"{table_name}.csv"
        dataframe.to_csv(output_path, index=False)

        print(f"Saved CSV: {output_path}")
        print(f"Rows saved: {len(dataframe)}")

    # Save SQLite tables
    with sqlite3.connect(DATABASE_PATH) as connection:
        for table_name, dataframe in datasets.items():
            dataframe.to_sql(
                name=table_name,
                con=connection,
                if_exists="replace",
                index=False,
            )

            print(f"Saved SQLite table: {table_name}")

    print(f"\nSQLite database: {DATABASE_PATH}")


def run_order_cleaning() -> pd.DataFrame:
    """Load raw orders and products, then clean orders."""

    orders = pd.read_csv(
        RAW_DIR / "orders.csv",
        keep_default_na=False,
    )

    products = pd.read_csv(
        RAW_DIR / "products.csv",
        keep_default_na=False,
    )

    print("\n===== A.Q4 ORDER CLEANING =====")

    cleaned = clean_orders(orders, products)
    
    validate_orders(cleaned)

    return cleaned

def validate_order_duplicates(orders: pd.DataFrame) -> None:
    """Validate A.Q4(ii): Exact duplicate removal."""

    assert len(orders) == 12965, (
        f"Expected 12965 rows, found {len(orders)}"
    )

    assert not orders.duplicated().any(), (
        "Exact duplicate rows still exist"
    )

    print("A.Q4(ii) validation passed.")

def validate_latest_orders(orders: pd.DataFrame) -> None:
    """Validate A.Q4(iii): Latest order selection."""

    assert len(orders) == 12309, (
        f"Expected 12309 rows, found {len(orders)}"
    )

    assert orders["order_id"].is_unique, (
        "Duplicate order IDs still exist"
    )

    assert not orders["status"].eq("pending").any(), (
        "Pending orders still exist"
    )

    print("A.Q4(iii) validation passed.")

def validate_orders(orders: pd.DataFrame) -> None:
    """Validate order-cleaning checkpoints for A.Q4."""

    # A.Q4(i): Order timestamp must be datetime
    assert pd.api.types.is_datetime64_any_dtype(
        orders["order_ts"]
    ), "order_ts must be datetime"

    # A.Q4(ii): Exact duplicate rows must be removed
    assert orders.duplicated().sum() == 0, (
        "Exact duplicate orders still exist"
    )
    # A.Q4(iii): Latest order records
    assert len(orders) == 12309, (
        f"Expected 12309 rows, found {len(orders)}"
    )

    assert orders["order_id"].is_unique, (
        "Duplicate order IDs still exist"
    )

    assert not orders["status"].eq("pending").any(), (
        "Pending orders still exist"
    )

    # A.Q4(iv): Order amount repair
    assert pd.api.types.is_numeric_dtype(
        orders["order_amount"]
    ), "order_amount must be numeric"

    assert orders["order_amount"].notna().all(), (
        "Missing order amounts still exist"
    )

    assert orders["quantity"].notna().all(), (
        "Missing quantity values found"
    )

    assert orders["discount_pct"].notna().all(), (
        "Missing discount values found"
    )
    # A.Q4(v): Delivery time validation
    validate_delivery_minutes(orders)

    assert orders["status"].eq("completed").sum() == 11110
    assert orders["status"].eq("cancelled").sum() == 1199

    completed = orders.loc[orders["status"].eq("completed")]

    assert round(completed["order_amount"].sum(), 2) == 2998668.80, (
        "Completed order revenue does not match reference"
    )

    print("A.Q4 final validation passed.")

def validate_product_prices(
    orders: pd.DataFrame,
    price_by_product: pd.Series,
) -> None:
    """Validate product price availability for A.Q4(iv)."""

    assert price_by_product.index.is_unique, (
        "Duplicate product IDs found in price lookup"
    )

    # Find orders whose product ID has no valid price
    mapped_prices = orders["product_id"].map(price_by_product)
    missing_prices = mapped_prices.isna()

    print("Orders without valid product prices:", missing_prices.sum())

    if missing_prices.any():
        print(
            orders.loc[
                missing_prices,
                ["order_id", "product_id"]
            ].to_string(index=False)
        )

    assert not missing_prices.any(), (
        "Some orders have missing or invalid product prices"
    )

    print("Product price validation passed.")

def validate_delivery_minutes(orders: pd.DataFrame) -> None:
    """Validate A.Q4(v): Delivery time cleaning."""

    completed = orders.loc[orders["status"].eq("completed")]
    cancelled = orders.loc[orders["status"].eq("cancelled")]

    assert pd.api.types.is_numeric_dtype(
        orders["delivery_minutes"]
    ), "delivery_minutes must be numeric"

    assert not orders["delivery_minutes"].lt(0).any(), (
        "Negative delivery times still exist"
    )

    assert completed["delivery_minutes"].notna().all(), (
        "Completed orders still have missing delivery times"
    )

    assert cancelled["delivery_minutes"].isna().all(), (
        "Cancelled orders must have missing delivery times"
    )

    print("A.Q4(v) validation passed.")

def run_data_cleaning_pipeline() -> None:
    """Execute Part A cleaning and storage."""

    # A.Q3: Clean customers
    cleaned_customers = run_customer_cleaning()

    # A.Q4: Clean orders
    cleaned_orders = run_order_cleaning()

    # A.Q5: Load products
    products = pd.read_csv(
        RAW_DIR / "products.csv",
        keep_default_na=False,
    )

    # A.Q5: Save cleaned data
    save_clean_data(
        customers=cleaned_customers,
        products=products,
        orders=cleaned_orders,
    )

if __name__ == "__main__":
    run_data_cleaning_pipeline()