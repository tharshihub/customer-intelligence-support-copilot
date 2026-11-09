
"""Part C.Q2 — Support ticket PII redaction."""

import re

import pandas as pd


# C.Q2: Exact regex patterns and replacement order.
PII_PATTERNS = {
    "email": (r"[\w.+-]+@[\w-]+\.[\w.]+", "[EMAIL]"),
    "phone": (r"\b[6-9]\d{9}\b", "[PHONE]"),
    "order_id": (r"ORD-\d{6}", "[ORDER_ID]"),
}


def redact_ticket_text(text: str) -> tuple[str, dict]:
    """Redact PII and count substitutions in one ticket."""

    text = "" if pd.isna(text) else str(text)
    counts = {}

    for pii_type, (pattern, replacement) in PII_PATTERNS.items():
        text, count = re.subn(pattern, replacement, text)
        counts[pii_type] = count

    return text, counts


def validate_ticket_redaction(results: pd.DataFrame) -> None:
    """Validate C.Q2 against assignment checkpoints."""

    assert len(results) == 164, "Expected 164 tickets"

    assert results["email_count"].sum() == 55, (
        "Expected 55 email substitutions"
    )

    assert results["phone_count"].sum() == 91, (
        "Expected 91 phone substitutions"
    )

    assert results["order_id_count"].sum() == 132, (
        "Expected 132 order ID substitutions"
    )

    total = results[
        ["email_count", "phone_count", "order_id_count"]
    ].to_numpy().sum()

    assert total == 278, "Expected 278 total substitutions"

    # Reapply all patterns: no PII should remain.
    for pii_type, (pattern, _) in PII_PATTERNS.items():
        remaining = results["redacted_text"].apply(
            lambda text: len(re.findall(pattern, text))
        ).sum()

        assert remaining == 0, (
            f"{pii_type}: {remaining} sensitive values remain"
        )

    # Confirm original ticket order and guardrail statuses are preserved.
    assert results["ticket_id"].is_unique, (
        "Duplicate ticket IDs detected"
    )

    assert results["status"].value_counts().get("eligible", 0) == 156
    assert results["status"].value_counts().get("skipped_empty", 0) == 4
    assert results["status"].value_counts().get("flagged_long", 0) == 4

    print("\nPASS | 55 emails redacted")
    print("PASS | 91 phone numbers redacted")
    print("PASS | 132 order IDs redacted")
    print("PASS | 278 total substitutions")
    print("PASS | Zero sensitive patterns remain")
    print("PASS | C.Q1 guardrail statuses preserved")
    print("\nC.Q2 ALL CHECKPOINTS PASSED")


def run_ticket_redaction(tickets: pd.DataFrame) -> pd.DataFrame:
    """Apply C.Q2 redaction to the existing C.Q1 results."""

    results = tickets.copy()

    redactions = results["text"].apply(redact_ticket_text)

    results["redacted_text"] = redactions.apply(
        lambda item: item[0]
    )

    results["email_count"] = redactions.apply(
        lambda item: item[1]["email"]
    )

    results["phone_count"] = redactions.apply(
        lambda item: item[1]["phone"]
    )

    results["order_id_count"] = redactions.apply(
        lambda item: item[1]["order_id"]
    )

    print("\n=== Part C.Q2: PII Redaction ===")
    print(f"Emails redacted: {results['email_count'].sum()}")
    print(f"Phones redacted: {results['phone_count'].sum()}")
    print(f"Order IDs redacted: {results['order_id_count'].sum()}")

    validate_ticket_redaction(results)

    return results


if __name__ == "__main__":
    # Standalone execution: prepare C.Q1 output first.
    from src.ticket_guardrails import run_ticket_guardrails

    tickets = run_ticket_guardrails()
    run_ticket_redaction(tickets)
