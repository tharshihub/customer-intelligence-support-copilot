
"""Part C.Q1 — Ticket input guardrails."""

import pandas as pd

from mock_llm import est_tokens
from src.config import RAW_DIR


MAX_INPUT_TOKENS = 400


# REQUIRED: Check each ticket before calling the model.
def check_ticket_input(text: str) -> dict:
    text = "" if pd.isna(text) else str(text)

    if not text.strip():
        return {
            "status": "skipped_empty",
            "estimated_tokens": 0,
        }

    tokens = est_tokens(text)

    if tokens > MAX_INPUT_TOKENS:
        return {
            "status": "flagged_long",
            "estimated_tokens": tokens,
        }

    return {
        "status": "eligible",
        "estimated_tokens": tokens,
    }


# VALIDATION: Keep assertions separate from business logic.
def validate_ticket_guardrails(results: pd.DataFrame) -> None:
    counts = results["status"].value_counts()

    assert len(results) == 164
    assert counts.get("skipped_empty", 0) == 4
    assert counts.get("flagged_long", 0) == 4
    assert counts.get("eligible", 0) == 156

    long_tokens = results.loc[
        results["status"] == "flagged_long",
        "estimated_tokens",
    ].tolist()

    assert long_tokens == [734] * 4

    assert results.loc[
        results["status"] == "eligible",
        "estimated_tokens",
    ].le(MAX_INPUT_TOKENS).all()

    print("C.Q1 ALL CHECKPOINTS PASSED")


# RUNNER: Load data, apply checks and validate.
def run_ticket_guardrails() -> pd.DataFrame:
    tickets = pd.read_csv(
        RAW_DIR / "tickets.csv",
        keep_default_na=False,
    )

    checks = tickets["text"].apply(check_ticket_input)

    results = tickets.copy()
    results["status"] = checks.apply(lambda x: x["status"])
    results["estimated_tokens"] = checks.apply(
        lambda x: x["estimated_tokens"]
    )

    print("\n=== Part C.Q1: Input Guardrails ===")
    print(results["status"].value_counts())
    print(
        "Long ticket token estimates:",
        results.loc[
            results["status"] == "flagged_long",
            "estimated_tokens",
        ].tolist(),
    )

    validate_ticket_guardrails(results)

    return results


if __name__ == "__main__":
    run_ticket_guardrails()
