
"""Part C.Q7 — Token budget, final ticket routing and CSV exports."""

import pandas as pd
import mock_llm

from src.config import OUTPUT_DIR
from src.ticket_llm_retry import call_with_retry
from src.ticket_response_parser import parse_classification
from src.ticket_prompts import ZERO_SHOT_PROMPT, FEW_SHOT_PROMPT


# C.Q7(a) — Zero-shot budget demonstration.
BUDGET_DEMO = 6000

# C.Q7(b) — Final few-shot run.
FINAL_BUDGET = 100000

CLASSIFIED = {"classified_json", "classified_keyword"}
CATEGORIES = {"billing", "delivery", "technical", "fraud"}

CSV_COLUMNS = [
    "ticket_id",
    "customer_id",
    "status",
    "category",
    "segment_name",
    "priority",
    "route",
    "tokens_used",
]


def classify_with_budget(tickets, system_prompt, budget):
    """C.Q7(a, b) — Classify tickets within the token budget."""

    results = tickets.copy()
    results["category"] = None
    results["tokens_used"] = 0

    total_tokens = 0

    # Each run starts with a fresh mock LLM state.
    mock_llm.reset()

    # Preserve original ticket order.
    for index, ticket in results.iterrows():
        if ticket["status"] != "eligible":
            continue

        # Check BEFORE calling the model.
        if total_tokens >= budget:
            results.at[index, "status"] = "budget_exhausted"
            continue

        response, _ = call_with_retry(
            ticket["ticket_id"],
            ticket["redacted_text"],
            system_prompt,
        )

        tokens = response.usage.total_tokens
        status, category = parse_classification(
            response.choices[0].message.content
        )

        results.at[index, "status"] = status
        results.at[index, "category"] = category
        results.at[index, "tokens_used"] = tokens

        total_tokens += tokens

    return results


def route_tickets(results):
    """C.Q7(b) — Route tickets using Part B customer segments."""

    segments = pd.read_csv(
        OUTPUT_DIR / "customer_segments.csv"
    )[["customer_id", "segment_name", "is_premium"]]

    results = results.merge(
        segments,
        on="customer_id",
        how="left",
        validate="many_to_one",
        sort=False,
    )

    # Missing customers and False strings are non-premium.
    results["is_premium"] = (
        results["is_premium"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )

    def get_route(ticket):
        # C.Q7(b.1) — Skipped, invalid or budget-exhausted.
        if ticket["status"] not in CLASSIFIED:
            return "NORMAL", "human_review"

        # C.Q7(b.2) — Fraud overrides premium status.
        if ticket["category"] == "fraud":
            return "HIGH", "security_team"

        # C.Q7(b.3) — Premium customer.
        if ticket["is_premium"]:
            return "HIGH", "premium_desk"

        # C.Q7(b.4) — Standard customer.
        return "NORMAL", "support_queue"

    routing = results.apply(get_route, axis=1)

    results["priority"] = routing.apply(lambda item: item[0])
    results["route"] = routing.apply(lambda item: item[1])

    # Missing category/segment values remain blank in CSV.
    return results[CSV_COLUMNS]


def validate_triage_results(results, is_final=False):
    """Validate ticket counts, budgets, categories and routing."""

    assert len(results) == 164
    assert list(results.columns) == CSV_COLUMNS
    assert results["ticket_id"].is_unique
    assert results["tokens_used"].ge(0).all()

    # C.Q7(a) — Exhausted tickets must not consume tokens.
    exhausted = results[
        results["status"] == "budget_exhausted"
    ]

    assert (exhausted["tokens_used"] == 0).all()
    assert (exhausted["route"] == "human_review").all()
    assert (exhausted["priority"] == "NORMAL").all()

    # C.Q7(b.1) — Unclassified tickets go to human review.
    unclassified = results[
        ~results["status"].isin(CLASSIFIED)
    ]

    assert (unclassified["route"] == "human_review").all()
    assert (unclassified["priority"] == "NORMAL").all()
    assert unclassified["category"].isna().all()

    # C.Q7(b.2) — Fraud always goes to security.
    classified = results[
        results["status"].isin(CLASSIFIED)
    ]

    assert classified["category"].isin(CATEGORIES).all()

    fraud = classified[classified["category"] == "fraud"]

    assert (fraud["route"] == "security_team").all()
    assert (fraud["priority"] == "HIGH").all()

    # C.Q7(b.3, b.4) — Validate premium and standard routing.
    segments = pd.read_csv(
        OUTPUT_DIR / "customer_segments.csv"
    )[["customer_id", "is_premium"]]

    checked = results.merge(
        segments,
        on="customer_id",
        how="left",
        validate="many_to_one",
    )

    checked["is_premium"] = (
        checked["is_premium"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )

    non_fraud = checked[
        checked["status"].isin(CLASSIFIED)
        & (checked["category"] != "fraud")
    ]

    premium = non_fraud[non_fraud["is_premium"]]
    standard = non_fraud[~non_fraud["is_premium"]]

    assert (premium["route"] == "premium_desk").all()
    assert (premium["priority"] == "HIGH").all()

    assert (standard["route"] == "support_queue").all()
    assert (standard["priority"] == "NORMAL").all()

    # C.Q7(a) — Demo must exhaust at least one ticket.
    if not is_final:
        assert len(exhausted) > 0
        print(
            f"PASS | Budget exhausted: {len(exhausted)} tickets"
        )

    # C.Q7(b) — Full few-shot reference counts.
    else:
        expected = {
            "classified_json": 135,
            "classified_keyword": 12,
            "flagged_invalid_output": 9,
            "skipped_empty": 4,
            "flagged_long": 4,
        }

        for status, count in expected.items():
            assert (results["status"] == status).sum() == count

        assert len(exhausted) == 0
        assert results["tokens_used"].sum() == 21724
        assert len(unclassified) == 17

        print("PASS | Final few-shot status counts")
        print("PASS | Final tokens: 21,724")
        print("PASS | Human review: 17 tickets")

    print("PASS | 164 unique tickets and required CSV columns")
    print("PASS | Human-review routing")
    print("PASS | Fraud routing")
    print("PASS | Premium customer routing")
    print("PASS | Standard customer routing")


def validate_saved_csv(path, is_final=False):
    """Reload the exported CSV and verify its contents."""

    assert path.exists(), f"Missing CSV: {path}"

    saved = pd.read_csv(path)

    validate_triage_results(saved, is_final=is_final)

    print(f"PASS | Exported CSV verified: {path}")


def run_ticket_triage(redacted_tickets):
    """Run C.Q7(a) and C.Q7(b), then verify both CSV files."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # C.Q7(a) — Zero-shot budget demonstration: 6,000 tokens.
    # ---------------------------------------------------------
    print("\n=== C.Q7(a): Budget Demonstration ===")

    demo = classify_with_budget(
        redacted_tickets,
        ZERO_SHOT_PROMPT,
        BUDGET_DEMO,
    )

    demo = route_tickets(demo)

    demo_path = OUTPUT_DIR / "triage_budget_demo.csv"
    demo.to_csv(demo_path, index=False)

    validate_saved_csv(demo_path)

    exhausted_count = (
        demo["status"] == "budget_exhausted"
    ).sum()

    print(f"Budget used: {BUDGET_DEMO:,}")
    print(f"Tokens consumed: {demo['tokens_used'].sum():,}")
    print(f"Budget-exhausted tickets: {exhausted_count}")

    # ---------------------------------------------------------
    # C.Q7(b) — Final few-shot run: 100,000 tokens.
    # ---------------------------------------------------------
    print("\n=== C.Q7(b): Final Few-shot Triage ===")

    final = classify_with_budget(
        redacted_tickets,
        FEW_SHOT_PROMPT,
        FINAL_BUDGET,
    )

    final = route_tickets(final)

    final_path = OUTPUT_DIR / "triage_results.csv"
    final.to_csv(final_path, index=False)

    validate_saved_csv(final_path, is_final=True)

    print(f"Budget used: {FINAL_BUDGET:,}")
    print(f"Tokens consumed: {final['tokens_used'].sum():,}")

    print("\nFinal status counts:")
    print(final["status"].value_counts())

    print("\nFinal routing counts:")
    print(final["route"].value_counts())

    print("\nC.Q7(a) AND C.Q7(b) ALL CHECKPOINTS PASSED")

    return demo, final


if __name__ == "__main__":
    from src.ticket_guardrails import run_ticket_guardrails
    from src.ticket_redaction import run_ticket_redaction

    tickets = run_ticket_guardrails()
    redacted_tickets = run_ticket_redaction(tickets)

    run_ticket_triage(redacted_tickets)
