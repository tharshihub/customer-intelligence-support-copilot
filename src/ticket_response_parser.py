
"""Part C.Q5 — Parse and validate LLM classification responses."""

import json
import re

from src.ticket_prompts import ALLOWED_CATEGORIES


def parse_classification(response_text):
    """Extract a valid category from an LLM response."""

    try:
        result = json.loads(response_text)

        if isinstance(result, dict):
            category = result.get("category")

            if category in ALLOWED_CATEGORIES:
                return "classified_json", category

        # Valid JSON with an invalid category: do not guess.
        return "flagged_invalid_output", None

    except (json.JSONDecodeError, TypeError):
        # Keyword recovery applies only to non-JSON responses.
        words = re.findall(r"[a-z]+", str(response_text).lower())
        matches = set(words) & ALLOWED_CATEGORIES

        if len(matches) == 1:
            return "classified_keyword", matches.pop()

        return "flagged_invalid_output", None


def validate_ticket_responses(results):
    """Verify the Part C.Q5 reference counts."""

    counts = results["status"].value_counts()

    assert len(results) == 164
    assert counts.get("classified_json", 0) == 135
    assert counts.get("classified_keyword", 0) == 12
    assert counts.get("flagged_invalid_output", 0) == 9
    assert counts.get("skipped_empty", 0) == 4
    assert counts.get("flagged_long", 0) == 4

    classified = results[
        results["status"].isin([
            "classified_json",
            "classified_keyword",
        ])
    ]

    assert classified["category"].isin(ALLOWED_CATEGORIES).all()

    invalid = results[
        results["status"] == "flagged_invalid_output"
    ]
    assert invalid["category"].isna().all()

    print("\nPASS | 135 valid JSON classifications")
    print("PASS | 12 keyword classifications recovered")
    print("PASS | 9 invalid outputs flagged for human review")
    print("PASS | 4 empty and 4 long tickets preserved")
    print("PASS | Only allowed categories accepted")
    print("\nC.Q5 ALL CHECKPOINTS PASSED")


def run_ticket_response_parser(tickets):
    """Parse model responses and update ticket statuses."""

    results = tickets.copy()
    results["category"] = None

    for index, ticket in results.iterrows():
        if ticket["status"] != "eligible":
            continue

        status, category = parse_classification(
            ticket["response"]
        )

        results.at[index, "status"] = status
        results.at[index, "category"] = category

    print("\n=== Part C.Q5: Response Parsing ===")
    print(results["status"].value_counts())

    validate_ticket_responses(results)

    return results


if __name__ == "__main__":
    from src.ticket_guardrails import run_ticket_guardrails
    from src.ticket_redaction import run_ticket_redaction
    from src.ticket_prompts import ZERO_SHOT_PROMPT
    from src.ticket_llm_retry import run_ticket_llm_retry

    tickets = run_ticket_guardrails()
    redacted_tickets = run_ticket_redaction(tickets)

    llm_results = run_ticket_llm_retry(
        redacted_tickets,
        ZERO_SHOT_PROMPT,
    )

    run_ticket_response_parser(llm_results)
