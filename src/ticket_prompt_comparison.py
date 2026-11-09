
"""Part C.Q6 — Compare zero-shot and few-shot classification."""

import pandas as pd

from src.ticket_llm_retry import run_ticket_llm_retry
from src.ticket_response_parser import run_ticket_response_parser
from src.ticket_prompts import ZERO_SHOT_PROMPT, FEW_SHOT_PROMPT


CLASSIFIED_STATUSES = {"classified_json", "classified_keyword"}


def calculate_metrics(results):
    """Calculate accuracy and token usage over classified tickets."""

    classified = results[
        results["status"].isin(CLASSIFIED_STATUSES)
    ]

    count = len(classified)
    correct = int(
        (classified["category"] == classified["true_category"]).sum()
    )
    total_tokens = int(results["tokens_used"].sum())

    return {
        "classified": count,
        "correct": correct,
        "accuracy": correct / count if count else 0,
        "total_tokens": total_tokens,
        "tokens_per_ticket": total_tokens / count if count else 0,
    }


def validate_prompt_comparison(zero_results, few_results):
    """Validate C.Q6 reference accuracy and error patterns."""

    zero = calculate_metrics(zero_results)
    few = calculate_metrics(few_results)

    assert zero["classified"] == 147
    assert few["classified"] == 147

    assert zero["correct"] == 126
    assert few["correct"] == 147

    assert abs(zero["accuracy"] - 0.857) <= 0.005
    assert abs(few["accuracy"] - 1.0) <= 0.005

    # Reference: all 21 zero-shot errors are predicted as billing.
    errors = zero_results[
        zero_results["status"].isin(CLASSIFIED_STATUSES)
        & (zero_results["category"] != zero_results["true_category"])
    ]

    error_counts = errors.groupby(
        ["true_category", "category"]
    ).size()

    assert len(errors) == 21
    assert error_counts.get(("fraud", "billing"), 0) == 11
    assert error_counts.get(("technical", "billing"), 0) == 10
    assert len(error_counts) == 2

    print("\nPASS | Both runs classified 147 tickets")
    print("PASS | Zero-shot: 126/147 correct (85.7%)")
    print("PASS | Few-shot: 147/147 correct (100%)")
    print("PASS | Zero-shot: 11 fraud and 10 technical errors")
    print("PASS | All 21 zero-shot errors predicted as billing")
    print("\nC.Q6 ALL CHECKPOINTS PASSED")


def run_ticket_prompt_comparison(redacted_tickets):
    """Run both classification prompts and compare results."""

    print("\n=== Part C.Q6: Zero-shot Run ===")

    zero_results = run_ticket_llm_retry(
        redacted_tickets,
        ZERO_SHOT_PROMPT,
    )
    zero_results = run_ticket_response_parser(zero_results)

    print("\n=== Part C.Q6: Few-shot Run ===")

    few_results = run_ticket_llm_retry(
        redacted_tickets,
        FEW_SHOT_PROMPT,
    )
    few_results = run_ticket_response_parser(few_results)

    zero = calculate_metrics(zero_results)
    few = calculate_metrics(few_results)

    print("\n=== Part C.Q6: Comparison ===")

    comparison = pd.DataFrame({
        "Metric": [
            "Classified tickets",
            "Correct classifications",
            "Accuracy (%)",
            "Total tokens",
            "Tokens per classified ticket",
        ],
        "Zero-shot": [
            zero["classified"],
            zero["correct"],
            round(zero["accuracy"] * 100, 2),
            zero["total_tokens"],
            round(zero["tokens_per_ticket"], 2),
        ],
        "Few-shot": [
            few["classified"],
            few["correct"],
            round(few["accuracy"] * 100, 2),
            few["total_tokens"],
            round(few["tokens_per_ticket"], 2),
        ],
    })

    print(comparison.to_string(index=False))

    print("\n=== Zero-shot Confusion Table ===")

    classified = zero_results[
        zero_results["status"].isin(CLASSIFIED_STATUSES)
    ]

    confusion = pd.crosstab(
        classified["true_category"],
        classified["category"],
        rownames=["Actual"],
        colnames=["Predicted"],
    )

    print(confusion.to_string())

    print("\n=== Cost-Benefit Analysis ===")

    accuracy_gain = (few["accuracy"] - zero["accuracy"]) * 100
    extra_tokens = few["total_tokens"] - zero["total_tokens"]
    extra_correct = few["correct"] - zero["correct"]

    print(
        f"Zero-shot classified {zero['classified']} tickets with "
        f"{zero['accuracy']:.1%} accuracy, while few-shot achieved "
        f"{few['accuracy']:.1%} accuracy."
    )
    print(
        f"Few-shot improved accuracy by "
        f"{accuracy_gain:.1f} percentage points."
    )
    print(
        f"Few-shot consumed {few['total_tokens']:,} tokens "
        f"compared with {zero['total_tokens']:,} for zero-shot, "
        f"an additional {extra_tokens:,} tokens."
    )
    print(
        f"The extra tokens are worthwhile when avoiding "
        f"{extra_correct} additional classification errors "
        f"justifies the cost, particularly for sensitive "
        f"support tickets."
    )

    validate_prompt_comparison(zero_results, few_results)

    return zero_results, few_results


if __name__ == "__main__":
    from src.ticket_guardrails import run_ticket_guardrails
    from src.ticket_redaction import run_ticket_redaction

    tickets = run_ticket_guardrails()
    redacted_tickets = run_ticket_redaction(tickets)

    run_ticket_prompt_comparison(redacted_tickets)
