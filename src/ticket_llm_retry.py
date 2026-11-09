
"""Part C.Q4 — Resilient LLM calls with exponential backoff."""

import random
import re
import time

import pandas as pd

import mock_llm
from mock_llm import TransientLLMError

from src.ticket_redaction import PII_PATTERNS


BASE_DELAY = 0.05
MAX_RETRIES = 3


def call_with_retry(ticket_id, redacted_text, system_prompt):
    """Call the LLM with up to 3 retries for transient errors."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": redacted_text},
    ]

    # Check every message for unredacted PII before calling the model.
    for message in messages:
        for pattern, _ in PII_PATTERNS.values():
            assert not re.search(pattern, message["content"]), (
                f"Unredacted PII in ticket {ticket_id}"
            )

    # The user message must contain only the redacted ticket text.
    assert messages[1]["content"] == redacted_text

    for retry_number in range(MAX_RETRIES + 1):
        try:
            response = mock_llm.chat(messages)
            return response, retry_number

        except TransientLLMError:
            if retry_number == MAX_RETRIES:
                raise

            wait = (
                BASE_DELAY
                * (2 ** retry_number)
                * random.uniform(0.9, 1.15)
            )

            print(
                f"RETRY | ticket={ticket_id} "
                f"retry={retry_number} "
                f"wait={wait:.5f}s"
            )

            time.sleep(wait)


def validate_ticket_retries(results, total_retries):
    """Validate the Part C.Q4 acceptance criteria."""

    assert len(results) == 164
    assert total_retries == 19

    eligible = results[results["status"] == "eligible"]
    rejected = results[results["status"] != "eligible"]

    assert len(eligible) == 156
    assert eligible["response"].notna().all()
    assert len(rejected) == 8
    assert rejected["response"].isna().all()

    print("\nPASS | 156 eligible tickets processed")
    print("PASS | 19 transient failures recovered")
    print("PASS | No permanent model errors")
    print("PASS | Empty and long tickets never sent to LLM")
    print("PASS | No unredacted PII in system or user messages")
    print("\nC.Q4 ALL CHECKPOINTS PASSED")


def run_ticket_llm_retry(tickets, system_prompt):
    """Run LLM classification on eligible, redacted tickets."""

    results = tickets.copy()

    results["response"] = None
    results["tokens_used"] = 0
    results["retry_count"] = 0

    # Reset the mock before each full run.
    mock_llm.reset()

    total_retries = 0

    for index, ticket in results.iterrows():

        # Never send empty or long tickets to the model.
        if ticket["status"] != "eligible":
            continue

        response, retries = call_with_retry(
            ticket["ticket_id"],
            ticket["redacted_text"],
            system_prompt,
        )

        results.at[index, "response"] = (
            response.choices[0].message.content
        )
        results.at[index, "tokens_used"] = (
            response.usage.total_tokens
        )
        results.at[index, "retry_count"] = retries

        total_retries += retries

    print("\n=== Part C.Q4: Resilient LLM Calls ===")
    print(f"Eligible tickets: {len(results[results['status'] == 'eligible'])}")
    print(f"Recovered transient failures: {total_retries}")
    print(f"Total tokens: {results['tokens_used'].sum()}")

    validate_ticket_retries(results, total_retries)

    return results


if __name__ == "__main__":
    from src.ticket_guardrails import run_ticket_guardrails
    from src.ticket_redaction import run_ticket_redaction
    from src.ticket_prompts import ZERO_SHOT_PROMPT

    tickets = run_ticket_guardrails()
    redacted_tickets = run_ticket_redaction(tickets)

    run_ticket_llm_retry(redacted_tickets, ZERO_SHOT_PROMPT)
