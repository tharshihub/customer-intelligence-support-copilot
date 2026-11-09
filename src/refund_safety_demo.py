
"""
Part D.Q7 — Stop Safely (3 marks).

Demonstrates:
1. Maximum iterations: S7 -> FALLBACK_MAX_ITERATIONS
2. Token limit: S1, token_limit=200 -> FALLBACK_TOKEN_LIMIT
3. Timeout: S1, time_limit_seconds=0 -> FALLBACK_TIMEOUT

The existing D.Q6 agent and its seven-scenario log remain unchanged.
"""

import csv
import json

from src.refund_agent import (
    SCENARIOS_PATH,
    RUNS_PATH,
    run_agent,
)


def main():
    # Read the S1 customer message from the original scenario CSV.
    with SCENARIOS_PATH.open(
        "r", newline="", encoding="utf-8-sig"
    ) as file:
        scenarios = {
            row["scenario_id"]: row
            for row in csv.DictReader(file)
        }

    s1_message = scenarios["S1"]["message"]

    # D.Q7(a): Use the previously verified S7 run.
    with RUNS_PATH.open("r", encoding="utf-8") as file:
        records = [json.loads(line) for line in file]

    s7_result = next(
        record for record in records
        if record["scenario_id"] == "S7"
    )

    # D.Q7(b): Stop immediately after cumulative tokens exceed 200.
    token_result = run_agent(
        user_message=s1_message,
        scenario_id="S1_TOKEN_DEMO",
        token_limit=200,
    )

    # D.Q7(c): Stop before the first model call.
    timeout_result = run_agent(
        user_message=s1_message,
        scenario_id="S1_TIMEOUT_DEMO",
        time_limit_seconds=0,
    )

    tests = [
        (
            "D.Q7(a) Maximum iterations",
            s7_result,
            "FALLBACK_MAX_ITERATIONS",
        ),
        (
            "D.Q7(b) Token limit",
            token_result,
            "FALLBACK_TOKEN_LIMIT",
        ),
        (
            "D.Q7(c) Timeout",
            timeout_result,
            "FALLBACK_TIMEOUT",
        ),
    ]

    print("=== Part D.Q7: Stop Safely ===")

    for name, result, expected_status in tests:
        passed = result["status"] == expected_status

        print(f"\n{name}: {'PASS' if passed else 'FAIL'}")
        print(f"Status: {result['status']}")
        print(f"Model calls: {result['llm_calls']}")
        print(f"Tokens used: {result['tokens_used']}")
        print(f"Completed steps: {len(result['steps'])}")
        print(f"Final answer: {result['final_answer']}")

        assert passed, (
            f"Expected {expected_status}, got {result['status']}"
        )

        # Every fallback must provide a human-handoff summary.
        assert "human handoff" in result["final_answer"].lower()

        # Every fallback must preserve completed tool steps.
        assert isinstance(result["steps"], list)

    # S7 must finish all six model calls without hitting 6000 tokens.
    assert s7_result["llm_calls"] == 6
    assert s7_result["tokens_used"] <= 6000

    # The token-limit fallback must occur after exceeding 200 tokens.
    assert token_result["tokens_used"] > 200

    # A zero-second timeout must occur before the first model call.
    assert timeout_result["llm_calls"] == 0
    assert timeout_result["tokens_used"] == 0
    assert timeout_result["steps"] == []

    print("\nD.Q7 ALL THREE SAFETY CHECKS PASSED")


if __name__ == "__main__":
    main()
