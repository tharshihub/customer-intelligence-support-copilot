
"""
Part D.Q8 — Demonstrate checkpoint and resume.

S1 must:
- Execute get_order and search_policy.
- Save checkpoint and simulate a crash.
- Resume with calculate_refund and issue_refund.
- Finish with REFUND of INR 128.25.
"""

import csv
import json

from src.refund_agent import (
    SCENARIOS_PATH,
    checkpoint_path,
    resume_agent,
    run_agent,
)


RUN_ID = "s1_crash_resume_demo"

EXPECTED_TOOLS = [
    "get_order",
    "search_policy",
    "calculate_refund",
    "issue_refund",
]


def main():
    # Load S1 from the assignment scenario file.
    with SCENARIOS_PATH.open(
        "r", newline="", encoding="utf-8-sig"
    ) as file:
        scenarios = {
            row["scenario_id"]: row
            for row in csv.DictReader(file)
        }

    s1_message = scenarios["S1"]["message"]

    # Remove only the previous demonstration checkpoint.
    path = checkpoint_path(RUN_ID)
    path.unlink(missing_ok=True)

    print("=== Part D.Q8: Make the Run Resumable ===")

    # 1. Run until the second tool has executed.
    try:
        run_agent(
            user_message=s1_message,
            scenario_id="S1",
            run_id=RUN_ID,
            crash_after_step=2,
        )
    except RuntimeError as error:
        if str(error) != "Simulated crash after tool 2":
            raise
        print(f"\n{error}")

    # 2. Verify the checkpoint was saved before the crash.
    assert path.exists(), "Checkpoint not found."

    with path.open("r", encoding="utf-8") as file:
        checkpoint = json.load(file)

    before_crash = checkpoint["tools_executed"]

    assert checkpoint["run_id"] == RUN_ID
    assert checkpoint["scenario_id"] == "S1"
    assert checkpoint["step_number"] == 2
    assert before_crash == EXPECTED_TOOLS[:2]

    print("Checkpoint saved:", path)
    print("Tools before crash:", before_crash)

    # 3. Resume the same run from its checkpoint.
    result = resume_agent(RUN_ID)

    # 4. Inspect the updated checkpoint.
    with path.open("r", encoding="utf-8") as file:
        final_checkpoint = json.load(file)

    all_tools = final_checkpoint["tools_executed"]
    after_resume = all_tools[len(before_crash):]

    print("\nTools after resume:", after_resume)
    print("All tools executed:", all_tools)
    print("Final status:", result["status"])
    print("Refund amount:", result["refund_inr"])
    print("Total model calls:", result["llm_calls"])
    print("Total tokens:", result["tokens_used"])

    # 5. Validate the assignment's exact requirements.
    assert all_tools == EXPECTED_TOOLS
    assert len(all_tools) == 4
    assert len(set(all_tools)) == 4
    assert final_checkpoint["step_number"] == 4
    assert len(result["steps"]) == 4

    assert result["status"] == "REFUND"
    assert abs(result["refund_inr"] - 128.25) <= 0.01
    assert result["llm_calls"] == 5

    print("\nD.Q8 CHECKPOINT AND RESUME PASSED")


if __name__ == "__main__":
    main()
