
"""
Part D.Q6–D.Q8 — ReAct Refund Agent.

D.Q6: Run seven refund scenarios.
D.Q7: Enforce iteration, token and timeout limits.
D.Q8: Save checkpoints and resume after a simulated crash.
"""

import csv
import json
import os
import time
import uuid
from pathlib import Path

from mock_llm import chat
from src.refund_tools import TOOLS_SPEC, execute_tool


ROOT = Path(__file__).resolve().parents[1]
SCENARIOS_PATH = ROOT / "data" / "raw" / "agent_scenarios.csv"
RUNS_PATH = ROOT / "outputs" / "agent_runs.jsonl"
STATE_DIR = ROOT / "state"

SYSTEM_PROMPT = (
    "You are an Instamart refund AGENT. "
    "Use the available tools to check orders and policies. "
    "Only completed orders delivered after 45 minutes qualify "
    "for a full refund. Refunds above INR 1000 must be "
    "escalated to a human, never issued automatically. "
    "Use tool observations to decide your next action."
)

EXPECTED_STATUSES = {
    "S1": "REFUND",
    "S2": "REFUND",
    "S3": "ESCALATE",
    "S4": "DENIED",
    "S5": "DENIED",
    "S6": "NOT_FOUND",
    "S7": "FALLBACK_MAX_ITERATIONS",
}


def get_refund_amount(steps):
    """Get the amount from the actual refund or escalation tool."""
    for step in reversed(steps):
        if step["tool"] in ("issue_refund", "escalate_to_human"):
            return float(step["observation"].get("refund_inr", 0.0))
    return 0.0


# ---------------------------------------------------------
# D.Q8 — Checkpoint management
# ---------------------------------------------------------

def checkpoint_path(run_id):
    """Get the checkpoint filename for a run."""
    return STATE_DIR / f"{run_id}.json"


def save_checkpoint(state):
    """Save the complete agent state after a tool observation."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    path = checkpoint_path(state["run_id"])
    temporary_path = path.with_suffix(".tmp")

    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(state, file, indent=2)

    os.replace(temporary_path, path)


# ---------------------------------------------------------
# D.Q6–D.Q8 — ReAct execution loop
# ---------------------------------------------------------

def continue_agent(
    state,
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=10,
    crash_after_step=None,
):
    """Continue execution using the current or restored state."""

    messages = state["messages"]
    steps = state["steps"]
    start_time = time.monotonic()

    def result(status, final_answer):
        return {
            "scenario_id": state["scenario_id"],
            "status": status,
            "final_answer": final_answer,
            "refund_inr": get_refund_amount(steps),
            "steps": steps,
            "llm_calls": state["llm_calls"],
            "tokens_used": state["tokens_used"],
        }

    while state["llm_calls"] < max_iterations:

        # D.Q7(c): Stop when the time limit is reached.
        elapsed = time.monotonic() - start_time

        if elapsed >= time_limit_seconds:
            return result(
                "FALLBACK_TIMEOUT",
                "Human handoff: agent time limit reached.",
            )

        response = chat(messages, tools=TOOLS_SPEC)

        state["llm_calls"] += 1
        state["tokens_used"] += response.usage.total_tokens

        # D.Q7(b): Check tokens before executing any requested tool.
        if state["tokens_used"] > token_limit:
            return result(
                "FALLBACK_TOKEN_LIMIT",
                "Human handoff: agent token limit exceeded.",
            )

        assistant = response.choices[0].message
        tool_calls = assistant.tool_calls or []

        if not tool_calls:
            final_answer = assistant.content or ""
            status = final_answer.split(":", 1)[0].strip()

            return result(status, final_answer)

        requested_tools = [
            {
                "id": call.id,
                "name": call.name,
                "arguments": call.arguments,
            }
            for call in tool_calls
        ]

        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": requested_tools,
        })

        for call in requested_tools:
            name = call["name"]
            arguments = call["arguments"]

            try:
                observation = execute_tool(name, arguments)
            except (ValueError, TypeError, KeyError) as error:
                observation = {"error": str(error)}

            steps.append({
                "tool": name,
                "arguments": arguments,
                "observation": observation,
            })

            messages.append({
                "role": "tool",
                "tool_call_id": call["id"],
                "name": name,
                "content": json.dumps(observation),
            })

            # D.Q8: Save a checkpoint after every tool observation.
            state["step_number"] = len(steps)
            state["tools_executed"].append(name)

            save_checkpoint(state)

            # D.Q8: Crash only after the checkpoint is saved.
            if (
                crash_after_step is not None
                and state["step_number"] == crash_after_step
            ):
                raise RuntimeError(
                    f"Simulated crash after tool {crash_after_step}"
                )

    # D.Q7(a): Maximum model calls reached.
    return result(
        "FALLBACK_MAX_ITERATIONS",
        "Human handoff: maximum agent iterations reached.",
    )


def run_agent(
    user_message,
    scenario_id,
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=10,
    run_id=None,
    crash_after_step=None,
):
    """Start a new refund agent run."""

    state = {
        "run_id": run_id or uuid.uuid4().hex,
        "scenario_id": scenario_id,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        "tokens_used": 0,
        "llm_calls": 0,
        "step_number": 0,
        "tools_executed": [],
        "steps": [],
    }

    return continue_agent(
        state,
        max_iterations=max_iterations,
        token_limit=token_limit,
        time_limit_seconds=time_limit_seconds,
        crash_after_step=crash_after_step,
    )


def resume_agent(run_id):
    """Restore a checkpoint and continue without repeating tools."""

    path = checkpoint_path(run_id)

    with path.open("r", encoding="utf-8") as file:
        state = json.load(file)

    return continue_agent(state)


# ---------------------------------------------------------
# D.Q6 — Scenario evaluation
# ---------------------------------------------------------

def check_scenario(record, scenario):
    """Compare the result with the assignment reference."""

    scenario_id = record["scenario_id"]
    expected_status = EXPECTED_STATUSES[scenario_id]
    expected_refund = float(scenario["expected_refund_inr"])

    checks = {
        "status": record["status"] == expected_status,
        "refund": abs(record["refund_inr"] - expected_refund) <= 0.01,
    }

    if scenario_id == "S1":
        checks["five_model_calls"] = record["llm_calls"] == 5

    if scenario_id == "S7":
        checks["six_model_calls"] = record["llm_calls"] == 6
        checks["within_token_limit"] = record["tokens_used"] <= 6000

    return checks


def run_all_scenarios():
    """Run all seven scenarios and save their results."""

    RUNS_PATH.parent.mkdir(parents=True, exist_ok=True)

    with SCENARIOS_PATH.open(
        "r", newline="", encoding="utf-8-sig"
    ) as file:
        scenarios = list(csv.DictReader(file))

    if len(scenarios) != 7:
        raise ValueError("Expected exactly seven agent scenarios.")

    records = []
    failures = []

    print("=== Part D.Q6: ReAct Refund Agent ===")
    print(
        f"{'ID':<4} {'Status':<26} {'Refund':>10} "
        f"{'Calls':>6} {'Tokens':>8} {'Check':>8}"
    )

    for scenario in scenarios:
        record = run_agent(
            user_message=scenario["message"],
            scenario_id=scenario["scenario_id"],
        )

        checks = check_scenario(record, scenario)
        passed = all(checks.values())

        record["checks"] = checks
        records.append(record)

        if not passed:
            failures.append(record["scenario_id"])

        print(
            f"{record['scenario_id']:<4} "
            f"{record['status']:<26} "
            f"{record['refund_inr']:>10.2f} "
            f"{record['llm_calls']:>6} "
            f"{record['tokens_used']:>8} "
            f"{'PASS' if passed else 'FAIL':>8}"
        )

    with RUNS_PATH.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record) + "\n")

    print(f"\nResults saved to: {RUNS_PATH}")

    if failures:
        print("FAILED scenarios:", ", ".join(failures))
    else:
        print("D.Q6 ALL SEVEN SCENARIOS PASSED")

    return records


if __name__ == "__main__":
    run_all_scenarios()
