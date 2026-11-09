
"""
Part D.Q5 — Define Tools and Tool Map (4 marks).

Implements:
1. get_order
2. search_policy
3. calculate_refund
4. issue_refund
5. escalate_to_human

Also includes TOOLS_SPEC, TOOL_MAP, tool dispatch,
business-rule validation and acceptance tests.
"""

import json
import math
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from src.policy_retrieval import retrieve


# --------------------------------------------------
# Configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "instamart.db"
LEDGER_PATH = ROOT / "outputs" / "refund_ledger.jsonl"


# --------------------------------------------------
# Q5.1 — Get order details from SQLite
# --------------------------------------------------

def get_order(order_id):
    """Retrieve an order's status, amount and delivery time."""

    with sqlite3.connect(DB_PATH) as connection:
        connection.row_factory = sqlite3.Row

        row = connection.execute(
            """
            SELECT order_id, status, order_amount,
                   delivery_minutes
            FROM orders
            WHERE order_id = ?
            """,
            (order_id,),
        ).fetchone()

    if row is None:
        return {"error": "order not found"}

    return dict(row)


# --------------------------------------------------
# Q5.2 — Search policy documents
# --------------------------------------------------

def search_policy(query):
    """Retrieve the three most relevant policy chunks."""

    return {"results": retrieve(query, k=3)}


# --------------------------------------------------
# Q5.3 — Calculate refund
# --------------------------------------------------

def calculate_refund(order_amount, delivery_minutes):
    """Calculate the refund using the late-delivery rule."""

    amount = float(order_amount)
    minutes = float(delivery_minutes)

    if not math.isfinite(amount) or not math.isfinite(minutes):
        raise ValueError("Amount and minutes must be finite")

    if amount < 0 or minutes < 0:
        raise ValueError("Amount and minutes must be non-negative")

    refund = round(amount, 2) if minutes > 45 else 0.0

    return {"refund_inr": refund}


# --------------------------------------------------
# Shared business-rule validation
# --------------------------------------------------

def validate_refund_action(order_id, refund_inr):
    """Verify eligibility and refund amount against SQLite."""

    # Re-read the actual order instead of trusting the model.
    order = get_order(order_id)

    if "error" in order:
        raise ValueError("Order not found")

    if order["status"] != "completed":
        raise ValueError("Only completed orders can be refunded")

    # Calculate the amount using the actual order details.
    expected_refund = calculate_refund(
        order["order_amount"],
        order["delivery_minutes"],
    )["refund_inr"]

    proposed_refund = float(refund_inr)

    if not math.isfinite(proposed_refund):
        raise ValueError("Refund amount must be finite")

    if proposed_refund <= 0:
        raise ValueError("Refund amount must be positive")

    if round(proposed_refund, 2) != expected_refund:
        raise ValueError("Refund amount does not match policy")

    if expected_refund == 0:
        raise ValueError("Order is not eligible for a refund")

    return expected_refund


# --------------------------------------------------
# Shared JSONL ledger writer
# --------------------------------------------------

def record_action(tool_name, order_id, refund_inr):
    """Append a timestamped refund action to the JSONL ledger."""

    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Avoid recording the same refund action twice.
    if LEDGER_PATH.exists():
        with LEDGER_PATH.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue

                previous = json.loads(line)

                if previous.get("order_id") == order_id:
                    if (
                        previous.get("tool") == tool_name
                        and previous.get("refund_inr") == refund_inr
                    ):
                        return {"refund_inr": refund_inr}

                    raise ValueError(
                        "Another refund action already exists "
                        "for this order"
                    )

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool": tool_name,
        "order_id": order_id,
        "refund_inr": refund_inr,
    }

    with LEDGER_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")

    return {"refund_inr": refund_inr}


# --------------------------------------------------
# Q5.4 — Issue refund
# --------------------------------------------------

def issue_refund(order_id, refund_inr):
    """Record an eligible refund of INR 1000 or less."""

    amount = validate_refund_action(order_id, refund_inr)

    if amount > 1000:
        raise ValueError(
            "Refund above INR 1000 requires human approval"
        )

    return record_action("issue_refund", order_id, amount)


# --------------------------------------------------
# Q5.5 — Escalate refund to human
# --------------------------------------------------

def escalate_to_human(order_id, refund_inr):
    """Record an eligible high-value refund for human approval."""

    amount = validate_refund_action(order_id, refund_inr)

    if amount <= 1000:
        raise ValueError(
            "Only refunds above INR 1000 require escalation"
        )

    return record_action("escalate_to_human", order_id, amount)


# --------------------------------------------------
# Q5.6 — JSON-style tool specifications
# --------------------------------------------------

def tool_spec(name, description, properties):
    """Create a JSON-style tool parameter specification."""

    return {
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": {
                key: {"type": value}
                for key, value in properties.items()
            },
            "required": list(properties),
            "additionalProperties": False,
        },
    }


TOOLS_SPEC = [
    tool_spec(
        "get_order",
        "Retrieve an order's status, amount and delivery time.",
        {"order_id": "string"},
    ),
    tool_spec(
        "search_policy",
        "Retrieve the three most relevant policy chunks.",
        {"query": "string"},
    ),
    tool_spec(
        "calculate_refund",
        "Calculate the refund using the delivery-time rule.",
        {
            "order_amount": "number",
            "delivery_minutes": "number",
        },
    ),
    tool_spec(
        "issue_refund",
        "Record an eligible refund of INR 1000 or less.",
        {
            "order_id": "string",
            "refund_inr": "number",
        },
    ),
    tool_spec(
        "escalate_to_human",
        "Record an eligible high-value refund for human approval.",
        {
            "order_id": "string",
            "refund_inr": "number",
        },
    ),
]


# --------------------------------------------------
# Q5.7 — Tool name to Python function mapping
# --------------------------------------------------

TOOL_MAP = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human,
}


# --------------------------------------------------
# Q5.8 — Execute tool requested by the model
# --------------------------------------------------

def execute_tool(name, arguments):
    """Execute a mapped tool using its arguments dictionary."""

    if name not in TOOL_MAP:
        raise ValueError(f"Unknown tool: {name}")

    if not isinstance(arguments, dict):
        raise ValueError("Tool arguments must be a dictionary")

    return TOOL_MAP[name](**arguments)


# --------------------------------------------------
# Test helper — find representative database orders
# --------------------------------------------------

def find_test_order(min_amount, max_amount):
    """Find a completed late order within an amount range."""

    with sqlite3.connect(DB_PATH) as connection:
        row = connection.execute(
            """
            SELECT order_id, order_amount, delivery_minutes
            FROM orders
            WHERE status = 'completed'
              AND delivery_minutes > 45
              AND order_amount > ?
              AND order_amount <= ?
            ORDER BY order_id
            LIMIT 1
            """,
            (min_amount, max_amount),
        ).fetchone()

    if row is None:
        raise AssertionError(
            "No matching test order found in SQLite"
        )

    return {
        "order_id": row[0],
        "order_amount": row[1],
        "delivery_minutes": row[2],
    }


def expect_rejected(action, *args):
    """Assert that an invalid refund action is rejected."""

    try:
        action(*args)
    except ValueError:
        return

    raise AssertionError(
        f"{action.__name__} accepted an invalid request"
    )


# --------------------------------------------------
# Q5.9 — Acceptance criteria
# --------------------------------------------------

def validate_tools():
    """Validate the five tools and refund business rules."""

    global LEDGER_PATH

    print("\n=== Part D.Q5: Tool Validation ===")

    # 1. Five tools with descriptions and schemas
    expected_names = {
        "get_order",
        "search_policy",
        "calculate_refund",
        "issue_refund",
        "escalate_to_human",
    }

    assert len(TOOLS_SPEC) == 5
    assert {spec["name"] for spec in TOOLS_SPEC} == expected_names
    assert set(TOOL_MAP) == expected_names

    for spec in TOOLS_SPEC:
        assert spec["description"]
        parameters = spec["parameters"]
        assert parameters["type"] == "object"
        assert parameters["properties"]
        assert set(parameters["required"]) == set(
            parameters["properties"]
        )

    print("PASS | Five tools, descriptions and JSON schemas")
    print("PASS | TOOLS_SPEC and TOOL_MAP match")

    # 2. SQLite lookup
    assert get_order("ORD-999999") == {
        "error": "order not found"
    }

    small = find_test_order(0, 1000)
    large = find_test_order(1000, 1000000)

    order = get_order(small["order_id"])

    assert set(order) == {
        "order_id",
        "status",
        "order_amount",
        "delivery_minutes",
    }

    assert order["status"] == "completed"

    print("PASS | SQLite lookup and missing-order handling")

    # 3. Policy retrieval
    policy = search_policy("refund policy late delivery")

    assert "results" in policy
    assert len(policy["results"]) == 3
    assert all(
        set(chunk) == {"source", "score", "text"}
        for chunk in policy["results"]
    )

    print("PASS | Reused TF-IDF policy retrieval")

    # 4. Refund calculation and boundaries
    assert calculate_refund(500, 46) == {
        "refund_inr": 500.0
    }

    assert calculate_refund(500, 45) == {
        "refund_inr": 0.0
    }

    assert calculate_refund(500, 30) == {
        "refund_inr": 0.0
    }

    assert calculate_refund(1500, 60) == {
        "refund_inr": 1500.0
    }

    print("PASS | Refund calculation and 45-minute boundary")

    # 5. Tool dispatch
    result = execute_tool(
        "calculate_refund",
        {
            "order_amount": 128.25,
            "delivery_minutes": 55,
        },
    )

    assert result == {"refund_inr": 128.25}

    expect_rejected(
        execute_tool,
        "unknown_tool",
        {},
    )

    print("PASS | Tool dispatch using name and arguments")

    # 6. Use a temporary ledger to avoid test side effects
    original_ledger = LEDGER_PATH

    try:
        with tempfile.TemporaryDirectory() as directory:
            LEDGER_PATH = Path(directory) / "refund_ledger.jsonl"

            small_amount = round(float(small["order_amount"]), 2)
            large_amount = round(float(large["order_amount"]), 2)

            # Issue eligible refund
            issued = issue_refund(
                small["order_id"],
                small_amount,
            )

            assert issued == {
                "refund_inr": small_amount
            }

            # Escalate high-value refund
            escalated = escalate_to_human(
                large["order_id"],
                large_amount,
            )

            assert escalated == {
                "refund_inr": large_amount
            }

            print("PASS | Refund issuance and human escalation")

            # Check that invalid actions are blocked
            expect_rejected(
                issue_refund,
                large["order_id"],
                large_amount,
            )

            expect_rejected(
                escalate_to_human,
                small["order_id"],
                small_amount,
            )

            expect_rejected(
                issue_refund,
                small["order_id"],
                1.0,
            )

            expect_rejected(
                issue_refund,
                "ORD-999999",
                100.0,
            )

            print("PASS | High-value and invalid refunds blocked")

            # Repeating the same action must not create
            # another ledger entry.
            issue_refund(
                small["order_id"],
                small_amount,
            )

            records = [
                json.loads(line)
                for line in LEDGER_PATH.read_text(
                    encoding="utf-8"
                ).splitlines()
            ]

            assert len(records) == 2

            assert [record["tool"] for record in records] == [
                "issue_refund",
                "escalate_to_human",
            ]

            for record in records:
                assert record["timestamp"]
                assert record["order_id"]
                assert record["refund_inr"] > 0

            print("PASS | Timestamped JSONL ledger records")
            print("PASS | Duplicate action prevented")

    finally:
        LEDGER_PATH = original_ledger

    print("\nD.Q5 ALL CHECKPOINTS PASSED")


# --------------------------------------------------
# Independent module execution
# --------------------------------------------------

def main():
    print("=== Part D.Q5: Define Tools and Tool Map ===")

    for spec in TOOLS_SPEC:
        print(f"{spec['name']}: {spec['description']}")

    validate_tools()


if __name__ == "__main__":
    main()
