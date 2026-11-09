
"""Part C.Q3 — Zero-shot and few-shot classification prompts."""

import json


ALLOWED_CATEGORIES = {
    "billing",
    "delivery",
    "technical",
    "fraud",
}


# C.Q3: Zero-shot system prompt — no examples.
ZERO_SHOT_PROMPT = """
CLASSIFY customer support tickets into exactly one category:
billing, delivery, technical, fraud.

Return JSON only, in this format:
{"category": "...", "reason": "..."}

The category must be one of the four allowed categories.
Do not include Markdown or additional text.
""".strip()


# C.Q3: Few-shot system prompt — includes labelled examples.
FEW_SHOT_PROMPT = """
CLASSIFY customer support tickets into exactly one category:
billing, delivery, technical, fraud.

Return JSON only, in this format:
{"category": "...", "reason": "..."}

The category must be one of the four allowed categories.
Do not include Markdown or additional text.

EXAMPLES:

Ticket: My account was hacked and someone placed an unauthorized order.
Output: {"category": "fraud", "reason": "Unauthorized account activity"}

Ticket: The delivery rider arrived very late.
Output: {"category": "delivery", "reason": "Late delivery"}

Ticket: I was charged twice for my purchase.
Output: {"category": "billing", "reason": "Duplicate charge"}
""".strip()


def validate_ticket_prompts() -> None:
    """Validate C.Q3 against assignment requirements."""

    for name, prompt in {
        "zero-shot": ZERO_SHOT_PROMPT,
        "few-shot": FEW_SHOT_PROMPT,
    }.items():
        assert "CLASSIFY" in prompt, (
            f"{name}: missing CLASSIFY marker"
        )

        for category in ALLOWED_CATEGORIES:
            assert category in prompt, (
                f"{name}: missing category {category}"
            )

        assert '"category"' in prompt
        assert '"reason"' in prompt
        assert "JSON only" in prompt

    assert "EXAMPLES:" not in ZERO_SHOT_PROMPT
    assert "EXAMPLES:" in FEW_SHOT_PROMPT

    examples_section = FEW_SHOT_PROMPT.split("EXAMPLES:", 1)[1]

    example_outputs = [
        line.removeprefix("Output: ").strip()
        for line in examples_section.splitlines()
        if line.startswith("Output: ")
    ]

    assert len(example_outputs) >= 2, (
        "Few-shot prompt needs at least two labelled examples"
    )

    categories = set()

    for output in example_outputs:
        parsed = json.loads(output)

        assert set(parsed) == {"category", "reason"}
        assert parsed["category"] in ALLOWED_CATEGORIES

        categories.add(parsed["category"])

    assert "fraud" in categories, (
        "Few-shot prompt must include a fraud example"
    )

    print("PASS | Both prompts contain CLASSIFY")
    print("PASS | Both prompts list all four categories")
    print("PASS | Both prompts require JSON-only output")
    print("PASS | Zero-shot prompt contains no EXAMPLES marker")
    print("PASS | Few-shot prompt has labelled examples")
    print("PASS | Few-shot prompt includes fraud")
    print("\nC.Q3 ALL CHECKPOINTS PASSED")


def run_ticket_prompts() -> dict:
    """Prepare and validate the two system prompts."""

    print("\n=== Part C.Q3: Classification Prompts ===")

    validate_ticket_prompts()

    return {
        "zero_shot": ZERO_SHOT_PROMPT,
        "few_shot": FEW_SHOT_PROMPT,
    }


if __name__ == "__main__":
    run_ticket_prompts()
