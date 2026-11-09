
"""
Part D.Q3 — Grounded Policy Assistant.

Retrieves three policy chunks and answers the question
using only the supplied policy context.
"""

from mock_llm import chat
from src.policy_retrieval import retrieve


def answer(question):
    """Return the mock LLM's answer and retrieved source filenames."""

    # Step 1: Retrieve the three most relevant policy chunks.
    chunks = retrieve(question, k=3)

    # Step 2: Define the system instruction.
    system_message = {
        "role": "system",
        "content": (
            "You are a POLICY ASSISTANT for Instamart. "
            "Answer only using the supplied policy context. "
            "Do not invent information or use outside knowledge. "
            "If the information is not available in the context, "
            "say 'I could not find that information in the policy documents.'"
        ),
    }

    # Step 3: Build the context with source-labelled chunks.
    context = "\n".join(
        f"[{chunk['source']}]\n{chunk['text']}"
        for chunk in chunks
    )

    # Step 4: Build the user message in the required format.
    user_message = {
        "role": "user",
        "content": (
            f"CONTEXT:\n{context}"
            f"\nQUESTION:\n{question}"
        ),
    }

    # Step 5: Send exactly two messages to the supplied mock LLM.
    messages = [system_message, user_message]
    response = chat(messages)

    # Step 6: Extract the model's answer and source filenames.
    answer_text = response.choices[0].message.content
    sources = [chunk["source"] for chunk in chunks]

    return answer_text, sources


def validate_answer(question):
    """Validate all D.Q3 prompt and return requirements."""

    chunks = retrieve(question, k=3)
    answer_text, sources = answer(question)

    assert len(chunks) == 3
    assert len(sources) == 3

    assert isinstance(answer_text, str)
    assert answer_text.strip()

    assert sources == [chunk["source"] for chunk in chunks]

    # Reconstruct the required user message to verify its format.
    context = "\n".join(
        f"[{chunk['source']}]\n{chunk['text']}"
        for chunk in chunks
    )

    expected_prompt = (
        f"CONTEXT:\n{context}"
        f"\nQUESTION:\n{question}"
    )

    assert expected_prompt.startswith("CONTEXT:\n")
    assert expected_prompt.endswith(f"\nQUESTION:\n{question}")

    print("PASS | Three policy chunks retrieved")
    print("PASS | Mock LLM returned an answer")
    print("PASS | Three source filenames returned")
    print("PASS | Required prompt format constructed")
    print("\nD.Q3 ALL CHECKPOINTS PASSED")


def run_policy_assistant():
    print("\n=== Part D.Q3: Grounded Policy Assistant ===")

    question = "Who must approve a refund above INR 1000?"

    answer_text, sources = answer(question)

    print(f"\nQuestion: {question}")
    print(f"Answer: {answer_text}")
    print(f"Sources: {sources}")

    validate_answer(question)


if __name__ == "__main__":
    run_policy_assistant()
