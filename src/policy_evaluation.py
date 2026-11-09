
"""
Part D.Q4 — Evaluate Retrieval and Grounding.

Evaluate all 14 policy questions using:
1. 40-word chunks with 10-word overlap
2. 40-word chunks with zero overlap

Report retrieval accuracy, factual accuracy,
fallback accuracy and chunk-boundary evidence.
"""

import csv
from pathlib import Path

from mock_llm import chat
from src.policy_retrieval import (
    load_documents,
    chunk_documents,
    build_retriever,
    retrieve,
)


ROOT = Path(__file__).resolve().parents[1]
EVAL_PATH = ROOT / "data" / "raw" / "policy_eval.csv"

SYSTEM_PROMPT = (
    "You are a POLICY ASSISTANT for Instamart. "
    "Answer only using the supplied policy context. "
    "Do not invent information or use outside knowledge. "
    "If the information is not available in the context, "
    "say 'I could not find that information in the policy documents.'"
)


# --------------------------------------------------
# Q4.1 — Load all 14 evaluation questions
# --------------------------------------------------
def load_evaluation_data():
    with EVAL_PATH.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 14
    assert sum(
        row["expected_doc"] != "NONE" for row in rows
    ) == 12
    assert sum(
        row["expected_doc"] == "NONE" for row in rows
    ) == 2

    return rows


# --------------------------------------------------
# Q4.2 — Generate answers using retrieved chunks
# --------------------------------------------------
def answer_with_context(question, chunks):
    """Use the same grounded prompt format as D.Q3."""

    context = "\n".join(
        f"[{chunk['source']}]\n{chunk['text']}"
        for chunk in chunks
    )

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": (
                f"CONTEXT:\n{context}"
                f"\nQUESTION:\n{question}"
            ),
        },
    ]

    response = chat(messages)
    return response.choices[0].message.content


# --------------------------------------------------
# Q4.3–Q4.5 — Evaluate retrieval, facts and fallback
# --------------------------------------------------
def evaluate(overlap=10):
    """Evaluate all 14 questions for the selected overlap."""

    if not 0 <= overlap < 40:
        raise ValueError("Overlap must be between 0 and 39")

    documents = load_documents()

    chunks = chunk_documents(
        documents,
        chunk_size=40,
        overlap=overlap,
    )

    vectorizer, chunk_matrix = build_retriever(chunks)

    report = {
        "overlap": overlap,
        "chunk_count": len(chunks),
        "retrieval_hits": 0,
        "fact_hits": 0,
        "fallbacks": 0,
        "details": [],
        "failures": [],
    }

    for row in load_evaluation_data():
        question = row["question"]
        expected_doc = row["expected_doc"]
        expected_fact = row["expected_fact"]

        # Retrieve the top three chunks.
        results = retrieve(
            question,
            k=3,
            chunks=chunks,
            vectorizer=vectorizer,
            chunk_matrix=chunk_matrix,
        )

        sources = [result["source"] for result in results]

        # Generate an answer using these exact chunks.
        answer_text = answer_with_context(question, results)

        answerable = expected_doc != "NONE"

        # Metric 1: Expected document among top three.
        retrieval_hit = (
            answerable and expected_doc in sources
        )

        # Metric 2: Case-sensitive expected fact.
        fact_hit = (
            answerable and expected_fact in answer_text
        )

        # Metric 3: Fallback for unanswerable questions.
        fallback_hit = (
            not answerable
            and answer_text.startswith("I could not find")
        )

        report["retrieval_hits"] += int(retrieval_hit)
        report["fact_hits"] += int(fact_hit)
        report["fallbacks"] += int(fallback_hit)

        detail = {
            "question": question,
            "expected_doc": expected_doc,
            "expected_fact": expected_fact,
            "answer": answer_text,
            "sources": sources,
            "retrieval_hit": retrieval_hit,
            "fact_hit": fact_hit,
            "fallback_hit": fallback_hit,
            "retrieved_chunks": results,
        }

        report["details"].append(detail)

        # Record questions that fail their acceptance checks.
        failed = (
            (answerable and (not retrieval_hit or not fact_hit))
            or (not answerable and not fallback_hit)
        )

        if failed:
            report["failures"].append(detail)

    return report


# --------------------------------------------------
# Q4.6 — Display evidence of zero-overlap failures
# --------------------------------------------------
def show_boundary_evidence(question, overlap=0):
    """Show the relevant policy chunks around a boundary."""

    row = next(
        row for row in load_evaluation_data()
        if row["question"] == question
    )

    documents = load_documents()

    chunks = chunk_documents(
        documents,
        chunk_size=40,
        overlap=overlap,
    )

    doc_chunks = [
        chunk for chunk in chunks
        if chunk["source"] == row["expected_doc"]
    ]

    print(f"\nQuestion: {question}")
    print(f"Expected fact: {row['expected_fact']}")

    # Display adjacent chunks near the expected fact.
    keywords = [
        word.lower()
        for word in row["expected_fact"].split()
        if len(word) > 3
    ]

    for index, chunk in enumerate(doc_chunks):
        if any(
            word in chunk["text"].lower()
            for word in keywords
        ):
            start = max(0, index - 1)
            end = min(len(doc_chunks), index + 2)

            for position in range(start, end):
                print(
                    f"\nChunk {position}:\n"
                    f"{doc_chunks[position]['text']}"
                )
            break


# --------------------------------------------------
# Q4.7 — Print comparison table
# --------------------------------------------------
def compare_results(overlap_report, zero_report):
    print("\n=== D.Q4 Evaluation Results ===")

    print(
        f"{'Metric':<25}"
        f"{'Overlap 10':>15}"
        f"{'Overlap 0':>15}"
    )
    print("-" * 55)

    metrics = [
        ("Retrieval hits", "retrieval_hits", 12),
        ("Fact hits", "fact_hits", 12),
        ("Fallback hits", "fallbacks", 2),
    ]

    for label, key, total in metrics:
        baseline = f"{overlap_report[key]}/{total}"
        zero = f"{zero_report[key]}/{total}"

        print(
            f"{label:<25}"
            f"{baseline:>15}"
            f"{zero:>15}"
        )

    print("\n=== Zero-Overlap Failure Evidence ===")

    if not zero_report["failures"]:
        print("No failures detected.")
        return

    for failure in zero_report["failures"]:
        print(f"\nFAILED QUESTION: {failure['question']}")
        print(f"Expected fact: {failure['expected_fact']}")
        print(f"Actual answer: {failure['answer']}")
        print(f"Retrieved sources: {failure['sources']}")

        if (
            failure["expected_doc"] != "NONE"
            and not failure["fact_hit"]
        ):
            show_boundary_evidence(failure["question"])


# --------------------------------------------------
# Q4.8 — Validate all acceptance criteria
# --------------------------------------------------
def validate_evaluation(overlap_report, zero_report):

    # All 14 questions must be evaluated twice.
    assert len(overlap_report["details"]) == 14
    assert len(zero_report["details"]) == 14

    print("\nPASS | All 14 questions evaluated twice")

    # Standard overlap must match reference scores.
    assert overlap_report["retrieval_hits"] == 12
    assert overlap_report["fact_hits"] == 12
    assert overlap_report["fallbacks"] == 2

    print("PASS | Overlap 10: retrieval 12/12")
    print("PASS | Overlap 10: factual answers 12/12")
    print("PASS | Overlap 10: fallback answers 2/2")

    # Zero overlap must cause at least one factual failure.
    assert zero_report["fact_hits"] < 12

    print("PASS | Zero overlap causes factual failures")
    print("\nD.Q4 ALL CHECKPOINTS PASSED")


# --------------------------------------------------
# Main execution
# --------------------------------------------------
def run_policy_evaluation():

    print("\n=== Part D.Q4: Policy Evaluation ===")

    # Experiment 1: Standard 10-word overlap.
    overlap_report = evaluate(overlap=10)

    # Experiment 2: Zero overlap.
    zero_report = evaluate(overlap=0)

    # Display comparison and failed questions.
    compare_results(overlap_report, zero_report)

    # Validate the assignment's acceptance criteria.
    validate_evaluation(overlap_report, zero_report)

    return overlap_report, zero_report


if __name__ == "__main__":
    run_policy_evaluation()
