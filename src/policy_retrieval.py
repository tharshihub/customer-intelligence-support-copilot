
"""
Part D.Q1–Q2 — Policy Document Chunking and TF-IDF Retrieval.

D.Q1: Split five policy documents into 40-word chunks
      with 10-word overlap.

D.Q2: Retrieve relevant chunks using TF-IDF and
      cosine similarity.
"""

from collections import Counter
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------
# Configuration
# --------------------------------------------------
POLICY_DIR = Path(__file__).resolve().parents[1] / "data" / "policies"

CHUNK_SIZE = 40
CHUNK_OVERLAP = 10

EXPECTED_CHUNKS = {
    "refund_policy.txt": 4,
    "delivery_policy.txt": 3,
    "escalation_policy.txt": 3,
    "payment_policy.txt": 3,
    "privacy_policy.txt": 2,
}


# --------------------------------------------------
# D.Q1 — Load policy documents
# --------------------------------------------------
def load_documents(policy_dir=POLICY_DIR):
    """Read policy text files and retain source filenames."""

    documents = []

    for file_path in sorted(Path(policy_dir).glob("*.txt")):
        documents.append({
            "source": file_path.name,
            "text": file_path.read_text(encoding="utf-8"),
        })

    return documents


# --------------------------------------------------
# D.Q1 — Split text into overlapping word chunks
# --------------------------------------------------
def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Create word chunks with the specified overlap."""

    if chunk_size <= 0 or not 0 <= overlap < chunk_size:
        raise ValueError("Require 0 <= overlap < chunk_size")

    words = text.split()
    chunks = []
    step = chunk_size - overlap

    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + chunk_size]))

        # Stop once this chunk reaches the document end.
        if start + chunk_size >= len(words):
            break

    return chunks


# --------------------------------------------------
# D.Q1 — Chunk documents and preserve source
# --------------------------------------------------
def chunk_documents(
    documents,
    chunk_size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP,
):
    """Return chunks with source, chunk_id and text."""

    all_chunks = []

    for document in documents:
        pieces = chunk_text(
            document["text"],
            chunk_size=chunk_size,
            overlap=overlap,
        )

        for index, piece in enumerate(pieces):
            all_chunks.append({
                "source": document["source"],
                "chunk_id": index,
                "text": piece,
            })

    return all_chunks


# --------------------------------------------------
# D.Q1 — Validate chunking acceptance criteria
# --------------------------------------------------
def validate_policy_chunks(documents, chunks):
    """Validate the five policies and 15 reference chunks."""

    assert len(documents) == 5, (
        f"Expected 5 documents, got {len(documents)}"
    )

    sources = {doc["source"] for doc in documents}

    assert sources == set(EXPECTED_CHUNKS), (
        f"Unexpected policy files: {sources}"
    )

    counts = Counter(chunk["source"] for chunk in chunks)

    assert len(chunks) == 15, (
        f"Expected 15 chunks, got {len(chunks)}"
    )

    assert counts == EXPECTED_CHUNKS, (
        f"Unexpected chunk counts: {dict(counts)}"
    )

    for document in documents:
        words = document["text"].split()

        doc_chunks = [
            chunk for chunk in chunks
            if chunk["source"] == document["source"]
        ]

        for index, chunk in enumerate(doc_chunks):
            start = index * (CHUNK_SIZE - CHUNK_OVERLAP)

            expected_words = words[start:start + CHUNK_SIZE]
            actual_words = chunk["text"].split()

            assert chunk["chunk_id"] == index

            assert actual_words == expected_words, (
                f"Incorrect chunk: {document['source']} "
                f"index {index}"
            )

            if index > 0:
                previous_words = doc_chunks[index - 1]["text"].split()

                assert (
                    previous_words[-CHUNK_OVERLAP:]
                    == actual_words[:CHUNK_OVERLAP]
                ), (
                    f"Incorrect overlap: {document['source']} "
                    f"index {index}"
                )

        assert doc_chunks[-1]["text"].split()[-1] == words[-1], (
            f"Final words missing: {document['source']}"
        )

    print("PASS | Five policy documents loaded")
    print("PASS | Total chunks: 15")
    print("PASS | Expected chunk distribution")
    print("PASS | 40-word boundaries and 10-word overlap")
    print("PASS | Source filenames preserved")
    print("\nD.Q1 ALL CHECKPOINTS PASSED")


# --------------------------------------------------
# D.Q2 — Fit TF-IDF on chunk texts
# --------------------------------------------------
def build_retriever(chunks):
    """Fit the TF-IDF vectorizer on policy chunks."""

    vectorizer = TfidfVectorizer()

    chunk_matrix = vectorizer.fit_transform(
        [chunk["text"] for chunk in chunks]
    )

    return vectorizer, chunk_matrix


# --------------------------------------------------
# D.Q2 — Retrieve relevant policy chunks
# --------------------------------------------------
def retrieve(
    query,
    k=3,
    chunks=None,
    vectorizer=None,
    chunk_matrix=None,
):
    """
    Return top-k chunks ranked by cosine similarity.

    Each result contains:
        source: policy filename
        score: cosine similarity score
        text: chunk content
    """

    # Build the index if one was not provided.
    if chunks is None:
        documents = load_documents()
        chunks = chunk_documents(documents)

    if vectorizer is None or chunk_matrix is None:
        vectorizer, chunk_matrix = build_retriever(chunks)

    if not 1 <= k <= len(chunks):
        raise ValueError(f"k must be between 1 and {len(chunks)}")

    # Use the same fitted vectorizer for the query.
    query_vector = vectorizer.transform([query])

    # Compare the query with all policy chunks.
    scores = cosine_similarity(
        query_vector,
        chunk_matrix,
    )[0]

    # Sort by descending similarity and select top k.
    ranked_indices = scores.argsort()[::-1][:k]

    return [
        {
            "source": chunks[index]["source"],
            "score": float(scores[index]),
            "text": chunks[index]["text"],
        }
        for index in ranked_indices
    ]


# --------------------------------------------------
# D.Q2 — Validate retrieval acceptance criteria
# --------------------------------------------------
def validate_retrieval(results):
    """Check D.Q2 result structure and reference query."""

    assert len(results) == 3, (
        f"Expected 3 results, got {len(results)}"
    )

    assert all(
        set(result.keys()) == {"source", "score", "text"}
        for result in results
    ), "Results must contain source, score and text"

    assert all(
        results[i]["score"] >= results[i + 1]["score"]
        for i in range(len(results) - 1)
    ), "Results must be sorted by descending similarity"

    assert all(
        result["source"] == "refund_policy.txt"
        for result in results
    ), "Reference query must return three refund policy chunks"

    print("PASS | TF-IDF retrieval returns top 3 chunks")
    print("PASS | Results contain source, score and text")
    print("PASS | Results ranked by cosine similarity")
    print("PASS | All top 3 chunks from refund_policy.txt")
    print("\nD.Q2 ALL CHECKPOINTS PASSED")


# --------------------------------------------------
# Independent execution — D.Q1 and D.Q2
# --------------------------------------------------
def run_policy_retrieval():
    print("\n=== Part D.Q1: Policy Document Chunking ===")

    documents = load_documents()
    chunks = chunk_documents(documents)

    counts = Counter(chunk["source"] for chunk in chunks)

    for source in sorted(counts):
        print(f"{source}: {counts[source]} chunks")

    validate_policy_chunks(documents, chunks)

    print("\n=== Part D.Q2: TF-IDF Policy Retrieval ===")

    vectorizer, chunk_matrix = build_retriever(chunks)

    query = "refund policy late delivery"

    results = retrieve(
        query,
        k=3,
        chunks=chunks,
        vectorizer=vectorizer,
        chunk_matrix=chunk_matrix,
    )

    print(f"\nQuery: {query}")

    for index, result in enumerate(results, start=1):
        print(
            f"{index}. {result['source']} "
            f"| similarity={result['score']:.4f}"
        )

    validate_retrieval(results)

    return results


if __name__ == "__main__":
    run_policy_retrieval()
