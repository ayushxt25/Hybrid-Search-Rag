import math
from typing import Sequence


def dot_product(vector_a: Sequence[float], vector_b: Sequence[float]) -> float:
    """Return the dot product of two vectors."""
    if len(vector_a) != len(vector_b):
        raise ValueError("Vectors must have the same number of dimensions.")

    return sum(a * b for a, b in zip(vector_a, vector_b))


def magnitude(vector: Sequence[float]) -> float:
    """Return the Euclidean magnitude of a vector."""
    return math.sqrt(sum(value**2 for value in vector))


def cosine_similarity(
    vector_a: Sequence[float],
    vector_b: Sequence[float],
) -> float:
    """Calculate cosine similarity between two vectors."""
    denominator = magnitude(vector_a) * magnitude(vector_b)

    if denominator == 0:
        raise ValueError("Cosine similarity is undefined for a zero vector.")

    return dot_product(vector_a, vector_b) / denominator


def main() -> None:
    query_vector = [0.95, 0.75, 0.05]

    document_vectors = {
        "remote_work_policy": [0.90, 0.80, 0.10],
        "leave_policy": [0.70, 0.10, 0.20],
        "football_news": [0.10, 0.00, 0.90],
    }

    ranked_results: list[tuple[str, float]] = []

    for document_name, document_vector in document_vectors.items():
        score = cosine_similarity(query_vector, document_vector)
        ranked_results.append((document_name, score))

    ranked_results.sort(key=lambda result: result[1], reverse=True)

    print("Query vector:", query_vector)
    print("\nRanked documents:")

    for position, (document_name, score) in enumerate(ranked_results, start=1):
        print(f"{position}. {document_name}: {score:.4f}")


if __name__ == "__main__":
    main()