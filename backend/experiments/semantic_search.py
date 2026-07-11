import argparse
from dataclasses import dataclass

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim


@dataclass(frozen=True)
class Document:
    document_id: str
    text: str


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the semantic search experiment."""
    parser = argparse.ArgumentParser(
        description="Rank sample documents using dense semantic search."
    )
    parser.add_argument(
        "--query",
        type=str,
        default="Can staff members work from home?",
        help="Question or search query to compare with the sample documents.",
    )
    return parser.parse_args()


def main() -> None:
    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    documents = [
        Document(
            document_id="remote_work_policy",
            text="Employees may work remotely for up to three days per week.",
        ),
        Document(
            document_id="leave_policy",
            text="Employees receive eighteen paid leave days every calendar year.",
        ),
        Document(
            document_id="travel_policy",
            text="Hotel expenses are reimbursed up to the approved daily limit.",
        ),
        Document(
            document_id="security_policy",
            text="All employees must enable multi-factor authentication.",
        ),
        Document(
            document_id="sports_news",
            text="The football team won its final match of the season.",
        ),
    ]

    args = parse_arguments()
    query = args.query

    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    document_texts = [document.text for document in documents]

    document_embeddings = model.encode(
        document_texts,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )

    query_embedding = model.encode(
        query,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )

    similarity_scores = cos_sim(
        query_embedding,
        document_embeddings,
    )[0]

    ranked_results = sorted(
        zip(documents, similarity_scores.tolist()),
        key=lambda result: result[1],
        reverse=True,
    )

    print(f"\nQuery: {query}")
    print(f"Embedding dimensions: {query_embedding.shape[0]}")
    print("\nRanked results:")

    for position, (document, score) in enumerate(ranked_results, start=1):
        print(
            f"\n{position}. {document.document_id}"
            f"\n   Score: {score:.4f}"
            f"\n   Text: {document.text}"
        )


if __name__ == "__main__":
    main()