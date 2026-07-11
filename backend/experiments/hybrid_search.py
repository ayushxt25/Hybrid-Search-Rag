import argparse
from dataclasses import dataclass

from bm25_search import BM25
from bm25_search import Document as BM25Document
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim


@dataclass(frozen=True)
class Document:
    document_id: str
    text: str


def reciprocal_rank_fusion(
    rankings: list[list[str]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """
    Combine multiple ranked document lists using Reciprocal Rank Fusion.

    Each document receives:
        1 / (k + rank)

    from every ranking in which it appears.
    """
    if k <= 0:
        raise ValueError("RRF constant k must be greater than zero.")

    fused_scores: dict[str, float] = {}

    for ranking in rankings:
        for rank, document_id in enumerate(ranking, start=1):
            current_score = fused_scores.get(document_id, 0.0)
            fused_scores[document_id] = current_score + 1 / (k + rank)

    return sorted(
        fused_scores.items(),
        key=lambda result: result[1],
        reverse=True,
    )


def run_dense_search(
    query: str,
    documents: list[Document],
    model: SentenceTransformer,
) -> list[tuple[str, float]]:
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
    )[0].tolist()

    ranked_results = sorted(
        zip(documents, similarity_scores),
        key=lambda result: result[1],
        reverse=True,
    )

    return [
        (document.document_id, score)
        for document, score in ranked_results
    ]


def run_sparse_search(
    query: str,
    documents: list[Document],
) -> list[tuple[str, float]]:
    sparse_documents = [
        BM25Document(
            document_id=document.document_id,
            text=document.text,
        )
        for document in documents
    ]

    search_engine = BM25(sparse_documents)
    ranked_results = search_engine.search(query)

    return [
        (document.document_id, score)
        for document, score in ranked_results
    ]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare dense, sparse and hybrid document retrieval."
    )

    parser.add_argument(
        "--query",
        type=str,
        default="Can employees work from home?",
        help="Query to search for.",
    )

    return parser.parse_args()


def print_results(
    title: str,
    results: list[tuple[str, float]],
) -> None:
    print(f"\n{title}")

    for position, (document_id, score) in enumerate(results, start=1):
        print(f"{position}. {document_id}: {score:.6f}")


def main() -> None:
    documents = [
        Document(
            document_id="authentication_error",
            text=(
                "Error code ERR_AUTH_401 occurs when an expired "
                "access token is used."
            ),
        ),
        Document(
            document_id="remote_work_policy",
            text=(
                "Employees may work remotely for up to "
                "three days per week."
            ),
        ),
        Document(
            document_id="leave_policy",
            text=(
                "Employees receive eighteen paid leave days "
                "every calendar year."
            ),
        ),
        Document(
            document_id="travel_policy",
            text=(
                "Hotel expenses are reimbursed up to the "
                "approved daily limit."
            ),
        ),
        Document(
            document_id="security_policy",
            text=(
                "All employees must enable multi-factor "
                "authentication."
            ),
        ),
    ]

    args = parse_arguments()

    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    dense_results = run_dense_search(
        query=args.query,
        documents=documents,
        model=model,
    )

    sparse_results = run_sparse_search(
        query=args.query,
        documents=documents,
    )

    dense_ranking = [
        document_id for document_id, _ in dense_results
    ]

    sparse_ranking = [
        document_id for document_id, _ in sparse_results
    ]

    hybrid_results = reciprocal_rank_fusion(
        rankings=[
            dense_ranking,
            sparse_ranking,
        ],
        k=60,
    )

    print(f"\nQuery: {args.query}")

    print_results(
        title="Dense semantic ranking",
        results=dense_results,
    )

    print_results(
        title="Sparse BM25 ranking",
        results=sparse_results,
    )

    print_results(
        title="Hybrid RRF ranking",
        results=hybrid_results,
    )


if __name__ == "__main__":
    main()