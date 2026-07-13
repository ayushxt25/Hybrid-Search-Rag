from collections.abc import Sequence

from app.evaluation.metrics import evaluate_ranked_chunk_ids, summarize_method
from app.evaluation.models import RetrievalComparisonReport, RetrievalEvaluationCase
from app.schemas.search_request import (
    DenseSearchRequest,
    HybridSearchRequest,
    SparseSearchRequest,
)
from app.services.dense_search import DenseSearchService
from app.services.hybrid_search import HybridSearchService
from app.services.sparse_search import SparseSearchService


class RetrievalEvaluationRunner:
    """Run dense, sparse, and hybrid retrieval against golden cases."""

    def __init__(
        self,
        *,
        dense_search_service: DenseSearchService,
        sparse_search_service: SparseSearchService,
        hybrid_search_service: HybridSearchService,
    ) -> None:
        self.dense_search_service = dense_search_service
        self.sparse_search_service = sparse_search_service
        self.hybrid_search_service = hybrid_search_service

    def run(
        self,
        cases: Sequence[RetrievalEvaluationCase],
        *,
        top_k: int = 5,
        candidate_limit: int = 20,
    ) -> RetrievalComparisonReport:
        """Evaluate all retrieval methods for the supplied cases."""
        if not cases:
            raise ValueError("cases cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        if candidate_limit < top_k:
            raise ValueError("candidate_limit must be greater than or equal to top_k.")

        case_ids = [case.case_id for case in cases]

        if len(set(case_ids)) != len(case_ids):
            raise ValueError("case_id values must be unique.")

        dense_evaluations = []
        sparse_evaluations = []
        hybrid_evaluations = []

        for case in cases:
            dense_response = self.dense_search_service.search(
                DenseSearchRequest(
                    query=case.query,
                    limit=top_k,
                    document_id=case.document_id,
                )
            )
            sparse_response = self.sparse_search_service.search(
                SparseSearchRequest(
                    query=case.query,
                    limit=top_k,
                    document_id=case.document_id,
                )
            )
            hybrid_response = self.hybrid_search_service.search(
                HybridSearchRequest(
                    query=case.query,
                    limit=top_k,
                    candidate_limit=candidate_limit,
                    document_id=case.document_id,
                )
            )

            dense_evaluations.append(
                evaluate_ranked_chunk_ids(
                    case=case,
                    retrieved_chunk_ids=[
                        result.chunk_id for result in dense_response.results
                    ],
                )
            )
            sparse_evaluations.append(
                evaluate_ranked_chunk_ids(
                    case=case,
                    retrieved_chunk_ids=[
                        result.chunk_id for result in sparse_response.results
                    ],
                )
            )
            hybrid_evaluations.append(
                evaluate_ranked_chunk_ids(
                    case=case,
                    retrieved_chunk_ids=[
                        result.chunk_id for result in hybrid_response.results
                    ],
                )
            )

        return RetrievalComparisonReport(
            top_k=top_k,
            dense=summarize_method(
                method="Dense",
                evaluations=dense_evaluations,
            ),
            sparse=summarize_method(
                method="Sparse",
                evaluations=sparse_evaluations,
            ),
            hybrid=summarize_method(
                method="Hybrid",
                evaluations=hybrid_evaluations,
            ),
        )
