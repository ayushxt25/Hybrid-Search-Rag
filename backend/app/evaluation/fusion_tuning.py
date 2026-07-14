from collections.abc import Sequence
from math import isfinite

from pydantic import BaseModel, model_validator

from app.evaluation.metrics import evaluate_ranked_chunk_ids
from app.evaluation.models import QueryRetrievalEvaluation, RetrievalEvaluationCase
from app.retrieval.rrf import reciprocal_rank_fusion
from app.schemas.search_request import DenseSearchRequest, SparseSearchRequest
from app.services.dense_search import DenseSearchService
from app.services.sparse_search import SparseSearchService


class FusionWeightConfiguration(BaseModel):
    """Dense and sparse weights for one weighted RRF trial."""

    name: str
    dense_weight: float
    sparse_weight: float

    @model_validator(mode="after")
    def validate_configuration(self) -> "FusionWeightConfiguration":
        if not self.name.strip():
            raise ValueError("configuration name cannot be blank.")

        if not isfinite(self.dense_weight) or not isfinite(self.sparse_weight):
            raise ValueError("weights must be finite.")

        if self.dense_weight <= 0 or self.sparse_weight <= 0:
            raise ValueError("weights must be greater than zero.")

        return self


class FusionWeightSummary(BaseModel):
    """Aggregate metrics for one weighted RRF configuration."""

    name: str
    dense_weight: float
    sparse_weight: float
    hit_rate: float
    mean_reciprocal_rank: float
    mean_recall: float
    evaluations: list[QueryRetrievalEvaluation]


class FusionWeightComparisonReport(BaseModel):
    """Comparison report for multiple weighted RRF configurations."""

    top_k: int
    candidate_limit: int
    document_filter_applied: bool
    configurations: list[FusionWeightSummary]


class FusionWeightEvaluationRunner:
    """Evaluate weighted dense/sparse fusion using cached candidates per case."""

    def __init__(
        self,
        *,
        dense_search_service: DenseSearchService,
        sparse_search_service: SparseSearchService,
    ) -> None:
        self.dense_search_service = dense_search_service
        self.sparse_search_service = sparse_search_service

    def run(
        self,
        cases: Sequence[RetrievalEvaluationCase],
        *,
        configurations: Sequence[FusionWeightConfiguration],
        top_k: int = 1,
        candidate_limit: int = 6,
        use_document_filter: bool = False,
        rrf_k: int = 60,
    ) -> FusionWeightComparisonReport:
        """Compare weighted fusion configurations against golden cases."""
        if not cases:
            raise ValueError("cases cannot be empty.")

        if not configurations:
            raise ValueError("configurations cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        if candidate_limit < top_k:
            raise ValueError("candidate_limit must be greater than or equal to top_k.")

        if rrf_k <= 0:
            raise ValueError("rrf_k must be greater than zero.")

        case_ids = [case.case_id for case in cases]

        if len(set(case_ids)) != len(case_ids):
            raise ValueError("case_id values must be unique.")

        configuration_names = [configuration.name for configuration in configurations]

        if len(set(configuration_names)) != len(configuration_names):
            raise ValueError("configuration names must be unique.")

        evaluations_by_configuration = {
            configuration.name: [] for configuration in configurations
        }

        for case in cases:
            document_id = case.document_id if use_document_filter else None
            dense_response = self.dense_search_service.search(
                DenseSearchRequest(
                    query=case.query,
                    limit=candidate_limit,
                    document_id=document_id,
                )
            )
            sparse_response = self.sparse_search_service.search(
                SparseSearchRequest(
                    query=case.query,
                    limit=candidate_limit,
                    document_id=document_id,
                )
            )

            for configuration in configurations:
                fused_results = reciprocal_rank_fusion(
                    [dense_response.results, sparse_response.results],
                    weights=[
                        configuration.dense_weight,
                        configuration.sparse_weight,
                    ],
                    limit=top_k,
                    k=rrf_k,
                )
                evaluations_by_configuration[configuration.name].append(
                    evaluate_ranked_chunk_ids(
                        case=case,
                        retrieved_chunk_ids=[
                            result.chunk_id for result in fused_results
                        ],
                    )
                )

        summaries = [
            self._summarize_configuration(
                configuration=configuration,
                evaluations=evaluations_by_configuration[configuration.name],
            )
            for configuration in configurations
        ]

        return FusionWeightComparisonReport(
            top_k=top_k,
            candidate_limit=candidate_limit,
            document_filter_applied=use_document_filter,
            configurations=summaries,
        )

    def _summarize_configuration(
        self,
        *,
        configuration: FusionWeightConfiguration,
        evaluations: Sequence[QueryRetrievalEvaluation],
    ) -> FusionWeightSummary:
        evaluation_list = list(evaluations)
        case_count = len(evaluation_list)

        return FusionWeightSummary(
            name=configuration.name,
            dense_weight=configuration.dense_weight,
            sparse_weight=configuration.sparse_weight,
            hit_rate=sum(1 for evaluation in evaluation_list if evaluation.hit)
            / case_count,
            mean_reciprocal_rank=sum(
                evaluation.reciprocal_rank for evaluation in evaluation_list
            )
            / case_count,
            mean_recall=sum(evaluation.recall for evaluation in evaluation_list)
            / case_count,
            evaluations=evaluation_list,
        )
