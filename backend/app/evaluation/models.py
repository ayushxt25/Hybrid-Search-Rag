from typing import Self

from pydantic import BaseModel, Field, model_validator


class RetrievalEvaluationCase(BaseModel):
    """One golden retrieval evaluation query."""

    case_id: str
    query: str
    relevant_chunk_ids: list[str] = Field(min_length=1)
    document_id: str | None = Field(default=None, min_length=64, max_length=64)

    @model_validator(mode="after")
    def validate_case(self) -> Self:
        if not self.case_id.strip():
            raise ValueError("case_id cannot be blank.")

        if not self.query.strip():
            raise ValueError("query cannot be blank.")

        if len(set(self.relevant_chunk_ids)) != len(self.relevant_chunk_ids):
            raise ValueError("relevant_chunk_ids must be unique.")

        if any(len(chunk_id) != 64 for chunk_id in self.relevant_chunk_ids):
            raise ValueError("relevant chunk IDs must be exactly 64 characters.")

        return self


class QueryRetrievalEvaluation(BaseModel):
    """Metrics for one query and one retrieval method."""

    case_id: str
    query: str
    relevant_chunk_ids: list[str]
    retrieved_chunk_ids: list[str]
    relevant_retrieved_count: int
    hit: bool
    reciprocal_rank: float
    recall: float


class RetrievalMethodSummary(BaseModel):
    """Aggregate metrics for one retrieval method."""

    method: str
    case_count: int
    hit_rate: float
    mean_reciprocal_rank: float
    mean_recall: float
    evaluations: list[QueryRetrievalEvaluation]


class RetrievalComparisonReport(BaseModel):
    """Dense, sparse, and hybrid retrieval evaluation results."""

    top_k: int
    dense: RetrievalMethodSummary
    sparse: RetrievalMethodSummary
    hybrid: RetrievalMethodSummary
