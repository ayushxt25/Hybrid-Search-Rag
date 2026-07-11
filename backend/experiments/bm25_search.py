import argparse
import math
import re
from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
    document_id: str
    text: str


def tokenize(text: str) -> list[str]:
    """
    Convert text into normalized searchable tokens.

    This tokenizer:
    - converts text to lowercase,
    - preserves letters, numbers, underscores and hyphens,
    - removes other punctuation.
    """
    return re.findall(r"[a-z0-9_-]+", text.lower())


class BM25:
    def __init__(
        self,
        documents: list[Document],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if not documents:
            raise ValueError("At least one document is required.")

        self.documents = documents
        self.k1 = k1
        self.b = b

        self.tokenized_documents = [
            tokenize(document.text) for document in documents
        ]

        self.document_lengths = [
            len(tokens) for tokens in self.tokenized_documents
        ]

        self.average_document_length = (
            sum(self.document_lengths) / len(self.document_lengths)
        )

        self.term_frequencies = [
            Counter(tokens) for tokens in self.tokenized_documents
        ]

        self.document_frequencies = self._calculate_document_frequencies()

    def _calculate_document_frequencies(self) -> Counter[str]:
        document_frequencies: Counter[str] = Counter()

        for tokens in self.tokenized_documents:
            for term in set(tokens):
                document_frequencies[term] += 1

        return document_frequencies

    def _inverse_document_frequency(self, term: str) -> float:
        total_documents = len(self.documents)
        documents_with_term = self.document_frequencies.get(term, 0)

        return math.log(
            1
            + (
                total_documents
                - documents_with_term
                + 0.5
            )
            / (
                documents_with_term
                + 0.5
            )
        )

    def score_document(
        self,
        query_tokens: list[str],
        document_index: int,
    ) -> float:
        term_frequency = self.term_frequencies[document_index]
        document_length = self.document_lengths[document_index]

        score = 0.0

        for term in query_tokens:
            frequency = term_frequency.get(term, 0)

            if frequency == 0:
                continue

            inverse_document_frequency = (
                self._inverse_document_frequency(term)
            )

            numerator = frequency * (self.k1 + 1)

            denominator = frequency + self.k1 * (
                1
                - self.b
                + self.b
                * document_length
                / self.average_document_length
            )

            score += (
                inverse_document_frequency
                * numerator
                / denominator
            )

        return score

    def search(
        self,
        query: str,
    ) -> list[tuple[Document, float]]:
        query_tokens = tokenize(query)
        ranked_results: list[tuple[Document, float]] = []

        for index, document in enumerate(self.documents):
            score = self.score_document(
                query_tokens=query_tokens,
                document_index=index,
            )
            ranked_results.append((document, score))

        ranked_results.sort(
            key=lambda result: result[1],
            reverse=True,
        )

        return ranked_results


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank sample documents using BM25 keyword search."
    )

    parser.add_argument(
        "--query",
        type=str,
        default="ERR_AUTH_401",
        help="Keyword query to search for.",
    )

    return parser.parse_args()


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

    search_engine = BM25(documents)
    ranked_results = search_engine.search(args.query)

    print(f"Query: {args.query}")
    print("\nRanked results:")

    for position, (document, score) in enumerate(
        ranked_results,
        start=1,
    ):
        print(
            f"\n{position}. {document.document_id}"
            f"\n   Score: {score:.4f}"
            f"\n   Text: {document.text}"
        )


if __name__ == "__main__":
    main()