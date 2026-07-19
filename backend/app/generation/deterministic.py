import re
from dataclasses import dataclass

from app.generation.base import GenerationProvider
from app.generation.models import GenerationOutput

DETERMINISTIC_MODEL_NAME = "deterministic-acceptance-provider"
INSUFFICIENT_CONTEXT_TEXT = (
    "The provided documents do not contain enough information to answer this question."
)

QUESTION_PATTERN = re.compile(
    r"(?ms)^Question:\n(?P<question>.*?)\n\nDocument context:\n"
)
CONTEXT_PATTERN = re.compile(
    r"(?ms)^Document context:\n(?P<context>.*?)(?:\n\nResponse requirements:|\Z)"
)
SOURCE_PATTERN = re.compile(
    r"(?ms)^\[Source (?P<number>[1-9][0-9]*)\]\n"
    r"(?P<body>.*?)(?=^\[Source [1-9][0-9]*\]\n|\Z)"
)
SENTENCE_PATTERN = re.compile(r"[^.!?\n]+(?:[.!?]|$)")
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
NUMBER_PATTERN = re.compile(r"\b\d+(?:[.,]\d+)?\b")
TEMPORAL_PATTERN = re.compile(
    r"\b(?:every|daily|weekly|monthly|quarterly|annually|yearly|"
    r"day|days|week|weeks|month|months|quarter|quarters|year|years|"
    r"hour|hours|minute|minutes)\b",
    re.IGNORECASE,
)

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "could",
    "do",
    "does",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "should",
    "that",
    "the",
    "this",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


@dataclass(frozen=True)
class CandidateSentence:
    source_number: int
    text: str
    score: float


class DeterministicAcceptanceGenerationProvider(GenerationProvider):
    """Acceptance-only provider that extracts cited evidence without network access."""

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> GenerationOutput:
        answer = self._answer_from_prompt(user_prompt)
        return GenerationOutput(
            text=answer,
            model_name=DETERMINISTIC_MODEL_NAME,
            input_characters=len(system_prompt) + len(user_prompt),
            output_characters=len(answer),
            finish_reason=(
                "insufficient_context"
                if answer == INSUFFICIENT_CONTEXT_TEXT
                else "deterministic"
            ),
        )

    def _answer_from_prompt(self, user_prompt: str) -> str:
        question = self._extract_question(user_prompt)
        context = self._extract_context(user_prompt)
        question_terms = self._important_terms(question)

        if not question_terms or not context.strip():
            return INSUFFICIENT_CONTEXT_TEXT

        candidates = [
            candidate
            for source_number, content in self._iter_source_contents(context)
            for candidate in self._score_source_sentences(
                source_number=source_number,
                content=content,
                question=question,
                question_terms=question_terms,
            )
        ]
        if not candidates:
            return INSUFFICIENT_CONTEXT_TEXT

        best = max(candidates, key=lambda candidate: candidate.score)
        if best.score <= 0:
            return INSUFFICIENT_CONTEXT_TEXT

        return f"{best.text} [Source {best.source_number}]"

    @staticmethod
    def _extract_question(user_prompt: str) -> str:
        match = QUESTION_PATTERN.search(user_prompt)
        return match.group("question").strip() if match else ""

    @staticmethod
    def _extract_context(user_prompt: str) -> str:
        match = CONTEXT_PATTERN.search(user_prompt)
        return match.group("context").strip() if match else ""

    def _iter_source_contents(self, context: str) -> list[tuple[int, str]]:
        sources: list[tuple[int, str]] = []
        for match in SOURCE_PATTERN.finditer(context):
            body = match.group("body").strip()
            source_number = int(match.group("number"))
            sources.append((source_number, self._extract_content(body)))
        return sources

    @staticmethod
    def _extract_content(source_body: str) -> str:
        marker = "\nContent:\n"
        if marker in source_body:
            return source_body.split(marker, 1)[1].strip()

        lines = [
            line
            for line in source_body.splitlines()
            if not line.startswith(("File:", "Chunk ID:", "Heading:", "Page:"))
            and line.strip() != "Content:"
        ]
        return "\n".join(lines).strip()

    def _score_source_sentences(
        self,
        *,
        source_number: int,
        content: str,
        question: str,
        question_terms: set[str],
    ) -> list[CandidateSentence]:
        return [
            CandidateSentence(
                source_number=source_number,
                text=sentence,
                score=self._score_sentence(
                    sentence=sentence,
                    question=question,
                    question_terms=question_terms,
                ),
            )
            for sentence in self._split_sentences(content)
        ]

    @staticmethod
    def _split_sentences(content: str) -> list[str]:
        return [
            " ".join(match.group(0).strip().split())
            for match in SENTENCE_PATTERN.finditer(content)
            if match.group(0).strip()
        ]

    @staticmethod
    def _important_terms(text: str) -> set[str]:
        return {
            token
            for token in TOKEN_PATTERN.findall(text.lower())
            if token not in STOPWORDS and len(token) > 1
        }

    def _score_sentence(
        self,
        *,
        sentence: str,
        question: str,
        question_terms: set[str],
    ) -> float:
        sentence_terms = self._important_terms(sentence)
        overlap = question_terms & sentence_terms
        if not overlap:
            return 0.0

        score = float(len(overlap))
        question_lower = question.lower()
        asks_for_numeric_fact = any(
            phrase in question_lower
            for phrase in (
                "how many",
                "how much",
                "how often",
                "when",
                "what date",
                "what time",
            )
        )
        if asks_for_numeric_fact and NUMBER_PATTERN.search(sentence):
            score += 3.0
        if "how often" in question_lower and TEMPORAL_PATTERN.search(sentence):
            score += 2.0

        return score
