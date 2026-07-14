from concurrent.futures import ThreadPoolExecutor

import pytest
from pydantic import ValidationError

from app.rate_limit.in_memory import InMemoryFixedWindowRateLimiter
from app.rate_limit.models import RateLimitDecision


class Clock:
    def __init__(self, value: float = 0.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


def test_first_request_allowed() -> None:
    limiter = InMemoryFixedWindowRateLimiter(limit=2, window_seconds=10)

    decision = limiter.check("client")

    assert decision.allowed is True
    assert decision.remaining == 1


def test_remaining_count_decreases_and_limit_request_allowed() -> None:
    limiter = InMemoryFixedWindowRateLimiter(limit=2, window_seconds=10)

    assert limiter.check("client").remaining == 1
    decision = limiter.check("client")

    assert decision.allowed is True
    assert decision.remaining == 0


def test_request_beyond_limit_denied() -> None:
    limiter = InMemoryFixedWindowRateLimiter(limit=1, window_seconds=10)

    assert limiter.check("client").allowed is True
    decision = limiter.check("client")

    assert decision.allowed is False
    assert decision.remaining == 0


def test_reset_occurs_after_window_expiry() -> None:
    clock = Clock()
    limiter = InMemoryFixedWindowRateLimiter(limit=1, window_seconds=10, clock=clock)
    limiter.check("client")

    clock.value = 10
    decision = limiter.check("client")

    assert decision.allowed is True
    assert decision.remaining == 0


def test_reset_after_seconds_uses_ceiling() -> None:
    clock = Clock(2.2)
    limiter = InMemoryFixedWindowRateLimiter(limit=2, window_seconds=10, clock=clock)

    limiter.check("client")
    clock.value = 3.1
    decision = limiter.check("client")

    assert decision.reset_after_seconds == 10


def test_blank_key_rejected() -> None:
    limiter = InMemoryFixedWindowRateLimiter(limit=1, window_seconds=10)

    with pytest.raises(ValueError, match="blank"):
        limiter.check(" ")


@pytest.mark.parametrize(("limit", "window"), [(0, 10), (-1, 10), (1, 0), (1, -1)])
def test_invalid_constructor_values_rejected(limit: int, window: int) -> None:
    with pytest.raises(ValueError):
        InMemoryFixedWindowRateLimiter(limit=limit, window_seconds=window)


def test_separate_keys_have_separate_counters() -> None:
    limiter = InMemoryFixedWindowRateLimiter(limit=1, window_seconds=10)

    assert limiter.check("one").allowed is True
    assert limiter.check("two").allowed is True
    assert limiter.check("one").allowed is False


def test_injected_clock_is_deterministic() -> None:
    clock = Clock()
    limiter = InMemoryFixedWindowRateLimiter(limit=1, window_seconds=5, clock=clock)

    assert limiter.check("client").reset_after_seconds == 5
    clock.value = 4.1

    assert limiter.check("client").reset_after_seconds == 1


def test_concurrent_checks_never_allow_more_than_limit() -> None:
    limiter = InMemoryFixedWindowRateLimiter(limit=5, window_seconds=10)

    with ThreadPoolExecutor(max_workers=20) as executor:
        decisions = list(executor.map(lambda _: limiter.check("client"), range(50)))

    assert sum(decision.allowed for decision in decisions) == 5


def test_stale_expired_entry_resets_lazily() -> None:
    clock = Clock()
    limiter = InMemoryFixedWindowRateLimiter(limit=1, window_seconds=5, clock=clock)
    limiter.check("stale")

    clock.value = 6
    decision = limiter.check("fresh")

    assert decision.allowed is True
    assert limiter._entries == {"fresh": (6, 1)}


def test_decision_model_validation() -> None:
    RateLimitDecision(
        allowed=True,
        limit=1,
        remaining=1,
        reset_after_seconds=0,
    )

    with pytest.raises(ValidationError):
        RateLimitDecision(
            allowed=True,
            limit=1,
            remaining=2,
            reset_after_seconds=0,
        )
