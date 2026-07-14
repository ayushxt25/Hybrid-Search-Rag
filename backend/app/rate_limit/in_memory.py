import time
from collections.abc import Callable
from math import ceil
from threading import Lock

from app.rate_limit.models import RateLimitDecision


class InMemoryFixedWindowRateLimiter:
    def __init__(
        self,
        limit: int,
        window_seconds: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if limit <= 0:
            raise ValueError("limit must be greater than zero.")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be greater than zero.")

        self.limit = limit
        self.window_seconds = window_seconds
        self._clock = clock
        self._entries: dict[str, tuple[float, int]] = {}
        self._lock = Lock()

    def check(self, key: str) -> RateLimitDecision:
        if not key.strip():
            raise ValueError("rate limit key cannot be blank.")

        now = self._clock()
        with self._lock:
            self._remove_expired_entries(now)
            window_start, count = self._entries.get(key, (now, 0))
            elapsed = now - window_start

            if elapsed >= self.window_seconds:
                window_start = now
                count = 0

            reset_after_seconds = ceil(self.window_seconds - (now - window_start))
            if count >= self.limit:
                self._entries[key] = (window_start, count)
                return RateLimitDecision(
                    allowed=False,
                    limit=self.limit,
                    remaining=0,
                    reset_after_seconds=reset_after_seconds,
                )

            count += 1
            self._entries[key] = (window_start, count)
            return RateLimitDecision(
                allowed=True,
                limit=self.limit,
                remaining=self.limit - count,
                reset_after_seconds=reset_after_seconds,
            )

    def _remove_expired_entries(self, now: float) -> None:
        expired_keys = [
            key
            for key, (window_start, _) in self._entries.items()
            if now - window_start >= self.window_seconds
        ]
        for key in expired_keys:
            del self._entries[key]
