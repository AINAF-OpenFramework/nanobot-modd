"""Rate limiting with token bucket algorithm."""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass


@dataclass
class RateLimitConfig:
    max_calls: int = 10
    window_seconds: int = 60
    enabled: bool = True


class RateLimiter:
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self._buckets: dict[str, deque[float]] = defaultdict(deque)
        self._lock = asyncio.Lock()

    async def is_allowed(self, user_id: str) -> bool:
        if not self.config.enabled:
            return True

        async with self._lock:
            now = time.time()
            bucket = self._buckets[user_id]

            while bucket and bucket[0] <= now - self.config.window_seconds:
                bucket.popleft()

            if len(bucket) >= self.config.max_calls:
                return False

            bucket.append(now)
            return True


class RateLimitExceeded(Exception):
    """Raised when request limit is exceeded."""
