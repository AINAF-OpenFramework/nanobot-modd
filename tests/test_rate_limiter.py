import asyncio

import pytest

from nanobot.middleware.rate_limiter import RateLimitConfig, RateLimiter


@pytest.mark.asyncio
async def test_allows_within_limit():
    limiter = RateLimiter(RateLimitConfig(max_calls=3))
    assert await limiter.is_allowed("user1") is True
    assert await limiter.is_allowed("user1") is True
    assert await limiter.is_allowed("user1") is True


@pytest.mark.asyncio
async def test_blocks_over_limit():
    limiter = RateLimiter(RateLimitConfig(max_calls=2))
    await limiter.is_allowed("user1")
    await limiter.is_allowed("user1")
    assert await limiter.is_allowed("user1") is False


@pytest.mark.asyncio
async def test_isolated_buckets_by_channel_key():
    limiter = RateLimiter(RateLimitConfig(max_calls=1))
    assert await limiter.is_allowed("telegram:user1") is True
    assert await limiter.is_allowed("discord:user1") is True


@pytest.mark.asyncio
async def test_window_expiration_allows_again():
    limiter = RateLimiter(RateLimitConfig(max_calls=1, window_seconds=1))
    assert await limiter.is_allowed("user1") is True
    assert await limiter.is_allowed("user1") is False
    await asyncio.sleep(1.05)
    assert await limiter.is_allowed("user1") is True
