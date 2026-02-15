import pytest

from nanobot.middleware.circuit_breaker import CircuitBreaker, CircuitState


@pytest.mark.asyncio
async def test_opens_on_failures():
    breaker = CircuitBreaker(failure_threshold=3)

    async def fail():
        raise ValueError()

    for _ in range(3):
        with pytest.raises(ValueError):
            await breaker.call(fail)

    assert breaker.state == CircuitState.OPEN
