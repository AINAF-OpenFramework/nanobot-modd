import asyncio

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


@pytest.mark.asyncio
async def test_transitions_to_half_open_and_closes_on_success():
    breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=1)

    async def failing_operation():
        raise ValueError()

    async def successful_operation():
        return "ok"

    with pytest.raises(ValueError):
        await breaker.call(failing_operation)
    assert breaker.state == CircuitState.OPEN

    await asyncio.sleep(1.05)
    assert await breaker.call(successful_operation) == "ok"
    assert breaker.state == CircuitState.CLOSED
