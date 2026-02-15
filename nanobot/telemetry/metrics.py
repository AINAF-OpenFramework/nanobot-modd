"""Prometheus metrics."""

from __future__ import annotations

import time
from functools import wraps

from prometheus_client import Counter, Gauge, Histogram

latent_reasoning_duration = Histogram(
    "nanobot_latent_reasoning_duration_seconds",
    "Latent reasoning duration",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
)

memory_retrieval_duration = Histogram(
    "nanobot_memory_retrieval_duration_seconds",
    "Memory retrieval duration",
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0],
)

tool_execution_count = Counter(
    "nanobot_tool_execution_total",
    "Tool executions",
    ["tool_name", "status"],
)

memory_ops_count = Counter(
    "nanobot_memory_ops_total",
    "Memory operations",
    ["operation", "status"],
)

active_sessions = Gauge("nanobot_active_sessions", "Active sessions")


def track_duration(metric: Histogram):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start = time.time()
            try:
                return await func(*args, **kwargs)
            finally:
                metric.observe(time.time() - start)

        return wrapper

    return decorator
