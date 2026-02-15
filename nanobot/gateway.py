"""Gateway helpers and optional health route."""

from importlib.metadata import PackageNotFoundError, version
from typing import Any

from nanobot.config.loader import load_config

try:
    from fastapi import APIRouter
except Exception:  # pragma: no cover - FastAPI is optional at runtime
    APIRouter = None  # type: ignore[assignment]


router = APIRouter() if APIRouter else None


def _version() -> str:
    try:
        return f"v{version('nanobot-ai')}"
    except PackageNotFoundError:
        return "v0.1.4"


def build_health_payload() -> dict[str, Any]:
    """Build health payload from current settings."""
    config = load_config()
    return {
        "status": "ok",
        "version": _version(),
        "memory": {
            "enabled": config.memory.enabled,
            "entanglement_weight": config.memory.entanglement_weight,
        },
        "latent_reasoning": {
            "enabled": config.enable_quantum_latent,
            "depth": config.memory.latent_max_depth,
        },
    }


if router:
    @router.get("/health")
    async def health_check() -> dict[str, Any]:
        """Basic health endpoint for observability."""
        return build_health_payload()
