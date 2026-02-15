"""Secure API key storage using OS keyring."""

from __future__ import annotations

import os

from loguru import logger

try:
    import keyring
except Exception:  # pragma: no cover - optional dependency import failure
    keyring = None


class KeyringManager:
    SERVICE_NAME = "nanobot"

    def __init__(self, use_keyring: bool = True):
        self.use_keyring = bool(use_keyring and self._is_available())

    @staticmethod
    def _is_available() -> bool:
        if keyring is None:
            return False
        try:
            keyring.get_keyring()
            return True
        except Exception:
            return False

    def set_key(self, provider: str, api_key: str) -> None:
        if not self.use_keyring or keyring is None:
            return
        try:
            keyring.set_password(self.SERVICE_NAME, provider, api_key)
        except Exception as exc:
            logger.error(f"Keyring error: {exc}")

    def get_key(self, provider: str) -> str:
        if not self.use_keyring or keyring is None:
            return ""
        try:
            return keyring.get_password(self.SERVICE_NAME, provider) or ""
        except Exception:
            return ""


def load_api_key(provider: str, config_key: str, km: KeyringManager) -> str:
    """Load with priority: env var > keyring > plain text."""
    env_key = os.getenv(f"{provider.upper()}_API_KEY")
    if env_key:
        return env_key

    keyring_key = km.get_key(provider)
    if keyring_key:
        return keyring_key

    return config_key
