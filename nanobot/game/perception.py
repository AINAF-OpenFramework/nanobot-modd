"""Perception sources for game state observation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field


class PerceptionData(BaseModel):
    """
    Normalized perception data from a perception source.

    Attributes:
        raw: Raw data from the source
        normalized: Normalized game state representation
        source_type: Type of perception source
        timestamp_ms: Timestamp in milliseconds
        confidence: Confidence in the perception (0.0 to 1.0)
        metadata: Additional metadata from the source
    """

    raw: Any = None
    normalized: dict[str, Any] = Field(default_factory=dict)
    source_type: str = ""
    timestamp_ms: int = 0
    confidence: float = 1.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class PerceptionSource(ABC):
    """
    Abstract base class for perception sources.

    Perception sources capture game state from various inputs
    (API, event streams, screens) and normalize them to a
    common format.
    """

    @abstractmethod
    async def capture(self) -> Any:
        """
        Capture raw data from the perception source.

        Returns:
            Raw data in source-specific format
        """
        ...

    @abstractmethod
    def normalize(self, raw_data: Any) -> PerceptionData:
        """
        Normalize raw data to a common format.

        Args:
            raw_data: Raw data from capture()

        Returns:
            Normalized PerceptionData
        """
        ...

    async def perceive(self) -> PerceptionData:
        """
        Capture and normalize perception data.

        Returns:
            Normalized PerceptionData
        """
        raw = await self.capture()
        return self.normalize(raw)


class APIPerceptionSource(PerceptionSource):
    """
    Perception source that reads game state from an API.

    Suitable for turn-based games with API endpoints.
    """

    def __init__(
        self,
        api_url: str,
        headers: dict[str, str] | None = None,
        timeout_seconds: int = 10,
    ):
        """
        Initialize the API perception source.

        Args:
            api_url: Base URL for the game API
            headers: Optional HTTP headers
            timeout_seconds: Request timeout
        """
        self._api_url = api_url
        self._headers = headers or {}
        self._timeout = timeout_seconds

    async def capture(self) -> Any:
        """
        Fetch game state from the API.

        Returns:
            JSON response from the API
        """
        import time

        import httpx

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.get(
                    self._api_url,
                    headers=self._headers,
                )
                response.raise_for_status()
                data = response.json()
                logger.debug(f"game.perception.api.capture url={self._api_url}")
                return {"data": data, "timestamp_ms": int(time.time() * 1000)}
        except Exception as e:
            logger.error(f"game.perception.api.capture error={e}")
            return {"error": str(e), "timestamp_ms": int(time.time() * 1000)}

    def normalize(self, raw_data: Any) -> PerceptionData:
        """Normalize API response to PerceptionData."""
        import time

        if isinstance(raw_data, dict) and "error" in raw_data:
            return PerceptionData(
                raw=raw_data,
                normalized={},
                source_type="api",
                timestamp_ms=raw_data.get("timestamp_ms", int(time.time() * 1000)),
                confidence=0.0,
                metadata={"error": raw_data["error"]},
            )

        data = raw_data.get("data", {}) if isinstance(raw_data, dict) else {}
        return PerceptionData(
            raw=raw_data,
            normalized=data,
            source_type="api",
            timestamp_ms=raw_data.get("timestamp_ms", int(time.time() * 1000)),
            confidence=1.0,
            metadata={"api_url": self._api_url},
        )


class EventStreamPerceptionSource(PerceptionSource):
    """
    Perception source that reads game state from an event stream.

    Suitable for real-time games with WebSocket or SSE connections.
    """

    def __init__(
        self,
        stream_url: str,
        event_buffer_size: int = 100,
    ):
        """
        Initialize the event stream perception source.

        Args:
            stream_url: URL for the event stream
            event_buffer_size: Maximum number of events to buffer
        """
        self._stream_url = stream_url
        self._buffer_size = event_buffer_size
        self._event_buffer: list[dict[str, Any]] = []
        self._latest_state: dict[str, Any] = {}

    async def capture(self) -> Any:
        """
        Get the latest state from buffered events.

        Returns:
            Latest game state from the event stream
        """
        import time

        # In a real implementation, this would read from an active connection
        # For now, return the latest buffered state
        return {
            "state": self._latest_state.copy(),
            "events": self._event_buffer[-10:],  # Last 10 events
            "timestamp_ms": int(time.time() * 1000),
        }

    def normalize(self, raw_data: Any) -> PerceptionData:
        """Normalize event stream data to PerceptionData."""
        import time

        state = raw_data.get("state", {}) if isinstance(raw_data, dict) else {}
        return PerceptionData(
            raw=raw_data,
            normalized=state,
            source_type="event_stream",
            timestamp_ms=raw_data.get("timestamp_ms", int(time.time() * 1000)),
            confidence=0.9,  # Slightly lower confidence for streaming
            metadata={
                "stream_url": self._stream_url,
                "event_count": len(raw_data.get("events", [])),
            },
        )

    def push_event(self, event: dict[str, Any]) -> None:
        """
        Push a new event to the buffer.

        Args:
            event: Event data to buffer
        """
        self._event_buffer.append(event)
        if len(self._event_buffer) > self._buffer_size:
            self._event_buffer.pop(0)

        # Update latest state if event contains state update
        if "state" in event:
            self._latest_state.update(event["state"])


class ScreenPerceptionSource(PerceptionSource):
    """
    Perception source that captures game state from screen.

    Suitable for visual games or when API access is not available.
    Requires external visual processing capabilities.
    """

    def __init__(
        self,
        capture_region: tuple[int, int, int, int] | None = None,
        processor: Any = None,
    ):
        """
        Initialize the screen perception source.

        Args:
            capture_region: (x, y, width, height) region to capture, or None for full screen
            processor: Visual processor for extracting game state
        """
        self._capture_region = capture_region
        self._processor = processor

    async def capture(self) -> Any:
        """
        Capture screen data.

        Returns:
            Screen capture data (placeholder implementation)
        """
        import time

        # Placeholder - real implementation would use screen capture
        logger.debug("game.perception.screen.capture region={self._capture_region}")
        return {
            "region": self._capture_region,
            "timestamp_ms": int(time.time() * 1000),
            "data": None,  # Would contain image data
        }

    def normalize(self, raw_data: Any) -> PerceptionData:
        """
        Normalize screen capture to PerceptionData.

        Uses the visual processor to extract game state.
        """
        import time

        normalized = {}
        confidence = 0.0

        if self._processor and raw_data.get("data"):
            try:
                normalized = self._processor.process(raw_data["data"])
                confidence = 0.8  # Visual processing has lower confidence
            except Exception as e:
                logger.error(f"game.perception.screen.normalize error={e}")

        return PerceptionData(
            raw=raw_data,
            normalized=normalized,
            source_type="screen",
            timestamp_ms=raw_data.get("timestamp_ms", int(time.time() * 1000)),
            confidence=confidence,
            metadata={"region": self._capture_region},
        )


class UnifiedPerceptionAdapter:
    """
    Unified adapter that combines multiple perception sources.

    Provides a single interface for game state perception with
    automatic fallback to backup sources.
    """

    def __init__(
        self,
        primary: PerceptionSource,
        fallbacks: list[PerceptionSource] | None = None,
        confidence_threshold: float = 0.5,
    ):
        """
        Initialize the unified perception adapter.

        Args:
            primary: Primary perception source
            fallbacks: List of fallback sources (tried in order)
            confidence_threshold: Minimum confidence to accept perception
        """
        self._primary = primary
        self._fallbacks = fallbacks or []
        self._confidence_threshold = confidence_threshold

    async def perceive(self) -> PerceptionData:
        """
        Perceive game state using primary source with fallbacks.

        Returns:
            PerceptionData from the first successful source
        """
        # Try primary source
        try:
            data = await self._primary.perceive()
            if data.confidence >= self._confidence_threshold:
                logger.debug(
                    f"game.perception.unified.perceive source=primary "
                    f"confidence={data.confidence:.3f}"
                )
                return data
        except Exception as e:
            logger.warning(f"game.perception.unified.perceive primary_error={e}")

        # Try fallbacks
        for i, source in enumerate(self._fallbacks):
            try:
                data = await source.perceive()
                if data.confidence >= self._confidence_threshold:
                    logger.debug(
                        f"game.perception.unified.perceive source=fallback_{i} "
                        f"confidence={data.confidence:.3f}"
                    )
                    return data
            except Exception as e:
                logger.warning(f"game.perception.unified.perceive fallback_{i}_error={e}")

        # Return empty perception if all sources fail
        import time

        logger.error("game.perception.unified.perceive all_sources_failed")
        return PerceptionData(
            source_type="unified",
            timestamp_ms=int(time.time() * 1000),
            confidence=0.0,
            metadata={"error": "All perception sources failed"},
        )

    def add_fallback(self, source: PerceptionSource) -> None:
        """
        Add a fallback perception source.

        Args:
            source: Perception source to add as fallback
        """
        self._fallbacks.append(source)

    @property
    def primary(self) -> PerceptionSource:
        """Get the primary perception source."""
        return self._primary

    @property
    def fallbacks(self) -> list[PerceptionSource]:
        """Get the list of fallback sources."""
        return list(self._fallbacks)
