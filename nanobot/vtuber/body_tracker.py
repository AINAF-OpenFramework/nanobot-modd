"""Body and hand tracking adapters for VTuber gestures."""

from __future__ import annotations


class BodyTracker:
    """Maps body tracking or gameplay cues to avatar gestures."""

    GESTURE_MAP = {
        "aggressive": "point",
        "defensive": "guard",
        "check": "point",
        "checkmate": "celebrate",
        "opening": "wave",
        "neutral": "idle",
    }

    def __init__(self, backend: str = "mediapipe", enabled: bool = False):
        self.backend = backend
        self.enabled = enabled

    def normalize(self, raw: dict[str, float]) -> dict[str, float]:
        """Normalize hand tracking payload."""
        return {
            "left_hand": max(0.0, min(1.0, raw.get("left_hand", 0.0))),
            "right_hand": max(0.0, min(1.0, raw.get("right_hand", 0.0))),
            "shoulder_yaw": max(-1.0, min(1.0, raw.get("shoulder_yaw", 0.0))),
        }

    def map_context_to_gesture(self, cues: dict[str, str]) -> str:
        """Map gameplay/persona context to a reusable gesture token."""
        phase = str(cues.get("phase", "neutral")).lower()
        return self.GESTURE_MAP.get(phase, "idle")
