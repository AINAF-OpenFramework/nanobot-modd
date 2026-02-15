"""Facial tracking adapters for VTuber expression mapping."""

from __future__ import annotations


class FaceTracker:
    """Normalizes tracker outputs (OpenSeeFace/MediaPipe/Kalidokit style)."""

    EXPRESSION_MAP = {
        "joy": "smile",
        "happy": "smile",
        "focus": "thinking",
        "thinking": "thinking",
        "blunder": "surprised",
        "mistake": "worried",
        "win": "excited",
        "neutral": "neutral",
    }

    def __init__(self, backend: str = "mediapipe", enabled: bool = False):
        self.backend = backend
        self.enabled = enabled

    def normalize(self, raw: dict[str, float]) -> dict[str, float]:
        """Normalize tracker values to stable ranges used by avatar controller."""
        return {
            "eye_openness": max(0.0, min(1.0, raw.get("eye_openness", 1.0))),
            "head_tilt": max(-1.0, min(1.0, raw.get("head_tilt", 0.0))),
            "brow_raise": max(0.0, min(1.0, raw.get("brow_raise", 0.0))),
            "mouth_open": max(0.0, min(1.0, raw.get("mouth_open", 0.0))),
        }

    def map_persona_to_expression(self, cues: dict[str, str | float]) -> tuple[str, float]:
        """Map persona + game cues to expression/intensity pair."""
        mood = str(cues.get("mood", "neutral")).lower()
        expression = self.EXPRESSION_MAP.get(mood, "neutral")
        confidence = float(cues.get("confidence", 0.7))
        intensity = max(0.0, min(1.0, confidence))
        return expression, intensity
