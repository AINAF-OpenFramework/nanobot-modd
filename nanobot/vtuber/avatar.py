"""Avatar control for VTuber streaming pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic


@dataclass(slots=True)
class AvatarState:
    """Runtime avatar state used by streaming outputs."""

    model_path: str | None = None
    expression: str = "neutral"
    expression_intensity: float = 0.0
    gesture: str = "idle"
    eye_openness: float = 1.0
    head_tilt: float = 0.0
    hand_pose: str = "idle"
    updated_at: float = field(default_factory=monotonic)


class AvatarController:
    """Controls VRM avatar state and lightweight expression/gesture mapping."""

    def __init__(self, enable_streaming: bool = True):
        self.enable_streaming = enable_streaming
        self.state = AvatarState()

    def load_vrm(self, model_path: str) -> AvatarState:
        """Load VRM model path for downstream renderers."""
        if not model_path:
            raise ValueError("model_path is required")
        self.state.model_path = str(Path(model_path))
        self.state.updated_at = monotonic()
        return self.state

    def apply_expression(self, expression: str, intensity: float = 0.7) -> AvatarState:
        """Apply facial expression to current avatar state."""
        self.state.expression = expression or "neutral"
        self.state.expression_intensity = max(0.0, min(1.0, intensity))
        self.state.updated_at = monotonic()
        return self.state

    def apply_gesture(self, gesture: str, hand_pose: str | None = None) -> AvatarState:
        """Apply gesture and optional hand pose."""
        self.state.gesture = gesture or "idle"
        if hand_pose is not None:
            self.state.hand_pose = hand_pose
        self.state.updated_at = monotonic()
        return self.state

    def apply_face_tracking(self, face_data: dict[str, float]) -> AvatarState:
        """Map normalized face tracking values to avatar controls."""
        self.state.eye_openness = max(0.0, min(1.0, face_data.get("eye_openness", 1.0)))
        self.state.head_tilt = max(-1.0, min(1.0, face_data.get("head_tilt", 0.0)))
        self.state.updated_at = monotonic()
        return self.state

    def snapshot(self) -> dict[str, float | str | None]:
        """Return serializable state for stream bridges (OBS/WebRTC/etc)."""
        return {
            "model_path": self.state.model_path,
            "expression": self.state.expression,
            "expression_intensity": self.state.expression_intensity,
            "gesture": self.state.gesture,
            "eye_openness": self.state.eye_openness,
            "head_tilt": self.state.head_tilt,
            "hand_pose": self.state.hand_pose,
            "updated_at": self.state.updated_at,
            "streaming": self.enable_streaming,
        }
