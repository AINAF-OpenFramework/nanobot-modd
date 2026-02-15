from nanobot.vtuber.avatar import AvatarController
from nanobot.vtuber.body_tracker import BodyTracker
from nanobot.vtuber.face_tracker import FaceTracker


def test_avatar_state_mapping_clamps_values():
    avatar = AvatarController()

    avatar.load_vrm("models/tanyalahd.vrm")
    avatar.apply_expression("smile", intensity=2.0)
    avatar.apply_gesture("wave", hand_pose="open")
    avatar.apply_face_tracking({"eye_openness": 2.0, "head_tilt": -3.0})

    snapshot = avatar.snapshot()
    assert snapshot["model_path"].endswith("tanyalahd.vrm")
    assert snapshot["expression"] == "smile"
    assert snapshot["expression_intensity"] == 1.0
    assert snapshot["gesture"] == "wave"
    assert snapshot["eye_openness"] == 1.0
    assert snapshot["head_tilt"] == -1.0


def test_trackers_map_persona_and_context_to_output_tokens():
    face = FaceTracker()
    body = BodyTracker()

    expression, intensity = face.map_persona_to_expression({"mood": "win", "confidence": 0.9})
    gesture = body.map_context_to_gesture({"phase": "checkmate"})

    assert expression == "excited"
    assert intensity == 0.9
    assert gesture == "celebrate"
