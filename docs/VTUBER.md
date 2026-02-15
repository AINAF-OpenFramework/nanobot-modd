# VTuber Module

Nanobot now includes modular VTuber primitives under `nanobot.vtuber`:

- `avatar.py`: VRM model path registration and runtime expression/gesture state
- `face_tracker.py`: personality/face cue normalization and expression mapping
- `body_tracker.py`: body cue normalization and gesture mapping

Use `nanobot.interaction.StreamController` to synchronize avatar events with voice output.
