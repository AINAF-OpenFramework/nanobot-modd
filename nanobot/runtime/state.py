"""Global runtime state for Nanobot."""


class RuntimeState:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RuntimeState, cls).__new__(cls)
            cls._instance.latent_reasoning_enabled = False
        return cls._instance


state = RuntimeState()

