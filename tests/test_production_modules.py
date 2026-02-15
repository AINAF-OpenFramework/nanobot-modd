import types

import pytest

from nanobot.agent.embeddings import LocalEmbeddings
from nanobot.config.keyring import KeyringManager, load_api_key
from nanobot.telemetry.exporter import MetricsExporter
from nanobot.telemetry.metrics import track_duration


def test_load_api_key_priority(monkeypatch):
    km = KeyringManager(use_keyring=False)
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    assert load_api_key("openai", "config-key", km) == "env-key"


def test_load_api_key_falls_back_to_keyring(monkeypatch):
    class StubKM:
        def get_key(self, provider):
            return "keyring-key"

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert load_api_key("openai", "config-key", StubKM()) == "keyring-key"


def test_load_api_key_falls_back_to_config(monkeypatch):
    class StubKM:
        def get_key(self, provider):
            return ""

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert load_api_key("openai", "config-key", StubKM()) == "config-key"


def test_metrics_exporter_starts(monkeypatch):
    called = {}

    def fake_start_http_server(port):
        called["port"] = port

    monkeypatch.setattr("nanobot.telemetry.exporter.start_http_server", fake_start_http_server)
    MetricsExporter(port=9999, enabled=True).start()
    assert called["port"] == 9999


@pytest.mark.asyncio
async def test_track_duration_decorator():
    observed = []

    class FakeMetric:
        def observe(self, value):
            observed.append(value)

    @track_duration(FakeMetric())
    async def wrapped():
        return "ok"

    assert await wrapped() == "ok"
    assert observed and observed[0] >= 0


def test_local_embeddings_encode_and_similarity(monkeypatch):
    class FakeArray:
        def __init__(self, values):
            self.values = values

        def tolist(self):
            return list(self.values)

    class FakeModel:
        def __init__(self, model_name):
            self.model_name = model_name

        def encode(self, text):
            assert text == "hello"
            return FakeArray([1.0, 0.0])

    fake_module = types.SimpleNamespace(SentenceTransformer=FakeModel)
    monkeypatch.setitem(__import__("sys").modules, "sentence_transformers", fake_module)

    emb = LocalEmbeddings()
    vector = emb.encode("hello")
    assert vector == [1.0, 0.0]
    assert emb.similarity([0.0, 0.0], [1.0, 0.0]) == 0.0
