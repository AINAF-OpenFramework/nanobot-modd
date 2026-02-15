"""Prometheus HTTP exporter."""

from __future__ import annotations

from prometheus_client import start_http_server


class MetricsExporter:
    def __init__(self, port: int = 9090, enabled: bool = True):
        self.port = port
        self.enabled = enabled

    def start(self) -> None:
        if self.enabled:
            start_http_server(self.port)
