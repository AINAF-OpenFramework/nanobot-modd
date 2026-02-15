# Production Deployment

nanobot now includes optional production-readiness building blocks:

- CI workflow (`.github/workflows/ci.yml`) for lint, tests, coverage, and Docker checks
- CodeQL workflow (`.github/workflows/codeql.yml`) for Python + JavaScript scanning
- Per-channel rate limiting defaults (10 calls / 60 seconds)
- Keyring-backed API key migration (`nanobot migrate-keys`)
- Optional Prometheus exporter (`telemetry.enabled`, default port `9090`)
- Async memory consolidation queue in the agent loop
- Latent reasoning circuit breaker fail-safe
- Optional local embeddings extra (`pip install -e .[embeddings]`)
