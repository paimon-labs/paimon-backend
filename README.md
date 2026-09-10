# paimon-backend

Cloud orchestrator for PAIMON — a modular monolith, not microservices. Built with **FastAPI**.

## Modules

| Module | Meaning | Responsibility |
|---|---|---|
| **Vani** | वाणी — voice | STT/TTS |
| **Narada** | messenger-sage | Task routing, device pairing, priority/fallback table |
| **Tattva** | तत्त्व — principle | Tool registry, Skill registry, sandbox, meta-tool |
| **Manas** | मनस् — mind-faculty | Structured/vector/episodic memory, mini-RAGs |
| **Sakshi** | साक्षी — witness | Event recording, observability data source |

## Status

🚧 Phase 1 in progress — migrating Whisperlay's STT engine in as the Vani module, adding TTS.

## Deployment model

This repo contains application code only. Each deployment (including the maintainer's own) supplies its own API keys, database, memory, and RAGs — none of that lives in this source tree.

## Setup

Dependency management is via [**uv**](https://docs.astral.sh/uv/).

```bash
uv sync                  # creates .venv and installs from uv.lock / pyproject.toml
cp .env.example .env     # fill in your own keys
uv run uvicorn app.main:app --reload
```

## License

MIT — see [LICENSE](./LICENSE).
