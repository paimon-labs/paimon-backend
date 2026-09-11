# paimon-backend

Cloud orchestrator for PAIMON — a modular monolith, not microservices. Built with **FastAPI** and [**uv**](https://docs.astral.sh/uv/).

## Modules

| Module | Meaning | Responsibility | Status |
|---|---|---|---|
| **Vani** | वाणी — voice | STT/TTS | ✅ Phase 1 complete |
| **Narada** | messenger-sage | Task routing, device pairing, priority/fallback table | 🚧 Phase 2 — LLM provider chain in progress |
| **Tattva** | तत्त्व — principle | Tool registry, Skill registry, sandbox, meta-tool | 🔲 Phase 3+ |
| **Manas** | मनस् — mind-faculty | Structured/vector/episodic memory, mini-RAGs | 🚧 Phase 2 — MongoDB Atlas connection in progress |
| **Sakshi** | साक्षी — witness | Event recording, observability data source | 🚧 Phase 2 — standing up early per plan |

## Status

🚧 **Phase 2 in progress** — cloud database, LLM hosting, and observability logging.

- ✅ Phase 1 — Vani (STT migrated from Whisperlay, hybrid Groq/faster-whisper; TTS hybrid edge-tts/Kokoro-ONNX)
- 🚧 Phase 2 — MongoDB Atlas (structured + vector + episodic, one cluster), Narada's LLM provider chain (NVIDIA build → Groq → local Ollama), Sakshi event logging, hosting deployment

## Architecture notes

- **One MongoDB Atlas cluster serves all of Manas.** Structured facts, vector embeddings (via Atlas Vector Search, native to the free M0 tier), and the episodic log are three collections in one database — not three separate services. Avoids running a dedicated vector DB for no functional gain at this scale.
- **The provider-fallback/retry pattern is shared, not Vani-specific.** `app/core/providers.py`'s `ProviderChain` is used identically by Vani (STT: Groq → local faster-whisper; TTS: edge-tts → local Kokoro) and Narada (LLM: NVIDIA build → Groq → local Ollama) — one retry/fallback implementation, several call sites.
- **Sakshi's event log lives in the same Atlas cluster as Manas**, not a dedicated logging stack (ELK/Loki/Datadog) — the canvas needs cross-device queryable access, which rules out VPS-local-disk-only logging, and a full logging stack is unjustified complexity at this scale.
- **Narada's LLM chain is fixed (NVIDIA → Groq → local Ollama) for now.** The plan's full dynamic per-task-type priority table, built up as task types are encountered, is a Phase 3 concern layered on top of this, not a replacement for it.

## Deployment model

This repo contains application code only. Each deployment (including the maintainer's own) supplies its own API keys, database, memory, and RAGs — none of that lives in this source tree.

## Setup

```bash
uv sync --dev              # creates .venv and installs from uv.lock / pyproject.toml
cp .env.example .env       # fill in your own keys
uv run uvicorn app.main:app --reload
```

Interactive API docs (Swagger UI) at `http://127.0.0.1:8000/docs` once running.

### Local Ollama fallback (optional, for Narada's LLM chain)

The local LLM fallback isn't auto-installed. To use it:

```bash
# install Ollama: https://ollama.com/download
ollama pull llama3.2
ollama serve   # runs on http://localhost:11434 by default
```

### Kokoro-ONNX model files (optional, for Vani's local TTS fallback)

Also not auto-downloaded — see `.env.example` for the two files and where they go.

## Test

```bash
uv run pytest -q
uv run ruff check .
```

## License

MIT — see [LICENSE](./LICENSE).
