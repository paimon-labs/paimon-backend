"""
Vani — Speech to Text.

Implements a hybrid STT engine pattern utilizing async `httpx` and running 
through Vani's shared ProviderChain (see providers/base.py) for an async-first
backend.

Features:
  - Primary: Groq's whisper-large-v3-turbo (cloud)
  - Fallback: faster-whisper "small", CPU, int8 (local)
  - Falls back on: missing key, 401/403, 429, any other non-200 status code,
    or a network/timeout error.

Groq is NOT retried before falling back — a failed Groq request triggers
an immediate fallback to local processing. The shared ProviderChain's retry 
knob is set to 1 for the Groq provider to enforce this behavior.
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
from pathlib import Path

import httpx

from app.core.config import get_settings
from app.modules.vani.providers.base import Provider, ProviderChain

settings = get_settings()
logger = logging.getLogger("paimon.vani.stt")

GROQ_API_URL = "https://api.groq.com/openai/v1/audio/transcriptions"

_local_model = None  # lazily loaded faster-whisper model, loaded once per process


class GroqAuthError(Exception):
    """
    Raised (then swallowed by the fallback chain) when Groq returns
    401/403. Kept as a distinct type — same intent as the original's
    `auth_error_callback` — so a future caller (e.g. the canvas, or a
    notification) can specifically detect "the API key needs
    rotating" rather than a generic failure.
    """


def _load_local_model():
    global _local_model
    if _local_model is None:
        from faster_whisper import WhisperModel

        logger.info(
            "Loading faster-whisper local model (%s, %s, %s)...",
            settings.stt_local_model,
            settings.stt_local_device,
            settings.stt_local_compute_type,
        )
        _local_model = WhisperModel(
            settings.stt_local_model,
            device=settings.stt_local_device,
            compute_type=settings.stt_local_compute_type,
        )
    return _local_model


async def _transcribe_groq(audio_bytes: bytes, filename: str = "audio.ogg", **_kwargs) -> str:
    """Ported from HybridEngine.transcribe()'s Groq branch."""
    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY not configured")

    headers = {"Authorization": f"Bearer {settings.groq_api_key}"}
    files = {"file": (filename, audio_bytes, "audio/ogg")}
    data = {"model": "whisper-large-v3-turbo", "response_format": "text"}

    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.post(GROQ_API_URL, headers=headers, files=files, data=data)

    # Quota tracking, ported from the original's ui_callback — logged for now;
    # Sakshi (Phase 2) will pick this up as a proper telemetry event later.
    limit = response.headers.get("x-ratelimit-limit-tokens")
    remaining = response.headers.get("x-ratelimit-remaining-tokens")
    if limit and remaining:
        try:
            pct = int((float(remaining) / float(limit)) * 100) if float(limit) > 0 else None
            if pct is not None:
                logger.info("groq stt quota remaining: %d%%", pct)
        except ValueError:
            pass

    if response.status_code == 200:
        return response.text.strip()
    if response.status_code in (401, 403):
        logger.warning("Groq API auth error (%d) — key needs rotating", response.status_code)
        raise GroqAuthError(f"Groq auth error {response.status_code}")
    if response.status_code == 429:
        logger.warning("Groq API rate limit reached")
        raise RuntimeError("Groq rate limit (429)")
    raise RuntimeError(f"Groq API error {response.status_code}: {response.text}")


async def _transcribe_local(audio_bytes: bytes, filename: str = "audio.ogg", **_kwargs) -> str:
    """Ported from HybridEngine._transcribe_local()."""
    suffix = Path(filename).suffix or ".ogg"

    def _run() -> str:
        model = _load_local_model()
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = Path(tmp.name)
        try:
            segments, _info = model.transcribe(str(tmp_path), beam_size=5)
            return " ".join(segment.text for segment in segments).strip()
        finally:
            tmp_path.unlink(missing_ok=True)

    # faster-whisper is CPU-bound and blocking — keep it off the event loop.
    return await asyncio.to_thread(_run)


stt_chain = ProviderChain(
    name="stt",
    providers=[
        Provider(name="groq", call=_transcribe_groq, retries=1),  # no retry — mirrors original
        Provider(name="local-faster-whisper", call=_transcribe_local, retries=1),
    ],
)


async def listen(audio_bytes: bytes, filename: str = "audio.ogg") -> str:
    """Public entrypoint: raw audio bytes in (any format ffmpeg/faster-whisper
    can decode — mp3/wav/ogg all worked in the original's file-upload mode),
    transcript text out."""
    if not audio_bytes:
        return ""
    return await stt_chain.run(audio_bytes, filename=filename)
