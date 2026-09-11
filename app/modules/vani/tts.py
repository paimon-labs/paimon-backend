"""
Vani — Text to Speech.

Hybrid pattern, mirroring STT: edge-tts (Microsoft Edge's free cloud
voices, no API key required) as primary, Kokoro-ONNX (local, ~82M
params, strong quality-to-size ratio) as fallback. Picked over paid
cloud APIs (OpenAI/ElevenLabs) and pyttsx3 since both halves of this
pair are free and Kokoro sounds meaningfully better than pyttsx3's
robotic output.

Unlike faster-whisper's STT fallback, Kokoro's model files are NOT
auto-downloaded — `kokoro-v1.0.onnx` and `voices-v1.0.bin` must be
placed on disk manually (see README / .env.example for the expected
paths) before the local fallback will work.
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
from pathlib import Path

from app.core.config import get_settings
from app.core.providers import Provider, ProviderChain

settings = get_settings()
logger = logging.getLogger("paimon.vani.tts")

_kokoro_model = None  # lazily loaded Kokoro instance, loaded once per process


def _load_kokoro():
    global _kokoro_model
    if _kokoro_model is None:
        from kokoro_onnx import Kokoro

        model_path = Path(settings.tts_kokoro_model_path)
        voices_path = Path(settings.tts_kokoro_voices_path)
        if not model_path.exists() or not voices_path.exists():
            raise RuntimeError(
                f"Kokoro model files not found at {model_path} / {voices_path} — "
                "download kokoro-v1.0.onnx and voices-v1.0.bin (see README) and "
                "place them at the paths configured in .env"
            )
        logger.info("Loading Kokoro-ONNX local model...")
        _kokoro_model = Kokoro(str(model_path), str(voices_path))
    return _kokoro_model


async def _speak_edge(text: str, **_kwargs) -> Path:
    """Primary: Microsoft Edge TTS — free, no API key, high quality."""
    import edge_tts

    out_path = Path(tempfile.mktemp(suffix=".mp3"))
    communicate = edge_tts.Communicate(text, voice=settings.tts_edge_voice)
    await communicate.save(str(out_path))

    if not out_path.exists() or out_path.stat().st_size == 0:
        out_path.unlink(missing_ok=True)
        raise RuntimeError("edge-tts returned empty audio")

    return out_path


async def _speak_kokoro(text: str, **_kwargs) -> Path:
    """Fallback: Kokoro-ONNX — local, small, no internet required."""

    def _run() -> Path:
        import soundfile as sf

        kokoro = _load_kokoro()
        samples, sample_rate = kokoro.create(
            text,
            voice=settings.tts_kokoro_voice,
            speed=settings.tts_kokoro_speed,
            lang=settings.tts_kokoro_lang,
        )
        out_path = Path(tempfile.mktemp(suffix=".wav"))
        sf.write(str(out_path), samples, sample_rate)
        return out_path

    # Kokoro/onnxruntime is CPU-bound and blocking — keep it off the event loop.
    return await asyncio.to_thread(_run)


tts_chain = ProviderChain(
    name="tts",
    providers=[
        # No retry on either provider — same "fail once, fall through immediately"
        # shape as the migrated STT chain, rather than retrying a likely-persistent
        # failure (missing model files, bad network) before falling back.
        Provider(name="edge-tts", call=_speak_edge, retries=1),
        Provider(name="kokoro-onnx", call=_speak_kokoro, retries=1),
    ],
)


async def speak(text: str) -> Path:
    """Public entrypoint: text in, path to generated audio file out."""
    if not text.strip():
        raise ValueError("Cannot synthesize empty text")
    return await tts_chain.run(text)
