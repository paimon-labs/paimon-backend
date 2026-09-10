"""
Centralized configuration. Reads from environment variables / .env.
Each deployment supplies its own values — nothing sensitive lives here
or in source control (see .gitignore).
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # App
    app_name: str = "paimon-backend"
    environment: str = "development"

    # Vani — STT (whisper-large-v3 Groq cloud API primary, faster-whisper local fallback)
    groq_api_key: str | None = None
    stt_local_model: str = "small"  # faster-whisper model size
    stt_local_device: str = "cpu"
    stt_local_compute_type: str = "int8"

    # Vani — TTS (edge-tts free cloud primary, Kokoro-ONNX local fallback)
    tts_edge_voice: str = "en-US-AriaNeural"
    tts_kokoro_model_path: str = "models/kokoro-v1.0.onnx"
    tts_kokoro_voices_path: str = "models/voices-v1.0.bin"
    tts_kokoro_voice: str = "af_sarah"
    tts_kokoro_lang: str = "en-us"
    tts_kokoro_speed: float = 1.0

    # Provider fallback behaviour (shared by STT + TTS chains)
    provider_max_retries: int = 2
    provider_retry_backoff_seconds: float = 1.5


@lru_cache
def get_settings() -> Settings:
    return Settings()
