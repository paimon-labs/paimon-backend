"""
Verifies TTS falls through from edge-tts to Kokoro-ONNX on failure,
same shape as the STT fallback test. Patches the provider's `.call`
directly since ProviderChain captures a direct reference at
construction time.
"""

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from app.modules.vani import tts


@pytest.mark.asyncio
async def test_falls_back_to_kokoro_when_edge_fails():
    edge_provider = tts.tts_chain.providers[0]
    kokoro_provider = tts.tts_chain.providers[1]

    original_edge = edge_provider.call
    original_kokoro = kokoro_provider.call

    edge_provider.call = AsyncMock(side_effect=RuntimeError("network unreachable"))
    kokoro_provider.call = AsyncMock(return_value=Path("/tmp/fake-kokoro-output.wav"))

    try:
        result = await tts.speak("hello world")
    finally:
        edge_provider.call = original_edge
        kokoro_provider.call = original_kokoro

    assert result == Path("/tmp/fake-kokoro-output.wav")


@pytest.mark.asyncio
async def test_empty_text_raises_without_calling_providers():
    with pytest.raises(ValueError):
        await tts.speak("   ")
