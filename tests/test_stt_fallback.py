"""
Verifies the migrated fallback behaviour from Whisperlay's HybridEngine:
if Groq fails (no key configured, in this test), Vani falls through to
the local faster-whisper provider rather than raising.

Patches the provider's `.call` directly rather than the module-level
function, since ProviderChain captures a direct reference to the
callable at construction time.
"""

from unittest.mock import AsyncMock, patch

import pytest

from app.modules.vani import stt


@pytest.mark.asyncio
async def test_falls_back_to_local_when_groq_has_no_key():
    local_provider = stt.stt_chain.providers[1]
    original_call = local_provider.call
    local_provider.call = AsyncMock(return_value="local transcript")

    try:
        with patch.object(stt.settings, "groq_api_key", None):
            result = await stt.listen(b"fake-audio-bytes", filename="test.ogg")
    finally:
        local_provider.call = original_call

    assert result == "local transcript"


@pytest.mark.asyncio
async def test_empty_audio_returns_empty_string_without_calling_providers():
    result = await stt.listen(b"", filename="test.ogg")
    assert result == ""
