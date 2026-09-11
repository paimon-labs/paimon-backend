"""
Verifies Narada's LLM chain falls through NVIDIA -> Groq -> local
Ollama on failure, same pattern as the Vani fallback tests. Patches
providers' `.call` directly since ProviderChain captures a direct
reference at construction time.
"""

from unittest.mock import AsyncMock

import pytest

from app.modules.narada import llm


@pytest.mark.asyncio
async def test_falls_through_all_three_providers_in_order():
    nvidia, groq, ollama = llm.llm_chain.providers
    originals = [p.call for p in (nvidia, groq, ollama)]

    nvidia_mock = AsyncMock(side_effect=RuntimeError("nvidia down"))
    groq_mock = AsyncMock(side_effect=RuntimeError("groq rate limited"))
    ollama_mock = AsyncMock(return_value="local response")
    nvidia.call, groq.call, ollama.call = nvidia_mock, groq_mock, ollama_mock

    try:
        result = await llm.complete([{"role": "user", "content": "hi"}])
    finally:
        nvidia.call, groq.call, ollama.call = originals

    assert result == "local response"
    nvidia_mock.assert_awaited_once()
    groq_mock.assert_awaited_once()
    ollama_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_stops_at_first_success():
    nvidia, groq, ollama = llm.llm_chain.providers
    originals = [p.call for p in (nvidia, groq, ollama)]

    nvidia_mock = AsyncMock(return_value="nvidia response")
    groq_mock = AsyncMock()
    ollama_mock = AsyncMock()
    nvidia.call, groq.call, ollama.call = nvidia_mock, groq_mock, ollama_mock

    try:
        result = await llm.complete([{"role": "user", "content": "hi"}])
    finally:
        nvidia.call, groq.call, ollama.call = originals

    assert result == "nvidia response"
    groq_mock.assert_not_awaited()
    ollama_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_empty_messages_raises():
    with pytest.raises(ValueError):
        await llm.complete([])
