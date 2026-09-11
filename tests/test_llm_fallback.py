"""
Verifies Narada's LLM chain falls through NVIDIA -> Groq -> Gemini ->
local Ollama on failure, same pattern as the Vani fallback tests.
Patches providers' `.call` directly since ProviderChain captures a
direct reference at construction time.
"""

from unittest.mock import AsyncMock

import pytest

from app.modules.narada import llm


@pytest.mark.asyncio
async def test_falls_through_all_four_providers_in_order():
    nvidia, groq, gemini, ollama = llm.llm_chain.providers
    originals = [p.call for p in (nvidia, groq, gemini, ollama)]

    nvidia_mock = AsyncMock(side_effect=RuntimeError("nvidia down"))
    groq_mock = AsyncMock(side_effect=RuntimeError("groq rate limited"))
    gemini_mock = AsyncMock(side_effect=RuntimeError("gemini rate limited"))
    ollama_mock = AsyncMock(return_value="local response")
    nvidia.call, groq.call, gemini.call, ollama.call = nvidia_mock, groq_mock, gemini_mock, ollama_mock

    try:
        result = await llm.complete([{"role": "user", "content": "hi"}])
    finally:
        nvidia.call, groq.call, gemini.call, ollama.call = originals

    assert result == "local response"
    nvidia_mock.assert_awaited_once()
    groq_mock.assert_awaited_once()
    gemini_mock.assert_awaited_once()
    ollama_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_stops_at_first_success():
    nvidia, groq, gemini, ollama = llm.llm_chain.providers
    originals = [p.call for p in (nvidia, groq, gemini, ollama)]

    nvidia_mock = AsyncMock(return_value="nvidia response")
    groq_mock, gemini_mock, ollama_mock = AsyncMock(), AsyncMock(), AsyncMock()
    nvidia.call, groq.call, gemini.call, ollama.call = nvidia_mock, groq_mock, gemini_mock, ollama_mock

    try:
        result = await llm.complete([{"role": "user", "content": "hi"}])
    finally:
        nvidia.call, groq.call, gemini.call, ollama.call = originals

    assert result == "nvidia response"
    groq_mock.assert_not_awaited()
    gemini_mock.assert_not_awaited()
    ollama_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_gemini_used_when_nvidia_and_groq_fail():
    nvidia, groq, gemini, ollama = llm.llm_chain.providers
    originals = [p.call for p in (nvidia, groq, gemini, ollama)]

    nvidia_mock = AsyncMock(side_effect=RuntimeError("nvidia down"))
    groq_mock = AsyncMock(side_effect=RuntimeError("groq down"))
    gemini_mock = AsyncMock(return_value="gemini response")
    ollama_mock = AsyncMock()
    nvidia.call, groq.call, gemini.call, ollama.call = nvidia_mock, groq_mock, gemini_mock, ollama_mock

    try:
        result = await llm.complete([{"role": "user", "content": "hi"}])
    finally:
        nvidia.call, groq.call, gemini.call, ollama.call = originals

    assert result == "gemini response"
    ollama_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_empty_messages_raises():
    with pytest.raises(ValueError):
        await llm.complete([])
