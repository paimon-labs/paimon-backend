"""
Narada — LLM provider chain.

All three providers speak the OpenAI-compatible chat-completions wire
format, so one request function serves all of them (DRY where it
actually earns its place — same shape, same parsing, just a different
base URL and key each time).

  - Primary: NVIDIA build (integrate.api.nvidia.com) — broadest model
    catalog of the three, includes genuine Llama weights.
  - Fallback: Groq (api.groq.com) — fast, cheap. Groq deprecated its
    literal "llama-*" model IDs in 2026; the current recommended
    open-weight replacement is the gpt-oss family, used here by
    default.
  - Local fallback: Ollama, running on this box/VPS. The model is NOT
    auto-pulled — run `ollama pull <model>` yourself first (see
    .env.example). Kept as a last resort, not a true offline path:
    see README for why "local" here doesn't give end users offline
    resilience, since Narada itself only runs when the VPS is
    reachable in the first place.

This is a fixed 3-provider chain for now — the plan's full dynamic
task->AI priority table (built up per task type, with manual override
syntax) is a Phase 3 concern layered on top of this, not replaced by
it.
"""

from __future__ import annotations

import logging

import httpx

from app.core.config import get_settings
from app.core.providers import Provider, ProviderChain

settings = get_settings()
logger = logging.getLogger("paimon.narada.llm")


async def _call_openai_compatible(
    base_url: str,
    model: str,
    messages: list[dict],
    api_key: str | None = None,
    timeout: float = 30.0,
) -> str:
    """Shared request/response handling for any OpenAI-compatible
    chat-completions endpoint — NVIDIA build, Groq, and Ollama all qualify."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {"model": model, "messages": messages}

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            f"{base_url}/chat/completions", headers=headers, json=payload
        )

    if response.status_code == 200:
        data = response.json()
        return data["choices"][0]["message"]["content"]
    if response.status_code in (401, 403):
        raise RuntimeError(f"Auth error {response.status_code} from {base_url}")
    if response.status_code == 429:
        raise RuntimeError(f"Rate limited (429) by {base_url}")
    raise RuntimeError(f"{base_url} returned {response.status_code}: {response.text[:200]}")


async def _call_nvidia(messages: list[dict], **_kwargs) -> str:
    if not settings.nvidia_api_key:
        raise RuntimeError("NVIDIA_API_KEY not configured")
    return await _call_openai_compatible(
        base_url="https://integrate.api.nvidia.com/v1",
        model=settings.nvidia_llm_model,
        messages=messages,
        api_key=settings.nvidia_api_key,
    )


async def _call_groq(messages: list[dict], **_kwargs) -> str:
    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY not configured")
    return await _call_openai_compatible(
        base_url="https://api.groq.com/openai/v1",
        model=settings.groq_llm_model,
        messages=messages,
        api_key=settings.groq_api_key,
    )


async def _call_local_ollama(messages: list[dict], **_kwargs) -> str:
    return await _call_openai_compatible(
        base_url=settings.local_llm_base_url,
        model=settings.local_llm_model,
        messages=messages,
        api_key=None,
    )


llm_chain = ProviderChain(
    name="narada-llm",
    providers=[
        Provider(name="nvidia-build", call=_call_nvidia, retries=1),
        Provider(name="groq", call=_call_groq, retries=1),
        Provider(name="local-ollama", call=_call_local_ollama, retries=1),
    ],
)


async def complete(messages: list[dict]) -> str:
    """Public entrypoint: OpenAI-style messages in, assistant text out."""
    if not messages:
        raise ValueError("messages cannot be empty")
    return await llm_chain.run(messages)
