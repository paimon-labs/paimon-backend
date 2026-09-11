"""
Shared provider-fallback/retry logic.

The pattern is the same everywhere it's used: try a primary provider,
retry it a bounded number of times, and fall through to the next
provider in the chain if it keeps failing. Originally built for Vani
(STT: Groq -> local faster-whisper; TTS: edge-tts -> local Kokoro),
and reused as-is by Narada (LLM: NVIDIA build -> Groq -> local Ollama)
rather than re-deriving the same retry/fallback logic per module.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

logger = logging.getLogger("paimon.providers")



@dataclass
class Provider[T]:
    """A single named provider in a fallback chain."""

    name: str
    call: Callable[..., Awaitable[T]]
    retries: int = 2
    backoff_seconds: float = 1.5


class AllProvidersFailedError(Exception):
    """Raised when every provider in a chain has been exhausted."""

    def __init__(self, chain_name: str, errors: dict[str, Exception]):
        self.chain_name = chain_name
        self.errors = errors
        summary = ", ".join(f"{name}: {err}" for name, err in errors.items())
        super().__init__(f"All providers failed for '{chain_name}': {summary}")


class ProviderChain[T]:
    """
    Ordered list of providers. Calls the first, retrying transient
    failures per-provider, and falls through to the next provider on
    exhaustion rather than failing the whole request outright.

    Used identically by STT (cloud Groq -> local faster-whisper), TTS
    (cloud edge-tts -> local Kokoro), and Narada's LLM chain (NVIDIA
    build -> Groq -> local Ollama).
    """

    def __init__(self, name: str, providers: list[Provider[T]]):
        if not providers:
            raise ValueError(f"ProviderChain '{name}' needs at least one provider")
        self.name = name
        self.providers = providers

    async def run(self, *args, order: list[str] | None = None, **kwargs) -> T:
        """
        Run the chain. By default uses self.providers in their fixed
        order. Pass `order` (a list of provider names) to try them in
        a different sequence for this call only — e.g. Narada's
        priority table reordering by task type, or a manual
        `@provider` override picking one specific provider. Unknown
        names in `order` are ignored; providers not mentioned in
        `order` are NOT dropped, just appended after it, so a partial
        override still falls through to the rest of the chain.
        """
        providers = self.providers
        if order:
            by_name = {p.name: p for p in self.providers}
            ordered = [by_name[name] for name in order if name in by_name]
            remaining = [p for p in self.providers if p.name not in set(order)]
            providers = ordered + remaining

        errors: dict[str, Exception] = {}

        for provider in providers:
            wrapped = retry(
                stop=stop_after_attempt(provider.retries),
                wait=wait_exponential(multiplier=provider.backoff_seconds, min=1, max=10),
                retry=retry_if_exception_type(Exception),
                reraise=True,
            )(provider.call)

            try:
                logger.info("%s: trying provider=%s", self.name, provider.name)
                result = await wrapped(*args, **kwargs)
                logger.info("%s: provider=%s succeeded", self.name, provider.name)
                return result
            except Exception as exc:  # noqa: BLE001 — deliberate: this is the fallback boundary
                logger.warning(
                    "%s: provider=%s exhausted retries, falling through: %s",
                    self.name,
                    provider.name,
                    exc,
                )
                errors[provider.name] = exc

        raise AllProvidersFailedError(self.name, errors)
