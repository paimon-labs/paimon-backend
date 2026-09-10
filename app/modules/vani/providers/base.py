"""
Shared provider-fallback/retry logic for Vani (STT + TTS).

Both STT and TTS follow the same shape: try a primary (usually cloud)
provider, retry it a bounded number of times, and fall through to the
next provider in the chain (usually local) if it keeps failing. This
module is the one place that logic lives, so STT and TTS don't
duplicate it (per Phase 1 of the project plan).
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

logger = logging.getLogger("paimon.vani")


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

    Used identically by STT (cloud whisper-large-v3 -> local faster-whisper) and
    TTS (cloud egde-tts-> local kokoro-onnx).
    """

    def __init__(self, name: str, providers: list[Provider[T]]):
        if not providers:
            raise ValueError(f"ProviderChain '{name}' needs at least one provider")
        self.name = name
        self.providers = providers

    async def run(self, *args, **kwargs) -> T:
        errors: dict[str, Exception] = {}

        for provider in self.providers:
            wrapped = retry(
                stop=stop_after_attempt(provider.retries),
                wait=wait_exponential(multiplier=provider.backoff_seconds, min=1, max=10),
                retry=retry_if_exception_type(Exception),
                reraise=True,
            )(provider.call)

            try:
                logger.info("vani.%s: trying provider=%s", self.name, provider.name)
                result = await wrapped(*args, **kwargs)
                logger.info("vani.%s: provider=%s succeeded", self.name, provider.name)
                return result
            except Exception as exc:  # noqa: BLE001 — deliberate: this is the fallback boundary
                logger.warning(
                    "vani.%s: provider=%s exhausted retries, falling through: %s",
                    self.name,
                    provider.name,
                    exc,
                )
                errors[provider.name] = exc

        raise AllProvidersFailedError(self.name, errors)
