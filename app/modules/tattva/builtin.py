"""
Built-in tools, registered at import time. Deliberately trivial —
these exist to prove the registry/executor/canvas path end-to-end
before real tools (Phase 6+) land. Import this module (for its
side effects) wherever the registry needs to be populated.
"""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field

from app.modules.tattva.registry import tool


class EchoParams(BaseModel):
    text: str = Field(..., description="Text to echo back")


@tool(name="echo", description="Echo back the given text. Used for testing the tool pipeline.", params_model=EchoParams)
async def echo(text: str) -> str:
    return text


class CurrentTimeParams(BaseModel):
    pass


@tool(name="current_time", description="Get the current UTC time.", params_model=CurrentTimeParams)
async def current_time() -> str:
    return datetime.now(UTC).isoformat()
