"""
Sakshi (साक्षी) — event recording.

Writes structured events into the same MongoDB Atlas cluster Manas
uses (see README for why: the observability canvas needs cross-device
queryable access, which rules out VPS-local-disk-only logging, and a
dedicated logging stack is unjustified complexity at this scale).

Stood up early, per the plan, so every later phase can record into it
from day one rather than events being retrofitted later. The canvas
itself (reading and rendering this data) is Phase 3.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from app.core.db import get_database

logger = logging.getLogger("paimon.sakshi")

COLLECTION = "sakshi_events"


async def record_event(
    event_type: str,
    module: str,
    data: dict[str, Any] | None = None,
    conversation_id: str | None = None,
) -> None:
    """
    Record one observability event: a tool call, a provider fallback,
    a completion, etc. Never raises — a logging failure must never
    break the request that triggered it; it's logged locally and
    swallowed instead.
    """
    document = {
        "event_type": event_type,  # e.g. "tool_call", "llm_completion", "stt_fallback"
        "module": module,  # e.g. "vani.stt", "narada.llm"
        "data": data or {},
        "conversation_id": conversation_id,
        "timestamp": datetime.now(UTC),
    }

    try:
        db = get_database()
        await db[COLLECTION].insert_one(document)
    except Exception as exc:  # noqa: BLE001 — deliberate: logging must never break the caller
        logger.warning("Sakshi failed to record event (%s/%s): %s", module, event_type, exc)
