"""
Sakshi (साक्षी) — event recording.

Writes structured events into the same MongoDB Atlas cluster Manas
uses (see README for why: the observability canvas needs cross-device
queryable access, which rules out VPS-local-disk-only logging, and a
dedicated logging stack is unjustified complexity at this scale).

Stood up early, per the plan, so every later phase can record into it
from day one rather than events being retrofitted later.

Phase 3 adds turn grouping: every event can carry a turn_id (one per
conversational turn — one user message through to one final answer).
`kind` classifies each event_type into the four canvas-facing
categories the plan calls for (tool call / reasoning / memory recall
/ final answer), so a client can render turns as expandable groups of
typed, visually distinct entries without re-deriving that mapping
itself.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from app.core.db import get_database

logger = logging.getLogger("paimon.sakshi")

COLLECTION = "sakshi_events"

# event_type -> canvas entry kind. Anything not listed here falls
# back to "other" rather than raising — new event_types shouldn't
# have to touch this file to avoid breaking recording.
_KIND_MAP: dict[str, str] = {
    "tool_call": "tool_call",
    "tool_result": "tool_call",
    "tool_call_failed": "tool_call",
    "skill_match": "reasoning",
    "routing_decision": "reasoning",
    "llm_completion": "reasoning",
    "llm_completion_failed": "reasoning",
    "memory_recall": "memory_recall",
    "user_message": "final_answer",
    "final_answer": "final_answer",
    "stt_transcribed": "final_answer",
    "stt_failed": "final_answer",
    "tts_synthesized": "final_answer",
    "tts_failed": "final_answer",
}


async def record_event(
    event_type: str,
    module: str,
    data: dict[str, Any] | None = None,
    conversation_id: str | None = None,
    turn_id: str | None = None,
) -> None:
    """
    Record one observability event: a tool call, a provider fallback,
    a completion, etc. Never raises — a logging failure must never
    break the request that triggered it; it's logged locally and
    swallowed instead.
    """
    document = {
        "event_type": event_type,  # e.g. "tool_call", "llm_completion", "stt_fallback"
        "kind": _KIND_MAP.get(event_type, "other"),
        "module": module,  # e.g. "vani.stt", "narada.llm"
        "data": data or {},
        "conversation_id": conversation_id,
        "turn_id": turn_id,
        "timestamp": datetime.now(UTC),
    }

    try:
        db = get_database()
        await db[COLLECTION].insert_one(document)
    except Exception as exc:  # noqa: BLE001 — deliberate: logging must never break the caller
        logger.warning("Sakshi failed to record event (%s/%s): %s", module, event_type, exc)


async def get_turn_events(turn_id: str) -> list[dict[str, Any]]:
    db = get_database()
    cursor = db[COLLECTION].find({"turn_id": turn_id}).sort("timestamp", 1)
    return await cursor.to_list(length=None)


async def get_conversation_turns(conversation_id: str) -> list[dict[str, Any]]:
    """
    All events for a conversation, grouped into turns in chronological
    order — the shape a canvas client renders directly: a list of
    turns, each an ordered list of typed entries.
    """
    db = get_database()
    cursor = db[COLLECTION].find({"conversation_id": conversation_id}).sort("timestamp", 1)
    events = await cursor.to_list(length=None)

    turns: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    for event in events:
        tid = event.get("turn_id") or "untitled"
        if tid not in turns:
            turns[tid] = []
            order.append(tid)
        turns[tid].append(event)

    return [{"turn_id": tid, "entries": turns[tid]} for tid in order]

