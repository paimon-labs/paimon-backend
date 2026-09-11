"""FastAPI routes exposing Sakshi's recorded events as canvas-ready, turn-grouped data."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from app.modules.sakshi.logger import get_conversation_turns, get_turn_events

router = APIRouter(prefix="/sakshi", tags=["sakshi"])


def _serialize(doc: dict[str, Any]) -> dict[str, Any]:
    out = dict(doc)
    out["_id"] = str(out["_id"])
    out["timestamp"] = out["timestamp"].isoformat()
    return out


@router.get("/canvas/{conversation_id}")
async def canvas(conversation_id: str) -> list[dict[str, Any]]:
    """Turn-grouped entries for a conversation — what a client renders
    directly as the observability canvas: a list of turns, each an
    ordered list of typed (kind: tool_call/reasoning/memory_recall/
    final_answer) entries."""
    turns = await get_conversation_turns(conversation_id)
    return [
        {"turn_id": t["turn_id"], "entries": [_serialize(e) for e in t["entries"]]}
        for t in turns
    ]


@router.get("/turn/{turn_id}")
async def turn(turn_id: str) -> list[dict[str, Any]]:
    events = await get_turn_events(turn_id)
    return [_serialize(e) for e in events]
