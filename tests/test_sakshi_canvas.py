from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.modules.sakshi.logger import get_conversation_turns, get_turn_events, record_event


@pytest.mark.asyncio
async def test_record_event_sets_kind_from_map():
    fake_collection = MagicMock()
    fake_collection.insert_one = AsyncMock()
    fake_db = {"sakshi_events": fake_collection}

    with patch("app.modules.sakshi.logger.get_database", return_value=fake_db):
        await record_event("tool_call", module="tattva.executor", turn_id="t1")

    document = fake_collection.insert_one.await_args.args[0]
    assert document["kind"] == "tool_call"
    assert document["turn_id"] == "t1"


@pytest.mark.asyncio
async def test_record_event_unknown_type_falls_back_to_other():
    fake_collection = MagicMock()
    fake_collection.insert_one = AsyncMock()
    fake_db = {"sakshi_events": fake_collection}

    with patch("app.modules.sakshi.logger.get_database", return_value=fake_db):
        await record_event("some_brand_new_event_type", module="x")

    document = fake_collection.insert_one.await_args.args[0]
    assert document["kind"] == "other"


def _fake_cursor(docs):
    cursor = MagicMock()
    cursor.sort = MagicMock(return_value=cursor)
    cursor.to_list = AsyncMock(return_value=docs)
    return cursor


@pytest.mark.asyncio
async def test_get_turn_events_returns_sorted_events():
    docs = [{"event_type": "tool_call", "turn_id": "t1", "timestamp": datetime.now(UTC)}]
    fake_collection = MagicMock()
    fake_collection.find = MagicMock(return_value=_fake_cursor(docs))
    fake_db = {"sakshi_events": fake_collection}

    with patch("app.modules.sakshi.logger.get_database", return_value=fake_db):
        result = await get_turn_events("t1")

    assert result == docs


@pytest.mark.asyncio
async def test_get_conversation_turns_groups_by_turn_id_preserving_order():
    now = datetime.now(UTC)
    docs = [
        {"event_type": "user_message", "turn_id": "t1", "timestamp": now},
        {"event_type": "tool_call", "turn_id": "t1", "timestamp": now},
        {"event_type": "user_message", "turn_id": "t2", "timestamp": now},
    ]
    fake_collection = MagicMock()
    fake_collection.find = MagicMock(return_value=_fake_cursor(docs))
    fake_db = {"sakshi_events": fake_collection}

    with patch("app.modules.sakshi.logger.get_database", return_value=fake_db):
        turns = await get_conversation_turns("conv-1")

    assert [t["turn_id"] for t in turns] == ["t1", "t2"]
    assert len(turns[0]["entries"]) == 2
    assert len(turns[1]["entries"]) == 1
