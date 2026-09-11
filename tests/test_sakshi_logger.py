"""
Verifies Sakshi writes events with the expected shape, and — just as
importantly — never raises even when the write fails (a logging
failure must never break the caller's request).
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.modules.sakshi.logger import record_event


@pytest.mark.asyncio
async def test_record_event_writes_expected_document():
    fake_collection = MagicMock()
    fake_collection.insert_one = AsyncMock()
    fake_db = {"sakshi_events": fake_collection}

    with patch("app.modules.sakshi.logger.get_database", return_value=fake_db):
        await record_event("tool_call", module="tattva.tool", data={"tool": "web_search"})

    fake_collection.insert_one.assert_awaited_once()
    document = fake_collection.insert_one.await_args.args[0]
    assert document["event_type"] == "tool_call"
    assert document["module"] == "tattva.tool"
    assert document["data"] == {"tool": "web_search"}
    assert "timestamp" in document


@pytest.mark.asyncio
async def test_record_event_never_raises_on_db_failure():
    with patch(
        "app.modules.sakshi.logger.get_database", side_effect=RuntimeError("MONGODB_URI not configured")
    ):
        # Should not raise, even though the DB is unreachable.
        await record_event("tool_call", module="tattva.tool")
