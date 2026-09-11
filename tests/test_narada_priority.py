from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.modules.narada.priority import (
    DEFAULT_ORDER,
    classify_task_type,
    get_or_create_priority,
    parse_override,
    set_priority,
)


def test_classify_task_type_coding():
    assert classify_task_type("there's a bug in my python script") == "coding"


def test_classify_task_type_summarization():
    assert classify_task_type("please summarize this article") == "summarization"


def test_classify_task_type_defaults_to_general():
    assert classify_task_type("hey there") == "general"


def test_parse_override_known_provider():
    provider, rest = parse_override(f"@{DEFAULT_ORDER[0]} what time is it")
    assert provider == DEFAULT_ORDER[0]
    assert rest == "what time is it"


def test_parse_override_unknown_provider_not_treated_as_override():
    provider, rest = parse_override("@home is a nice place")
    assert provider is None
    assert rest == "@home is a nice place"


def test_parse_override_no_at_sign():
    provider, rest = parse_override("just a normal message")
    assert provider is None
    assert rest == "just a normal message"


@pytest.mark.asyncio
async def test_get_or_create_priority_creates_default_on_first_encounter():
    fake_collection = MagicMock()
    fake_collection.find_one = AsyncMock(return_value=None)
    fake_collection.insert_one = AsyncMock()
    fake_db = {"narada_priority_table": fake_collection}

    with patch("app.modules.narada.priority.get_database", return_value=fake_db):
        order = await get_or_create_priority("new-task-type")

    assert order == DEFAULT_ORDER
    fake_collection.insert_one.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_or_create_priority_returns_existing():
    fake_collection = MagicMock()
    fake_collection.find_one = AsyncMock(return_value={"_id": "coding", "order": ["groq", "nvidia-build"]})
    fake_db = {"narada_priority_table": fake_collection}

    with patch("app.modules.narada.priority.get_database", return_value=fake_db):
        order = await get_or_create_priority("coding")

    assert order == ["groq", "nvidia-build"]


@pytest.mark.asyncio
async def test_set_priority_rejects_unknown_provider():
    with pytest.raises(ValueError):
        await set_priority("coding", ["not-a-real-provider"])


@pytest.mark.asyncio
async def test_set_priority_persists_valid_order():
    fake_collection = MagicMock()
    fake_collection.replace_one = AsyncMock()
    fake_db = {"narada_priority_table": fake_collection}

    with patch("app.modules.narada.priority.get_database", return_value=fake_db):
        order = await set_priority("coding", [DEFAULT_ORDER[1], DEFAULT_ORDER[0]])

    assert order == [DEFAULT_ORDER[1], DEFAULT_ORDER[0]]
    fake_collection.replace_one.assert_awaited_once()
