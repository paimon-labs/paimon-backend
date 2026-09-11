from unittest.mock import AsyncMock, patch

import pytest

from app.modules.narada.orchestrator import handle_turn
from app.modules.tattva.skills import Skill


@pytest.mark.asyncio
async def test_handle_turn_no_skill_uses_classified_task_type():
    with (
        patch("app.modules.narada.orchestrator.record_event", new_callable=AsyncMock),
        patch("app.modules.narada.orchestrator.match_skill", new=AsyncMock(return_value=None)),
        patch(
            "app.modules.narada.orchestrator.priority.get_or_create_priority",
            new=AsyncMock(return_value=["nvidia-build", "groq", "local-ollama"]),
        ),
        patch("app.modules.narada.orchestrator.complete", new=AsyncMock(return_value="42")),
    ):
        result = await handle_turn("what is 6 times 7", "conv-1")

    assert result.response == "42"
    assert result.skill_used is None
    assert result.task_type == "factual_qa"
    assert result.provider_order == ["nvidia-build", "groq", "local-ollama"]
    assert result.turn_id


@pytest.mark.asyncio
async def test_handle_turn_skill_match_uses_skill_name_as_task_type():
    skill = Skill(name="summarize-pr", trigger="summarize a pull request", tools=["github_get_pr"], body="do the thing")

    with (
        patch("app.modules.narada.orchestrator.record_event", new_callable=AsyncMock) as mock_record,
        patch("app.modules.narada.orchestrator.match_skill", new=AsyncMock(return_value=skill)),
        patch(
            "app.modules.narada.orchestrator.priority.get_or_create_priority",
            new=AsyncMock(return_value=["groq"]),
        ),
        patch("app.modules.narada.orchestrator.complete", new=AsyncMock(return_value="done")) as mock_complete,
    ):
        result = await handle_turn("summarize this pull request", "conv-1")

    assert result.skill_used == "summarize-pr"
    assert result.task_type == "summarize-pr"
    # skill body should have been folded into the system message
    sent_messages = mock_complete.await_args.args[0]
    assert "do the thing" in sent_messages[0]["content"]
    event_types = [call.args[0] for call in mock_record.await_args_list]
    assert "skill_match" in event_types


@pytest.mark.asyncio
async def test_handle_turn_manual_override_skips_priority_table():
    with (
        patch("app.modules.narada.orchestrator.record_event", new_callable=AsyncMock),
        patch("app.modules.narada.orchestrator.match_skill", new=AsyncMock(return_value=None)),
        patch("app.modules.narada.orchestrator.priority.get_or_create_priority", new=AsyncMock()) as mock_get_priority,
        patch("app.modules.narada.orchestrator.complete", new=AsyncMock(return_value="ok")),
    ):
        result = await handle_turn("@groq what time is it", "conv-1")

    assert result.provider_order == ["groq"]
    mock_get_priority.assert_not_awaited()


@pytest.mark.asyncio
async def test_handle_turn_llm_failure_is_recorded_and_reraised():
    with (
        patch("app.modules.narada.orchestrator.record_event", new_callable=AsyncMock) as mock_record,
        patch("app.modules.narada.orchestrator.match_skill", new=AsyncMock(return_value=None)),
        patch("app.modules.narada.orchestrator.priority.get_or_create_priority", new=AsyncMock(return_value=["groq"])),
        patch("app.modules.narada.orchestrator.complete", new=AsyncMock(side_effect=RuntimeError("all failed"))),
        pytest.raises(RuntimeError),
    ):
        await handle_turn("hello", "conv-1")

    event_types = [call.args[0] for call in mock_record.await_args_list]
    assert "llm_completion_failed" in event_types
