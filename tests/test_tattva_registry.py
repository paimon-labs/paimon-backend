"""
Verifies Tattva's Tool registry (registration, schema shape, param
validation) and the executor's Sakshi event recording on success and
on failure.
"""

from unittest.mock import AsyncMock, patch

import pytest
from pydantic import BaseModel

from app.modules import tattva  # noqa: F401 — registers builtin tools
from app.modules.tattva.executor import execute_tool
from app.modules.tattva.registry import ToolError, all_schemas, get_tool, list_tools, tool


def test_builtin_tools_are_registered():
    names = {t.name for t in list_tools()}
    assert {"echo", "current_time"} <= names


def test_schema_shape():
    schema = get_tool("echo").schema
    assert schema["name"] == "echo"
    assert "description" in schema
    assert "properties" in schema["input_schema"]
    assert "text" in schema["input_schema"]["properties"]


def test_all_schemas_returns_one_entry_per_tool():
    assert len(all_schemas()) == len(list_tools())


def test_duplicate_registration_raises():
    class Params(BaseModel):
        pass

    async def handler() -> None:
        return None

    with pytest.raises(ValueError):
        tool(name="echo", description="dup", params_model=Params)(handler)


@pytest.mark.asyncio
async def test_execute_tool_success_records_call_and_result():
    with patch("app.modules.tattva.executor.record_event", new_callable=AsyncMock) as mock_record:
        result = await execute_tool("echo", {"text": "hi"}, conversation_id="conv-1")

    assert result == "hi"
    event_types = [call.args[0] for call in mock_record.await_args_list]
    assert event_types == ["tool_call", "tool_result"]


@pytest.mark.asyncio
async def test_execute_tool_missing_tool_raises_and_records_failure():
    with (
        patch("app.modules.tattva.executor.record_event", new_callable=AsyncMock) as mock_record,
        pytest.raises(ToolError),
    ):
        await execute_tool("does_not_exist", {})

    mock_record.assert_awaited_once()
    assert mock_record.await_args.args[0] == "tool_call_failed"


@pytest.mark.asyncio
async def test_execute_tool_invalid_params_raises_tool_error():
    with (
        patch("app.modules.tattva.executor.record_event", new_callable=AsyncMock),
        pytest.raises(ToolError),
    ):
        await execute_tool("echo", {"wrong_field": "x"})
