"""
Executes a registered tool and records the call into Sakshi, so the
observability canvas (Phase 3, later) has something to render. This
is the single choke point Narada should call through rather than
invoking Tool.run() directly — keeps event recording out of routing
logic.
"""

from __future__ import annotations

from typing import Any

from app.modules.sakshi.logger import record_event
from app.modules.tattva.registry import ToolError, get_tool


async def execute_tool(
    name: str,
    params: dict[str, Any],
    conversation_id: str | None = None,
) -> Any:
    tool = get_tool(name)
    if tool is None:
        await record_event(
            "tool_call_failed",
            module="tattva.executor",
            data={"tool": name, "error": "not registered", "params": params},
            conversation_id=conversation_id,
        )
        raise ToolError(f"No tool registered with name '{name}'")

    await record_event(
        "tool_call",
        module="tattva.executor",
        data={"tool": name, "params": params},
        conversation_id=conversation_id,
    )

    try:
        result = await tool.run(**params)
    except ToolError as exc:
        await record_event(
            "tool_call_failed",
            module="tattva.executor",
            data={"tool": name, "params": params, "error": str(exc)},
            conversation_id=conversation_id,
        )
        raise

    await record_event(
        "tool_result",
        module="tattva.executor",
        data={"tool": name, "result": _safe_preview(result)},
        conversation_id=conversation_id,
    )
    return result


def _safe_preview(result: Any, limit: int = 2000) -> Any:
    """Keep large tool outputs from bloating the event log."""
    text = str(result)
    return text if len(text) <= limit else text[:limit] + "…(truncated)"
