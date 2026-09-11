"""
Tattva — Tool interface and registry.

A Tool is code: a name, a description Narada/the LLM uses to decide
when to call it, a Pydantic model describing its parameters (doubles
as JSON-schema for the LLM tool-use API), and an async callable that
does the work. Registration happens via the `@tool` decorator at
import time, mirroring FastAPI's route-registration pattern.

Tools are deliberately NOT persisted to Mongo — they're code, matching
Phase 12's meta-tool goal (the LLM drafts an *implementation*, not a
data record). This is what distinguishes a Tool from a Skill: Skills
(next in Phase 3) are markdown+frontmatter data, mutable at runtime
without a deploy.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

ToolFunc = Callable[..., Awaitable[Any]]


class ToolError(Exception):
    """Raised when a tool is invoked with bad params or fails to run."""


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    params_model: type[BaseModel]
    func: ToolFunc

    @property
    def schema(self) -> dict[str, Any]:
        """Anthropic/OpenAI-style tool-use schema for this tool."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.params_model.model_json_schema(),
        }

    async def run(self, **kwargs: Any) -> Any:
        try:
            validated = self.params_model(**kwargs)
        except Exception as exc:
            raise ToolError(f"Invalid params for tool '{self.name}': {exc}") from exc

        try:
            return await self.func(**validated.model_dump())
        except ToolError:
            raise
        except Exception as exc:
            raise ToolError(f"Tool '{self.name}' failed: {exc}") from exc


_registry: dict[str, Tool] = {}


def tool(name: str, description: str, params_model: type[BaseModel]) -> Callable[[ToolFunc], ToolFunc]:
    """Decorator that registers an async function as a Tool."""

    def decorator(func: ToolFunc) -> ToolFunc:
        if name in _registry:
            raise ValueError(f"Tool '{name}' is already registered")
        _registry[name] = Tool(name=name, description=description, params_model=params_model, func=func)
        return func

    return decorator


def get_tool(name: str) -> Tool | None:
    return _registry.get(name)


def list_tools() -> list[Tool]:
    return list(_registry.values())


def all_schemas() -> list[dict[str, Any]]:
    """Schemas for every registered tool — what gets handed to the LLM."""
    return [t.schema for t in _registry.values()]


def _clear_registry_for_tests() -> None:
    """Test-only escape hatch — importing builtin tools twice in one
    process would otherwise raise on the duplicate-name check."""
    _registry.clear()
