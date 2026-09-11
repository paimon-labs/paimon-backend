"""FastAPI routes exposing Tattva's tool registry and skill registry, for manual testing."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.modules.tattva import skills as skills_module
from app.modules.tattva.executor import execute_tool
from app.modules.tattva.registry import ToolError, all_schemas
from app.modules.tattva.skills import Skill, SkillError

router = APIRouter(prefix="/tattva", tags=["tattva"])


@router.get("/tools")
async def list_tools() -> list[dict[str, Any]]:
    return all_schemas()


class InvokeRequest(BaseModel):
    params: dict[str, Any] = {}
    conversation_id: str | None = None


@router.post("/tools/{name}/invoke")
async def invoke_tool(name: str, payload: InvokeRequest) -> dict[str, Any]:
    try:
        result = await execute_tool(name, payload.params, conversation_id=payload.conversation_id)
    except ToolError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"tool": name, "result": result}


def _skill_out(skill: Skill) -> dict[str, Any]:
    return {"name": skill.name, "trigger": skill.trigger, "tools": skill.tools, "body": skill.body, "version": skill.version}


class UpsertSkillRequest(BaseModel):
    markdown: str


@router.get("/skills")
async def list_skills() -> list[dict[str, Any]]:
    return [_skill_out(s) for s in await skills_module.list_skills()]


@router.get("/skills/{name}")
async def get_skill(name: str) -> dict[str, Any]:
    skill = await skills_module.get_skill(name)
    if skill is None:
        raise HTTPException(status_code=404, detail=f"No skill named '{name}'")
    return _skill_out(skill)


@router.post("/skills")
async def upsert_skill(payload: UpsertSkillRequest) -> dict[str, Any]:
    try:
        skill = await skills_module.upsert_skill(payload.markdown)
    except SkillError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _skill_out(skill)


@router.delete("/skills/{name}")
async def delete_skill(name: str) -> dict[str, bool]:
    deleted = await skills_module.delete_skill(name)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"No skill named '{name}'")
    return {"deleted": True}


class MatchSkillRequest(BaseModel):
    query: str


@router.post("/skills/match")
async def match_skill(payload: MatchSkillRequest) -> dict[str, Any] | None:
    skill = await skills_module.match_skill(payload.query)
    return _skill_out(skill) if skill else None
