"""FastAPI routes exposing Narada's LLM provider chain and the orchestrator, for manual testing."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.modules.narada import llm, priority
from app.modules.narada.orchestrator import handle_turn
from app.modules.sakshi.logger import record_event

router = APIRouter(prefix="/narada", tags=["narada"])


class CompleteRequest(BaseModel):
    prompt: str


class CompleteResponse(BaseModel):
    text: str


@router.post("/complete", response_model=CompleteResponse)
async def complete(payload: CompleteRequest) -> CompleteResponse:
    """LLM completion through the NVIDIA -> Groq -> local Ollama chain."""
    messages = [{"role": "user", "content": payload.prompt}]

    try:
        text = await llm.complete(messages)
    except Exception as exc:
        await record_event("llm_completion_failed", module="narada.llm", data={"error": str(exc)})
        raise HTTPException(status_code=502, detail=f"LLM completion failed: {exc}") from exc

    await record_event(
        "llm_completion",
        module="narada.llm",
        data={"prompt_length": len(payload.prompt), "response_length": len(text)},
    )
    return CompleteResponse(text=text)


class TurnRequest(BaseModel):
    text: str
    conversation_id: str = "default"


class TurnResponse(BaseModel):
    turn_id: str
    response: str
    task_type: str
    skill_used: str | None
    provider_order: list[str]


@router.post("/turn", response_model=TurnResponse)
async def turn(payload: TurnRequest) -> TurnResponse:
    """The core agent loop, text in/out: skill match -> priority-table
    routing -> LLM completion, recorded as one Sakshi turn. This is
    what Vani's /vani/converse calls under the hood for voice."""
    try:
        result = await handle_turn(payload.text, payload.conversation_id)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Turn failed: {exc}") from exc
    return TurnResponse(
        turn_id=result.turn_id,
        response=result.response,
        task_type=result.task_type,
        skill_used=result.skill_used,
        provider_order=result.provider_order,
    )


class PriorityOut(BaseModel):
    task_type: str
    order: list[str]


@router.get("/priority/{task_type}", response_model=PriorityOut)
async def get_priority(task_type: str) -> PriorityOut:
    order = await priority.get_or_create_priority(task_type)
    return PriorityOut(task_type=task_type, order=order)


class SetPriorityRequest(BaseModel):
    order: list[str]


@router.put("/priority/{task_type}", response_model=PriorityOut)
async def set_priority(task_type: str, payload: SetPriorityRequest) -> PriorityOut:
    try:
        order = await priority.set_priority(task_type, payload.order)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return PriorityOut(task_type=task_type, order=order)
