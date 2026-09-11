"""FastAPI route exposing Narada's LLM provider chain, for manual testing."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.modules.narada import llm
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
