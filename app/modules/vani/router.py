"""FastAPI routes exposing Vani's listen()/speak() round trip."""

from __future__ import annotations

import base64

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.modules.narada.orchestrator import handle_turn
from app.modules.sakshi.logger import record_event
from app.modules.vani import stt, tts

router = APIRouter(prefix="/vani", tags=["vani"])


class TranscriptResponse(BaseModel):
    text: str


class SpeakRequest(BaseModel):
    text: str


@router.post("/listen", response_model=TranscriptResponse)
async def listen(audio: UploadFile = File(...)) -> TranscriptResponse:
    """STT: audio file in, transcript out."""
    audio_bytes = await audio.read()
    filename = audio.filename or "audio.ogg"

    try:
        text = await stt.listen(audio_bytes, filename=filename)
    except Exception as exc:
        await record_event("stt_failed", module="vani.stt", data={"error": str(exc)})
        raise HTTPException(status_code=502, detail=f"STT failed: {exc}") from exc

    await record_event("stt_transcribed", module="vani.stt", data={"text_length": len(text)})
    return TranscriptResponse(text=text)


@router.post("/speak")
async def speak(payload: SpeakRequest) -> FileResponse:
    """TTS: text in, audio file out."""
    try:
        audio_path = await tts.speak(payload.text)
    except Exception as exc:
        await record_event("tts_failed", module="vani.tts", data={"error": str(exc)})
        raise HTTPException(status_code=502, detail=f"TTS failed: {exc}") from exc

    await record_event(
        "tts_synthesized", module="vani.tts", data={"text_length": len(payload.text)}
    )
    media_type = "audio/mpeg" if audio_path.suffix == ".mp3" else "audio/wav"
    return FileResponse(audio_path, media_type=media_type, filename=audio_path.name)


class ConverseResponse(BaseModel):
    turn_id: str
    transcript: str
    response_text: str
    task_type: str
    skill_used: str | None
    audio_base64: str
    audio_mime: str


@router.post("/converse", response_model=ConverseResponse)
async def converse(audio: UploadFile = File(...), conversation_id: str = "default") -> ConverseResponse:
    """
    The full voice round trip: client listen() -> this endpoint
    (Vani STT -> Narada orchestrator -> Vani TTS) -> speak() back to
    the client. Audio comes back base64-encoded in the JSON body
    rather than as a raw file response, so transcript/response text/
    turn_id can travel alongside it in one round trip instead of two.
    """
    audio_bytes = await audio.read()
    filename = audio.filename or "audio.ogg"

    try:
        transcript = await stt.listen(audio_bytes, filename=filename)
    except Exception as exc:
        await record_event("stt_failed", module="vani.stt", data={"error": str(exc)})
        raise HTTPException(status_code=502, detail=f"STT failed: {exc}") from exc

    try:
        turn_result = await handle_turn(transcript, conversation_id)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Orchestration failed: {exc}") from exc

    try:
        audio_path = await tts.speak(turn_result.response)
    except Exception as exc:
        await record_event(
            "tts_failed", module="vani.tts", data={"error": str(exc)},
            conversation_id=conversation_id, turn_id=turn_result.turn_id,
        )
        raise HTTPException(status_code=502, detail=f"TTS failed: {exc}") from exc

    audio_bytes_out = audio_path.read_bytes()
    media_type = "audio/mpeg" if audio_path.suffix == ".mp3" else "audio/wav"

    return ConverseResponse(
        turn_id=turn_result.turn_id,
        transcript=transcript,
        response_text=turn_result.response,
        task_type=turn_result.task_type,
        skill_used=turn_result.skill_used,
        audio_base64=base64.b64encode(audio_bytes_out).decode("ascii"),
        audio_mime=media_type,
    )
