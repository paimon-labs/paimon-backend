"""FastAPI routes exposing Vani's listen()/speak() round trip."""

from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

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
