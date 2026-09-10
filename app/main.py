"""paimon-backend — FastAPI entrypoint."""

from fastapi import FastAPI

from app.core.config import get_settings
from app.modules.vani.router import router as vani_router

settings = get_settings()

app = FastAPI(
    title="paimon-backend",
    description="PAIMON cloud orchestrator — Vani, Narada, Tattva, Manas, Sakshi",
    version="0.1.0",
)

app.include_router(vani_router)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "environment": settings.environment}
