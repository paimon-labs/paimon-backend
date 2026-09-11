"""paimon-backend — FastAPI entrypoint."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core import db
from app.core.config import get_settings
from app.modules import tattva  # noqa: F401 — import registers built-in tools
from app.modules.narada.router import router as narada_router
from app.modules.tattva.router import router as tattva_router
from app.modules.vani.router import router as vani_router

settings = get_settings()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Startup: nothing to eagerly connect — Motor connects lazily on
    # first use. Shutdown: close the Mongo client cleanly.
    yield
    await db.close()


app = FastAPI(
    title="paimon-backend",
    description="PAIMON cloud orchestrator — Vani, Narada, Tattva, Manas, Sakshi",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(vani_router)
app.include_router(narada_router)
app.include_router(tattva_router)


@app.get("/health")
async def health() -> dict:
    db_ok = await db.ping() if settings.mongodb_uri else None
    return {
        "status": "ok",
        "environment": settings.environment,
        "mongodb": db_ok,  # True/False if configured, null if MONGODB_URI unset
    }
