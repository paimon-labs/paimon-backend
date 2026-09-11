"""
MongoDB Atlas connection layer.

One cluster serves all of Manas (structured, vector via Atlas Vector
Search, and episodic — three collections, one database) plus Sakshi's
event log. See README for why a dedicated vector DB isn't used.
"""

from __future__ import annotations

import logging

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from app.core.config import get_settings

logger = logging.getLogger("paimon.db")
settings = get_settings()

_client: AsyncIOMotorClient | None = None


def get_client() -> AsyncIOMotorClient:
    global _client
    if _client is None:
        if not settings.mongodb_uri:
            raise RuntimeError("MONGODB_URI not configured")
        _client = AsyncIOMotorClient(settings.mongodb_uri)
    return _client


def get_database() -> AsyncIOMotorDatabase:
    return get_client()[settings.mongodb_db_name]


async def ping() -> bool:
    """Health-check helper — used by /health and on startup. Never raises."""
    try:
        await get_client().admin.command("ping")
        return True
    except Exception as exc:  # noqa: BLE001 — deliberate: a health check must not raise
        logger.warning("MongoDB ping failed: %s", exc)
        return False


async def close() -> None:
    global _client
    if _client is not None:
        _client.close()
        _client = None
