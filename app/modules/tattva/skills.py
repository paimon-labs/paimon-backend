"""
Tattva — Skill Registry.

A Skill is data, not code: markdown+frontmatter describing a class of
task (name, trigger description, optional tool references, an
instruction body). Unlike Tools, Skills live in Mongo — the desktop
skill-creation workspace (Phase 4) edits them at runtime and they
sync across devices via the cloud backend automatically, with no
deploy and no separate sync mechanism.

Matching (`match_skill`) is deliberately a simple keyword-overlap
scorer for now, not vector similarity — Manas' vector store doesn't
exist until Phase 7, and pulling in embeddings for what's likely a
handful of skills is complexity that hasn't earned its place yet
(YAGNI). The matcher is a swappable single function so it can be
replaced with vector or LLM-based matching later without touching
callers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import yaml
from pydantic import BaseModel, Field, ValidationError

from app.core.db import get_database

COLLECTION = "tattva_skills"

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)


class SkillError(Exception):
    """Raised on malformed skill markdown or a missing skill."""


class SkillFrontmatter(BaseModel):
    name: str
    trigger: str = Field(..., description="Description of when this skill should fire")
    tools: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class Skill:
    name: str
    trigger: str
    tools: list[str]
    body: str
    version: int = 1


def parse_skill_markdown(text: str) -> Skill:
    """Parse a `---\\nyaml frontmatter\\n---\\nmarkdown body` document."""
    match = _FRONTMATTER_RE.match(text.strip() + "\n")
    if not match:
        raise SkillError("Skill markdown must start with a '---' YAML frontmatter block")

    raw_frontmatter, body = match.groups()
    try:
        parsed = yaml.safe_load(raw_frontmatter) or {}
        frontmatter = SkillFrontmatter(**parsed)
    except (yaml.YAMLError, ValidationError) as exc:
        raise SkillError(f"Invalid skill frontmatter: {exc}") from exc

    return Skill(
        name=frontmatter.name,
        trigger=frontmatter.trigger,
        tools=frontmatter.tools,
        body=body.strip(),
    )


def _to_document(skill: Skill) -> dict[str, Any]:
    now = datetime.now(UTC)
    return {
        "_id": skill.name,
        "trigger": skill.trigger,
        "tools": skill.tools,
        "body": skill.body,
        "version": skill.version,
        "updated_at": now,
    }


def _from_document(doc: dict[str, Any]) -> Skill:
    return Skill(
        name=doc["_id"],
        trigger=doc["trigger"],
        tools=doc.get("tools", []),
        body=doc["body"],
        version=doc.get("version", 1),
    )


async def upsert_skill(text: str) -> Skill:
    """Parse markdown and persist it, replacing any existing skill of the same name."""
    skill = parse_skill_markdown(text)
    db = get_database()
    existing = await db[COLLECTION].find_one({"_id": skill.name})
    if existing:
        skill = Skill(name=skill.name, trigger=skill.trigger, tools=skill.tools, body=skill.body, version=existing.get("version", 1) + 1)
    document = _to_document(skill)
    document["created_at"] = existing["created_at"] if existing else document["updated_at"]
    await db[COLLECTION].replace_one({"_id": skill.name}, document, upsert=True)
    return skill


async def get_skill(name: str) -> Skill | None:
    db = get_database()
    doc = await db[COLLECTION].find_one({"_id": name})
    return _from_document(doc) if doc else None


async def list_skills() -> list[Skill]:
    db = get_database()
    docs = await db[COLLECTION].find().to_list(length=None)
    return [_from_document(doc) for doc in docs]


async def delete_skill(name: str) -> bool:
    db = get_database()
    result = await db[COLLECTION].delete_one({"_id": name})
    return result.deleted_count > 0


_STOPWORDS = {"a", "an", "the", "user", "wants", "to", "is", "of", "for", "and", "or", "with", "on", "in"}


def _keywords(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if w not in _STOPWORDS}


def score_match(query: str, skill: Skill) -> float:
    """Fraction of the skill's trigger keywords present in the query. 0..1."""
    trigger_words = _keywords(skill.trigger)
    if not trigger_words:
        return 0.0
    query_words = _keywords(query)
    return len(trigger_words & query_words) / len(trigger_words)


async def match_skill(query: str, threshold: float = 0.3) -> Skill | None:
    """Best-scoring skill above `threshold`, or None. See module docstring
    for why this is keyword overlap rather than vector similarity."""
    skills = await list_skills()
    if not skills:
        return None

    scored = [(score_match(query, s), s) for s in skills]
    best_score, best_skill = max(scored, key=lambda pair: pair[0])
    return best_skill if best_score >= threshold else None
