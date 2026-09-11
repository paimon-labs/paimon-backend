"""
Verifies Tattva's Skill Registry: frontmatter parsing, CRUD against a
mocked Mongo collection, and the keyword-overlap trigger matcher.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.modules.tattva.skills import (
    Skill,
    SkillError,
    match_skill,
    parse_skill_markdown,
    score_match,
    upsert_skill,
)

SAMPLE_MARKDOWN = """---
name: summarize-pr
trigger: user wants a pull request summarized or reviewed
tools: [github_get_pr]
---

Fetch the PR diff with `github_get_pr`, then write a 3-bullet summary.
"""


def test_parse_skill_markdown():
    skill = parse_skill_markdown(SAMPLE_MARKDOWN)
    assert skill.name == "summarize-pr"
    assert skill.tools == ["github_get_pr"]
    assert "3-bullet summary" in skill.body


def test_parse_skill_markdown_missing_frontmatter_raises():
    with pytest.raises(SkillError):
        parse_skill_markdown("just a body, no frontmatter")


def test_parse_skill_markdown_invalid_fields_raises():
    with pytest.raises(SkillError):
        parse_skill_markdown("---\nname: only-name\n---\nbody")  # missing required 'trigger'


def test_score_match_full_overlap():
    skill = Skill(name="s", trigger="summarize a pull request", tools=[], body="")
    assert score_match("please summarize this pull request", skill) == 1.0


def test_score_match_no_overlap():
    skill = Skill(name="s", trigger="summarize a pull request", tools=[], body="")
    assert score_match("what's the weather today", skill) == 0.0


@pytest.mark.asyncio
async def test_upsert_skill_new_sets_version_one():
    fake_collection = MagicMock()
    fake_collection.find_one = AsyncMock(return_value=None)
    fake_collection.replace_one = AsyncMock()
    fake_db = {"tattva_skills": fake_collection}

    with patch("app.modules.tattva.skills.get_database", return_value=fake_db):
        skill = await upsert_skill(SAMPLE_MARKDOWN)

    assert skill.version == 1
    fake_collection.replace_one.assert_awaited_once()


@pytest.mark.asyncio
async def test_upsert_skill_existing_increments_version():
    fake_collection = MagicMock()
    fake_collection.find_one = AsyncMock(return_value={"_id": "summarize-pr", "version": 4, "created_at": "then"})
    fake_collection.replace_one = AsyncMock()
    fake_db = {"tattva_skills": fake_collection}

    with patch("app.modules.tattva.skills.get_database", return_value=fake_db):
        skill = await upsert_skill(SAMPLE_MARKDOWN)

    assert skill.version == 5


@pytest.mark.asyncio
async def test_match_skill_returns_best_above_threshold():
    skill_a = Skill(name="a", trigger="summarize a pull request", tools=[], body="")
    skill_b = Skill(name="b", trigger="get the current weather forecast", tools=[], body="")

    with patch("app.modules.tattva.skills.list_skills", new=AsyncMock(return_value=[skill_a, skill_b])):
        result = await match_skill("can you summarize this pull request for me")

    assert result is not None
    assert result.name == "a"


@pytest.mark.asyncio
async def test_match_skill_returns_none_below_threshold():
    skill_a = Skill(name="a", trigger="summarize a pull request", tools=[], body="")

    with patch("app.modules.tattva.skills.list_skills", new=AsyncMock(return_value=[skill_a])):
        result = await match_skill("what time is it in tokyo")

    assert result is None
