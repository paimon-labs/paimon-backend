"""
Narada — orchestrator. This is the "core agent loop" the plan
describes: one user message in, one turn of recorded reasoning out.

    parse @override -> match Skill Registry -> classify task_type
    (skipped if a skill matched — the skill name IS the task_type
    bucket) -> get/create priority order -> call the LLM chain in
    that order -> record every step into Sakshi under one turn_id

Known gap, stated plainly rather than glossed over: this does NOT yet
give the LLM a tool-calling loop (the model autonomously deciding to
invoke a Tattva tool mid-turn). A matched skill's `tools` list is
surfaced to the model as context/instructions only — nothing here
executes them automatically. Wiring real function-calling across all
three providers is real work and deliberately deferred rather than
half-built into something fragile. Track this as the next thing after
Phase 3's other pieces land.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from app.modules.narada import priority
from app.modules.narada.llm import complete
from app.modules.sakshi.logger import record_event
from app.modules.tattva.skills import Skill, match_skill

SYSTEM_PREAMBLE = "You are PAIMON, a personal agentic assistant. Be direct and concise."


@dataclass(frozen=True)
class TurnResult:
    turn_id: str
    response: str
    task_type: str
    skill_used: str | None
    provider_order: list[str]


def _build_messages(user_text: str, skill: Skill | None) -> list[dict]:
    system = SYSTEM_PREAMBLE
    if skill:
        tool_note = f" Relevant tools you may reference: {', '.join(skill.tools)}." if skill.tools else ""
        system += f"\n\nMatched skill '{skill.name}':\n{skill.body}{tool_note}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user_text}]


async def handle_turn(text: str, conversation_id: str) -> TurnResult:
    turn_id = str(uuid.uuid4())

    await record_event(
        "user_message", module="narada.orchestrator", data={"text": text},
        conversation_id=conversation_id, turn_id=turn_id,
    )

    forced_provider, remaining_text = priority.parse_override(text)

    skill = await match_skill(remaining_text)
    if skill:
        await record_event(
            "skill_match", module="narada.orchestrator", data={"skill": skill.name},
            conversation_id=conversation_id, turn_id=turn_id,
        )

    task_type = skill.name if skill else priority.classify_task_type(remaining_text)

    if forced_provider:
        provider_order = [forced_provider]
    else:
        provider_order = await priority.get_or_create_priority(task_type)

    await record_event(
        "routing_decision", module="narada.orchestrator",
        data={"task_type": task_type, "provider_order": provider_order, "forced": forced_provider is not None},
        conversation_id=conversation_id, turn_id=turn_id,
    )

    messages = _build_messages(remaining_text, skill)

    try:
        response = await complete(messages, provider_order=provider_order)
    except Exception as exc:
        await record_event(
            "llm_completion_failed", module="narada.orchestrator", data={"error": str(exc)},
            conversation_id=conversation_id, turn_id=turn_id,
        )
        raise

    await record_event(
        "final_answer", module="narada.orchestrator", data={"text": response},
        conversation_id=conversation_id, turn_id=turn_id,
    )

    return TurnResult(
        turn_id=turn_id,
        response=response,
        task_type=task_type,
        skill_used=skill.name if skill else None,
        provider_order=provider_order,
    )
