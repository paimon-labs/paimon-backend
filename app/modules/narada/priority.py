"""
Narada — task->AI priority/fallback table.

The first time a given task_type is encountered, establish and
persist the provider preference order for it going forward (Mongo
collection narada_priority_table, keyed by task_type). Starts from
the existing fixed NVIDIA->Groq->local-Ollama chain as the default —
Phase 14 is where this gets smarter based on real cost/latency data;
for now it's deliberately simple: classify, look up or create, done.

Task classification is a small keyword heuristic, not an LLM call —
spending a model call just to decide which model to call would be
backwards. It only needs to be good enough to bucket task types
consistently, not perfectly accurate.

Manual override syntax: a leading "@provider" token in the user's
message (e.g. "@groq summarize this") skips both classification and
the table entirely and forces that one provider. Unknown provider
names are NOT treated as an override — "@home is nice" stays intact
as a normal message rather than being silently mangled.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime

from app.core.db import get_database
from app.modules.narada.llm import llm_chain

COLLECTION = "narada_priority_table"

DEFAULT_ORDER = [p.name for p in llm_chain.providers]
KNOWN_PROVIDERS = set(DEFAULT_ORDER)

_OVERRIDE_RE = re.compile(r"^@(?P<provider>[\w-]+)\s+(?P<rest>.+)$", re.DOTALL)

# Ordered (label, keywords) pairs — first match wins. Deliberately
# coarse; add buckets here as real usage reveals distinct task types
# rather than trying to anticipate all of them upfront.
_TASK_KEYWORDS: list[tuple[str, set[str]]] = [
    ("coding", {"code", "function", "bug", "error", "debug", "script", "refactor", "python", "javascript"}),
    ("summarization", {"summarize", "summary", "tldr", "recap", "condense"}),
    ("creative", {"story", "poem", "write", "creative", "imagine", "brainstorm"}),
    ("factual_qa", {"what", "who", "when", "where", "why", "how", "explain", "define"}),
]


def classify_task_type(text: str) -> str:
    words = set(re.findall(r"[a-z0-9]+", text.lower()))
    for label, keywords in _TASK_KEYWORDS:
        if words & keywords:
            return label
    return "general"


def parse_override(text: str) -> tuple[str | None, str]:
    """Returns (provider_name_or_None, remaining_text)."""
    match = _OVERRIDE_RE.match(text.strip())
    if not match:
        return None, text
    provider = match.group("provider")
    if provider not in KNOWN_PROVIDERS:
        return None, text
    return provider, match.group("rest").strip()


async def get_or_create_priority(task_type: str) -> list[str]:
    db = get_database()
    doc = await db[COLLECTION].find_one({"_id": task_type})
    if doc:
        return doc["order"]

    await db[COLLECTION].insert_one(
        {"_id": task_type, "order": DEFAULT_ORDER, "created_at": datetime.now(UTC)}
    )
    return DEFAULT_ORDER


async def set_priority(task_type: str, order: list[str]) -> list[str]:
    """Manual override of a task type's provider order (e.g. via an
    admin endpoint) — kept separate from the auto-adjusting logic
    Phase 14 will add, so a person can always just set it directly."""
    unknown = [name for name in order if name not in KNOWN_PROVIDERS]
    if unknown:
        raise ValueError(f"Unknown provider(s): {unknown}. Known: {sorted(KNOWN_PROVIDERS)}")

    db = get_database()
    await db[COLLECTION].replace_one(
        {"_id": task_type},
        {"_id": task_type, "order": order, "updated_at": datetime.now(UTC)},
        upsert=True,
    )
    return order
