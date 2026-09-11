"""
Narada — task routing, device communication, the task->AI
priority/fallback table, device pairing.

`llm.py` (Phase 2): the LLM provider chain (NVIDIA build -> Groq ->
local Ollama). The dynamic per-task-type priority table and device
pairing are still Phase 3+.
"""
