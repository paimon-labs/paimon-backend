"""
Tattva (तत्त्व) — Tool registry, Skill registry, sandbox, meta-tool
(write-new-tools).

Phase 3 status: Tool interface + registry (registry.py), execution
wrapper with Sakshi event recording (executor.py), and a couple of
built-in test tools (builtin.py) are in. Skill Registry, sandbox, and
the meta-tool are still Phase 3+/Phase 12.

Importing this package registers the built-in tools as a side effect
— anything that needs the registry populated (main.py, tests) should
import `app.modules.tattva` before calling into it.
"""

from app.modules.tattva import builtin  # noqa: F401 — import for registration side effect
