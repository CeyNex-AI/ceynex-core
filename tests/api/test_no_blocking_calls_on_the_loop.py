"""No route's `async def` makes a blocking database or bcrypt call directly.

An `async def` handler runs on the event loop, which on a uvicorn worker serves
every other request too, the chat streams' heartbeats included. A synchronous
psycopg connect or a bcrypt hash inside one stalls all of them (SAD C14). Such
calls belong in a plain `def` handler, which FastAPI runs in its thread pool,
or behind `await asyncio.to_thread(...)`.

This walks the route modules' source and fails on a direct call, inside an
`async def`, to the stores' synchronous functions. Passing one to
`asyncio.to_thread` is not a call and is fine.
"""

from __future__ import annotations

import ast
from pathlib import Path

import ceynex.api.routes as routes_package

#: Modules whose public functions are synchronous Postgres (and bcrypt) calls.
BLOCKING_MODULES = {"users", "history", "preferences", "api_keys", "audit", "site_settings"}
#: Route-module helpers and imported functions that wrap the same calls.
BLOCKING_NAMES = {"authenticate", "_check_postgres", "_require_current_password", "_freshness_sync"}


def _violations(path: Path) -> list[str]:
    found = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if not isinstance(node, ast.AsyncFunctionDef):
            continue
        for call in ast.walk(node):
            if not isinstance(call, ast.Call):
                continue
            func = call.func
            direct = (
                isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and func.value.id in BLOCKING_MODULES
            ) or (isinstance(func, ast.Name) and func.id in BLOCKING_NAMES)
            if direct:
                found.append(f"{path.name}:{call.lineno} {ast.unparse(func)}() in async def {node.name}")
    return found


def test_no_async_route_calls_a_blocking_store_function_directly():
    directory = Path(routes_package.__file__).parent
    violations = [v for path in sorted(directory.glob("*.py")) for v in _violations(path)]
    assert violations == []


def test_the_check_would_catch_one():
    source = "async def handler():\n    users.create_user('a', 'b', 'c')\n"
    path = Path(__file__).with_name("_blocking_probe.py")
    path.write_text(source, encoding="utf-8")
    try:
        assert _violations(path) == ["_blocking_probe.py:2 users.create_user() in async def handler"]
    finally:
        path.unlink()
