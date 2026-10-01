"""Every API route needs a signed-in user, except an explicit, short list.

SRS 3.1.11 / FR-ACC-01: "account based authentication for all users prior to
query submission". Until 2026-10, `POST /api/query`, the stateless chat stream
and the news routes answered anonymous callers. Closing them one by one leaves
the next new route free to reopen the gap quietly, so this walks every router
`main.py` includes and fails on any route that neither requires a user nor is on
the allowlist below. Adding a public route means adding it here, in a diff a
reviewer sees.
"""

from __future__ import annotations

from fastapi.dependencies.models import Dependant
from fastapi.routing import APIRoute

from ceynex.api.routes import (
    account,
    admin,
    auth,
    chat,
    conversations,
    data,
    graph,
    health,
    history,
    news,
    query,
    scenario,
    site,
    usage,
)

ROUTE_MODULES = (
    account,
    admin,
    auth,
    chat,
    conversations,
    data,
    graph,
    health,
    history,
    news,
    query,
    scenario,
    site,
    usage,
)

#: Public on purpose, each for a stated reason.
PUBLIC = {
    ("GET", "/health"),  # the uptime probe and the container healthcheck
    ("POST", "/api/auth/signup"),  # how an account comes to exist
    ("POST", "/api/auth/login"),  # how a token comes to exist
    ("GET", "/api/chat/shared/{token}"),  # a shared link's reader has no account
    ("GET", "/api/graph/expand"),  # that same reader exploring the answer's graph
    ("GET", "/api/site/theme"),  # the sign-in page is themed too
}


def _requires_a_user(dependant: Dependant) -> bool:
    for sub in dependant.dependencies:
        if sub.call in (auth.require_user, auth.require_admin) or _requires_a_user(sub):
            return True
    return False


def _routes() -> list[tuple[str, str, APIRoute]]:
    found = []
    for module in ROUTE_MODULES:
        for route in module.router.routes:
            if isinstance(route, APIRoute):
                for method in sorted(route.methods):
                    found.append((method, route.path, route))
    return found


def test_every_route_module_main_includes_is_walked():
    import ceynex.api.main as main

    included = {name for name in dir(main) if getattr(main, name, None) in ROUTE_MODULES}
    assert len(included) == len(ROUTE_MODULES), "a routes module is missing from this test"


def test_only_the_allowlisted_routes_answer_anonymous_callers():
    anonymous = {
        (method, path) for method, path, route in _routes() if not _requires_a_user(route.dependant)
    }
    assert anonymous == PUBLIC


def test_the_llm_and_news_routes_require_a_user():
    """The four that used to answer anonymous callers, named so that a failure
    says which one reopened."""
    required = {
        (method, path) for method, path, route in _routes() if _requires_a_user(route.dependant)
    }
    for closed in (
        ("POST", "/api/query"),
        ("POST", "/api/chat/stream"),
        ("GET", "/api/news/search"),
        ("GET", "/api/news/trending"),
    ):
        assert closed in required, f"{closed} answers anonymous callers again"
