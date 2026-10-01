"""Shared fixtures for the API tests.

The rate limiter (SRS 3.4.6) counts per identity in a process-lifetime object.
Under `TestClient` every request arrives from the same client host, so without a
reset the counter would carry across tests and the 31st request in the whole
file — whichever test happened to make it — would get a 429 that has nothing to
do with what that test is asserting. Each test gets a fresh window instead.

The rate limit itself is exercised deliberately in `test_rate_limit.py`.
"""

import pytest

from ceynex.api import auth as auth_module
from ceynex.api import history as history_module
from ceynex.api import rate_limit
from ceynex.api import users as users_module
from ceynex.api.routes import auth as auth_routes
from ceynex.api.routes import chat as chat_routes
from ceynex.api.routes import news as news_routes
from ceynex.api.routes import query as query_routes
from ceynex.api.routes import scenario as scenario_routes


@pytest.fixture(autouse=True)
def forget_known_epochs():
    """`auth.verify_token` remembers each account's last-read epoch for use
    during a Postgres outage. Module state, so cleared for every test."""
    auth_module.forget_known_epochs()
    yield
    auth_module.forget_known_epochs()


@pytest.fixture(autouse=True)
def stub_token_epoch(monkeypatch):
    """Since RBAC's session-invalidation work, `auth.verify_token` calls
    `users.current_token_epoch` once per authed request — a real Postgres
    lookup. The route tests here mint tokens with `issue_token(...)` (epoch 0)
    and never touch a database, so default that lookup to 0 for the whole
    package. Tests that exercise invalidation itself (`test_users.py`,
    `test_auth.py`) re-patch it with a store-aware version in their own
    fixtures — a later monkeypatch wins and both unwind at teardown."""
    monkeypatch.setattr(users_module, "current_token_epoch", lambda email: 0)


@pytest.fixture(autouse=True)
def stub_query_history(request, monkeypatch):
    """`POST /api/query` and the chat stream require sign-in, so every query
    in these tests is attributed and recorded — a real Postgres write. Recording
    is a no-op here; `test_history.py`'s `recorded` fixture re-patches it to
    capture the calls it asserts on. Integration tests keep the real one: they
    exist to exercise it against Postgres."""
    if request.node.get_closest_marker("integration"):
        return
    monkeypatch.setattr(history_module, "record", lambda **_kwargs: None)


@pytest.fixture(autouse=True)
def stub_standing_instructions(request, monkeypatch):
    """Every chat turn now has an owner, so the turn reads the owner's standing
    instruction from Postgres. None here, except in integration tests; the files
    that test instructions (`test_turn_persistence.py`, `test_regenerate.py`)
    re-patch it."""
    if request.node.get_closest_marker("integration"):
        return

    async def none(_user_email):
        return "", True

    monkeypatch.setattr("ceynex.chat.instructions.get", none)


@pytest.fixture(autouse=True)
def fresh_rate_limit_window():
    # Every endpoint with its own allowance keeps its own window singleton;
    # without resetting each, the counter bleed this fixture exists to prevent
    # comes straight back on the next one. `test_auth.py` alone makes dozens of
    # login/signup calls, so the auth window matters most here, and a
    # `chat:`-namespaced counter that survived between tests would 429 the 46th
    # conversational turn in the file (D13), whichever test happened to make it.
    query_routes.set_window(rate_limit.InProcessWindow())
    news_routes.set_window(rate_limit.InProcessWindow())
    auth_routes.set_window(rate_limit.InProcessWindow())
    chat_routes.set_chat_window(rate_limit.InProcessWindow())
    scenario_routes.set_window(rate_limit.InProcessWindow())
    yield
    query_routes.set_window(None)
    news_routes.set_window(None)
    auth_routes.set_window(None)
    chat_routes.set_chat_window(None)
    scenario_routes.set_window(None)
