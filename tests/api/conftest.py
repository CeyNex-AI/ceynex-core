"""Shared fixtures for the API tests.

The rate limiter (SRS 3.4.6) counts per identity in a process-lifetime object.
Under `TestClient` every request arrives from the same client host, so without a
reset the counter would carry across tests and the 31st request in the whole
file — whichever test happened to make it — would get a 429 that has nothing to
do with what that test is asserting. Each test gets a fresh window instead.

The rate limit itself is exercised deliberately in `test_rate_limit.py`.
"""

import pytest

from ceynex.api import rate_limit
from ceynex.api import users as users_module
from ceynex.api.routes import auth as auth_routes
from ceynex.api.routes import news as news_routes
from ceynex.api.routes import query as query_routes


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
def fresh_rate_limit_window():
    # Every endpoint with its own allowance keeps its own window singleton;
    # without resetting each, the counter bleed this fixture exists to prevent
    # comes straight back on the next one. `test_auth.py` alone makes dozens of
    # login/signup calls, so the auth window matters most here.
    query_routes.set_window(rate_limit.InProcessWindow())
    news_routes.set_window(rate_limit.InProcessWindow())
    auth_routes.set_window(rate_limit.InProcessWindow())
    yield
    query_routes.set_window(None)
    news_routes.set_window(None)
    auth_routes.set_window(None)
