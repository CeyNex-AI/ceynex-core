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
from ceynex.api.routes import chat as chat_routes
from ceynex.api.routes import news as news_routes
from ceynex.api.routes import query as query_routes


@pytest.fixture(autouse=True)
def fresh_rate_limit_window():
    # The news sidecar (D11) has its own allowance and its own window singleton;
    # without resetting it too, the counter bleed this fixture exists to prevent
    # comes straight back on the second endpoint.
    # ...and the conversational surface (D13) has a third, for the same reason:
    # a `chat:`-namespaced counter that survives between tests would 429 the
    # 46th turn in the file, whichever test happened to make it.
    query_routes.set_window(rate_limit.InProcessWindow())
    news_routes.set_window(rate_limit.InProcessWindow())
    chat_routes.set_chat_window(rate_limit.InProcessWindow())
    yield
    query_routes.set_window(None)
    news_routes.set_window(None)
    chat_routes.set_chat_window(None)
