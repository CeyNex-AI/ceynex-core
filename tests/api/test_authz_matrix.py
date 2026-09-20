"""The complete admin authorization matrix, and JWT algorithm-downgrade defence.

Written after a 2026-09-19 security pass attack-tested the endpoints the
15-Sep production ZAP scan had excluded. The attacks found no defect, but they
did surface a *test* gap worth closing so it stays closed:

`test_admin.py::ADMIN_ROUTES` gated only eight of the fourteen admin routes and
exercised only the researcher role. The six it omitted are the
`/api/admin/users/*` user-management routes -- the highest-blast-radius admin
surface, the one that can disable, reassign or reset a real account. A role-gate
regression on exactly those routes would not have failed a single test.

This file exercises every admin route against every non-admin role plus the
anonymous caller, and guards the one JWT bypass class the existing auth tests do
not name explicitly: an `alg=none` token, which `verify_token` rejects only
because it pins `algorithms=["HS256"]`. All of it runs in the unit suite: the
role gate (`require_admin`) rejects on the token's own claim, before any route
body or database access.
"""

from __future__ import annotations

import base64
import json

import pytest
from fastapi.testclient import TestClient

from ceynex.api.auth import issue_token
from ceynex.api.main import app

# Every admin route (SRS 3.5.4), with a request body valid enough that the only
# thing left to fail is the role gate -- so a non-admin gets 403, never a 422
# that would mask a missing gate. Kept complete on purpose; compare against
# ceynex/api/routes/admin.py's @router decorators if a route is added.
ADMIN_ROUTES: list[tuple[str, str, dict | None]] = [
    ("GET", "/api/admin/models", None),
    ("POST", "/api/admin/retrain", {"sector": "agriculture", "item": "cinnamon"}),
    ("POST", "/api/admin/pipeline/ingest", {}),
    ("GET", "/api/admin/pipeline/status", None),
    ("GET", "/api/admin/dq-flags", None),
    ("POST", "/api/admin/dq-flags/1/resolve", None),
    ("GET", "/api/admin/llm/status", None),
    ("GET", "/api/admin/audit-log", None),
    # The user-management surface test_admin.py's own gate test omits:
    ("GET", "/api/admin/users", None),
    (
        "POST",
        "/api/admin/users",
        {"email": "x@ceynex.dev", "password": "Abcdefg1!", "role": "researcher"},
    ),
    ("POST", "/api/admin/users/1/role", {"role": "admin"}),
    ("POST", "/api/admin/users/1/disable", None),
    ("POST", "/api/admin/users/1/enable", None),
    ("POST", "/api/admin/users/1/password", {"password": "Abcdefg1!"}),
]

NON_ADMIN_ROLES = ["researcher", "exporter", "policymaker"]


@pytest.fixture
def client():
    return TestClient(app)


def _headers(email: str, role: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {issue_token(email, role)}"}


@pytest.mark.parametrize("method,path,body", ADMIN_ROUTES)
@pytest.mark.parametrize("role", NON_ADMIN_ROLES)
def test_admin_route_is_forbidden_to_every_non_admin_role(client, method, path, body, role):
    resp = client.request(method, path, json=body, headers=_headers(f"{role}@ceynex.dev", role))
    assert resp.status_code == 403, f"{role} reached {method} {path} (got {resp.status_code})"


@pytest.mark.parametrize("method,path,body", ADMIN_ROUTES)
def test_admin_route_is_unauthorized_without_a_token(client, method, path, body):
    resp = client.request(method, path, json=body)
    assert resp.status_code == 401, f"anon reached {method} {path} (got {resp.status_code})"


def _alg_none_token(role: str = "admin") -> str:
    """A forged token with `"alg": "none"` and no signature -- the classic JWT
    downgrade. A verifier that does not pin its algorithms accepts it."""

    def b64(obj: dict) -> str:
        return base64.urlsafe_b64encode(json.dumps(obj).encode()).rstrip(b"=").decode()

    header = b64({"alg": "none", "typ": "JWT"})
    payload = b64({"sub": "attacker@ceynex.dev", "role": role, "iat": 0, "exp": 9_999_999_999})
    return f"{header}.{payload}."


def test_an_alg_none_token_is_rejected(client):
    resp = client.get("/api/auth/me", headers={"Authorization": f"Bearer {_alg_none_token()}"})
    assert resp.status_code == 401


def test_an_alg_none_token_cannot_reach_an_admin_route(client):
    resp = client.get("/api/admin/users", headers={"Authorization": f"Bearer {_alg_none_token()}"})
    assert resp.status_code == 401
