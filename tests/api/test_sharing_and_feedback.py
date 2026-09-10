"""Assertions for answer feedback and shared conversations (execution plan §5).

Both features hand out access to something, so both are tested for what they
*refuse*: rating a message that is not yours, and reading a conversation nobody
shared.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from ceynex.api.auth import DemoUser, issue_token
from ceynex.api.main import app
from ceynex.api.routes import conversations as conv_routes

OWNER = "policymaker@ceynex.dev"
OTHER = "researcher@ceynex.dev"


def auth(email: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {issue_token(DemoUser(email=email, role='policymaker', password_hash=b''))}"
    }


class ShareStore:
    """In-memory, but ownership-enforcing — the property under test."""

    def __init__(self):
        self.tokens: dict[int, str | None] = {}
        self.owners = {1: OWNER}
        self.feedback: list[tuple] = []

    async def owns(self, conversation_id, user_email):
        return self.owners.get(conversation_id) == user_email

    async def set_share_token(self, conversation_id, user_email, token):
        if self.owners.get(conversation_id) != user_email:
            return None
        self.tokens[conversation_id] = token
        return token

    async def shared_conversation(self, token):
        for cid, value in self.tokens.items():
            if value is not None and value == token:
                return {"id": cid, "title": "T", "created_at": "now", "messages": []}
        return None

    async def record_feedback(self, message_id, user_email, rating, reason=""):
        # Mirrors the real query's join: only the owner's own message matches.
        if user_email != OWNER or message_id != 7:
            return False
        self.feedback.append((message_id, user_email, rating, reason))
        return True


@pytest.fixture
def store(monkeypatch):
    fake = ShareStore()
    for name in ("owns", "set_share_token", "shared_conversation", "record_feedback"):
        monkeypatch.setattr(conv_routes.store, name, getattr(fake, name))
    return fake


def test_rating_your_own_answer_is_recorded(store):
    client = TestClient(app)
    response = client.post(
        "/api/chat/messages/7/feedback", json={"rating": -1, "reason": "wrong year"},
        headers=auth(OWNER),
    )
    assert response.status_code == 200
    assert store.feedback == [(7, OWNER, -1, "wrong year")]


def test_rating_someone_elses_answer_is_a_404(store):
    """A BIGSERIAL is guessable, and rating someone else's answer would poison
    exactly the eval data this feature exists to collect."""
    client = TestClient(app)
    assert client.post(
        "/api/chat/messages/7/feedback", json={"rating": 1}, headers=auth(OTHER)
    ).status_code == 404


def test_a_rating_outside_the_scale_is_rejected(store):
    client = TestClient(app)
    assert client.post(
        "/api/chat/messages/7/feedback", json={"rating": 5}, headers=auth(OWNER)
    ).status_code == 422


def test_a_conversation_is_not_shared_until_it_is_shared(store):
    """NULL until asked for: the safe state is the default state."""
    client = TestClient(app)
    assert client.get("/api/chat/shared/anything").status_code == 404


def test_sharing_mints_an_unguessable_token_and_revoking_removes_it(store):
    client = TestClient(app)
    minted = client.post(
        "/api/chat/conversations/1/share", json={"shared": True}, headers=auth(OWNER)
    ).json()
    token = minted["token"]
    assert minted["shared"] is True
    # Not the conversation id, and not countable.
    assert token and len(token) >= 20 and token != "1"
    assert client.get(f"/api/chat/shared/{token}").status_code == 200

    revoked = client.post(
        "/api/chat/conversations/1/share", json={"shared": False}, headers=auth(OWNER)
    ).json()
    assert revoked["token"] is None
    assert client.get(f"/api/chat/shared/{token}").status_code == 404


def test_only_the_owner_can_share_a_conversation(store):
    client = TestClient(app)
    assert client.post(
        "/api/chat/conversations/1/share", json={"shared": True}, headers=auth(OTHER)
    ).status_code == 404


def test_a_shared_transcript_carries_no_identity(store):
    """It shows what the system did, not who asked."""
    client = TestClient(app)
    token = client.post(
        "/api/chat/conversations/1/share", json={"shared": True}, headers=auth(OWNER)
    ).json()["token"]
    body = client.get(f"/api/chat/shared/{token}").json()
    assert "user_email" not in body
    assert OWNER not in str(body)
