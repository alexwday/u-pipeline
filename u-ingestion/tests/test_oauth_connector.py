"""Tests for OAuth token management."""

import time

from ingestion.utils import oauth_connector
from ingestion.utils.oauth_connector import (
    _should_retry_with_body_credentials as should_retry_with_body_credentials,
)


class DummyResponse:
    """Minimal requests.Response stand-in for OAuth tests."""

    def __init__(
        self,
        *,
        status_code=200,
        json_data=None,
        json_error=None,
        raise_error=None,
    ):
        self.status_code = status_code
        self._json_data = json_data or {}
        self._json_error = json_error
        self._raise_error = raise_error

    def json(self):
        """Return configured JSON or raise."""
        if self._json_error is not None:
            raise self._json_error
        return self._json_data

    def raise_for_status(self) -> None:
        """Raise the configured HTTP error."""
        if self._raise_error is not None:
            raise self._raise_error


def test_should_retry_with_body_credentials():
    """Retry only when the error suggests client-auth fallback."""
    assert (
        should_retry_with_body_credentials(DummyResponse(status_code=401))
        is False
    )
    assert (
        should_retry_with_body_credentials(
            DummyResponse(status_code=400, json_error=ValueError("bad json"))
        )
        is False
    )
    assert (
        should_retry_with_body_credentials(
            DummyResponse(
                status_code=400,
                json_data={"error": "invalid_client"},
            )
        )
        is True
    )
    assert (
        should_retry_with_body_credentials(
            DummyResponse(
                status_code=400,
                json_data={"error_description": "send client_secret in body"},
            )
        )
        is True
    )


def test_get_token_and_expiry(monkeypatch):
    """Refresh tokens only when missing or near expiry."""
    client = oauth_connector.OAuthClient(
        config={
            "token_endpoint": "https://auth.example.com/token",
            "client_id": "id",
            "client_secret": "secret",
            "scope": "",
        }
    )

    assert client.is_expired() is True

    calls = []

    def fake_fetch():
        calls.append("fetch")
        setattr(client, "_access_token", "token")
        setattr(client, "_expires_at", time.time() + 600)

    monkeypatch.setattr(client, "_fetch_token", fake_fetch)

    assert client.get_token() == "token"
    assert calls == ["fetch"]
    assert client.is_expired() is False


def test_fetch_token_with_basic_auth(monkeypatch):
    """Fetch a token successfully on the first request."""
    calls = []
    response = DummyResponse(
        json_data={"access_token": "abc", "expires_in": 60}
    )

    def fake_post(url, **kwargs):
        calls.append((url, kwargs))
        return response

    monkeypatch.setattr(oauth_connector.requests, "post", fake_post)

    client = oauth_connector.OAuthClient(
        config={
            "token_endpoint": "https://auth.example.com/token",
            "client_id": "id",
            "client_secret": "secret",
            "scope": "scope",
        }
    )
    getattr(client, "_fetch_token")()

    assert getattr(client, "_access_token") == "abc"
    assert calls[0][1]["data"] == {
        "grant_type": "client_credentials",
        "scope": "scope",
    }
    assert calls[0][1]["auth"].username == "id"


def test_fetch_token_retries_with_body_credentials(monkeypatch):
    """Retry with request-body credentials when basic auth fails."""
    calls = []
    responses = [
        DummyResponse(
            status_code=400,
            json_data={"error": "invalid_client"},
        ),
        DummyResponse(
            json_data={"access_token": "retry-token", "expires_in": 30}
        ),
    ]

    def fake_post(url, **kwargs):
        calls.append((url, kwargs))
        return responses.pop(0)

    monkeypatch.setattr(oauth_connector.requests, "post", fake_post)

    client = oauth_connector.OAuthClient(
        config={
            "token_endpoint": "https://auth.example.com/token",
            "client_id": "id",
            "client_secret": "secret",
            "scope": "",
        }
    )
    getattr(client, "_fetch_token")()

    assert getattr(client, "_access_token") == "retry-token"
    assert "auth" in calls[0][1]
    assert calls[1][1]["data"]["client_id"] == "id"
    assert calls[1][1]["data"]["client_secret"] == "secret"
