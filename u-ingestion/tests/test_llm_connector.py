"""Tests for the LLM connector."""

from types import SimpleNamespace

from ingestion.utils import llm_connector


def make_dummy_response(payload):
    """Return an object with the SDK model_dump interface."""

    def model_dump():
        return payload

    return SimpleNamespace(model_dump=model_dump)


def make_openai_factory(created_clients):
    """Build an OpenAI test double factory."""

    def create_openai(api_key, base_url):
        requests = []

        def create_completion(**kwargs):
            requests.append(kwargs)
            return make_dummy_response(
                {"choices": [{"message": {"tool_calls": []}}]}
            )

        client = SimpleNamespace(
            api_key=api_key,
            base_url=base_url,
            requests=requests,
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=create_completion)
            ),
        )
        created_clients.append(client)
        return client

    return create_openai


def make_oauth_client(config):
    """Return a fixed bearer token client."""

    def get_token():
        return "oauth-token"

    return SimpleNamespace(config=config, get_token=get_token)


def test_llm_client_api_key_mode(monkeypatch):
    """Use a static OpenAI client for API-key auth."""
    created_clients = []
    tools = [{"type": "function", "function": {"name": "tool"}}]
    monkeypatch.setattr(
        llm_connector,
        "OpenAI",
        make_openai_factory(created_clients),
    )
    monkeypatch.setattr(llm_connector, "get_auth_mode", lambda: "api_key")
    monkeypatch.setattr(llm_connector, "get_api_key", lambda: "sk-test")
    monkeypatch.setattr(
        llm_connector,
        "get_llm_endpoint",
        lambda: "https://api.example.com/v1",
    )
    monkeypatch.setattr(
        llm_connector,
        "get_stage_model_config",
        lambda _stage: {
            "model": "gpt-5-mini",
            "max_tokens": 123,
            "temperature": 0.2,
            "reasoning_effort": "low",
            "verbosity": "high",
        },
    )

    client = llm_connector.LLMClient()
    response = client.call(
        messages=[{"role": "user", "content": "hi"}],
        tools=tools,
        tool_choice="required",
        context="health",
    )

    request = created_clients[0].requests[0]
    assert client.get_client() is created_clients[0]
    assert request["model"] == "gpt-5-mini"
    assert request["max_completion_tokens"] == 123
    assert request["temperature"] == 0.2
    assert request["reasoning_effort"] == "low"
    assert request["extra_body"] == {
        "text": {"verbosity": "high"},
    }
    assert request["tool_choice"] == "required"
    assert request["tools"][0]["function"]["name"] == "tool"
    assert request["tools"][0]["function"]["strict"] is True
    assert "strict" not in tools[0]["function"]
    assert response == {"choices": [{"message": {"tool_calls": []}}]}


def test_llm_client_oauth_mode(monkeypatch):
    """Create a fresh OpenAI client with the OAuth token."""
    created_clients = []
    monkeypatch.setattr(
        llm_connector,
        "OpenAI",
        make_openai_factory(created_clients),
    )
    monkeypatch.setattr(llm_connector, "OAuthClient", make_oauth_client)
    monkeypatch.setattr(llm_connector, "get_auth_mode", lambda: "oauth")
    monkeypatch.setattr(
        llm_connector,
        "get_oauth_config",
        lambda: {
            "token_endpoint": "https://auth.example.com/token",
            "client_id": "id",
            "client_secret": "secret",
            "scope": "",
        },
    )
    monkeypatch.setattr(
        llm_connector,
        "get_llm_endpoint",
        lambda: "https://api.example.com/v1",
    )
    monkeypatch.setattr(
        llm_connector,
        "get_stage_model_config",
        lambda _stage: {
            "model": "gpt-5",
            "max_tokens": 50,
            "temperature": None,
            "reasoning_effort": None,
        },
    )

    client = llm_connector.LLMClient()
    returned_client = client.get_client()
    client.call(messages=[{"role": "user", "content": "hello"}])

    request = created_clients[-1].requests[0]
    assert returned_client.api_key == "oauth-token"
    assert "temperature" not in request
    assert "reasoning_effort" not in request
    assert "tools" not in request
    assert "tool_choice" not in request


def test_test_connection_success(monkeypatch):
    """Accept health checks only when the model returns a tool call."""
    client = object.__new__(llm_connector.LLMClient)
    monkeypatch.setattr(
        client,
        "call",
        lambda **kwargs: {
            "choices": [{"message": {"tool_calls": [{"id": "1"}]}}]
        },
    )

    assert llm_connector.LLMClient.test_connection(client) is True


def test_test_connection_failure(monkeypatch):
    """Raise when the health check does not return a tool call."""
    client = object.__new__(llm_connector.LLMClient)
    monkeypatch.setattr(
        client,
        "call",
        lambda **kwargs: {"choices": [{"message": {}}]},
    )

    try:
        llm_connector.LLMClient.test_connection(client)
    except RuntimeError as exc:
        assert "tool call" in str(exc)
    else:
        raise AssertionError("RuntimeError not raised")


def test_embed_returns_ordered_vectors(monkeypatch):
    """Return embeddings sorted by index."""
    created_clients = []
    monkeypatch.setattr(
        llm_connector,
        "OpenAI",
        make_openai_factory(created_clients),
    )
    monkeypatch.setattr(llm_connector, "get_auth_mode", lambda: "api_key")
    monkeypatch.setattr(llm_connector, "get_api_key", lambda: "sk-test")
    monkeypatch.setattr(
        llm_connector,
        "get_llm_endpoint",
        lambda: "https://api.example.com/v1",
    )

    client = llm_connector.LLMClient()
    embedding_b = SimpleNamespace(index=1, embedding=[0.4, 0.5])
    embedding_a = SimpleNamespace(index=0, embedding=[0.1, 0.2])
    fake_response = SimpleNamespace(data=[embedding_b, embedding_a])

    created_clients[0].embeddings = SimpleNamespace(
        create=lambda **kwargs: fake_response,
    )

    result = client.embed(
        ["hello", "world"],
        model="text-embedding-3-large",
        dimensions=2,
    )

    assert result == [[0.1, 0.2], [0.4, 0.5]]


def test_embed_returns_empty_for_empty_input(monkeypatch):
    """Return an empty list when no texts are provided."""
    created_clients = []
    monkeypatch.setattr(
        llm_connector,
        "OpenAI",
        make_openai_factory(created_clients),
    )
    monkeypatch.setattr(llm_connector, "get_auth_mode", lambda: "api_key")
    monkeypatch.setattr(llm_connector, "get_api_key", lambda: "sk-test")
    monkeypatch.setattr(
        llm_connector,
        "get_llm_endpoint",
        lambda: "https://api.example.com/v1",
    )

    client = llm_connector.LLMClient()

    assert client.embed([]) == []


def test_llm_client_call_skips_non_function_tool_shapes(monkeypatch):
    """Strict mode is only added to valid function tool dicts."""
    created_clients = []
    monkeypatch.setattr(
        llm_connector,
        "OpenAI",
        make_openai_factory(created_clients),
    )
    monkeypatch.setattr(llm_connector, "get_auth_mode", lambda: "api_key")
    monkeypatch.setattr(llm_connector, "get_api_key", lambda: "sk-test")
    monkeypatch.setattr(
        llm_connector,
        "get_llm_endpoint",
        lambda: "https://api.example.com/v1",
    )
    monkeypatch.setattr(
        llm_connector,
        "get_stage_model_config",
        lambda _stage: {
            "model": "gpt-5-mini",
            "max_tokens": 123,
            "temperature": None,
            "reasoning_effort": None,
        },
    )

    client = llm_connector.LLMClient()
    client.call(
        messages=[{"role": "user", "content": "hi"}],
        tools=[
            {"type": "other", "function": {"name": "skip-me"}},
            {"type": "function", "function": "invalid"},
            {"type": "function", "function": {"name": "keep-me"}},
        ],
    )

    request_tools = created_clients[0].requests[0]["tools"]
    assert request_tools[0]["function"] == {"name": "skip-me"}
    assert request_tools[1]["function"] == "invalid"
    assert request_tools[2]["function"]["strict"] is True
