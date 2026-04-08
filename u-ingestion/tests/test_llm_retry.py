"""Tests for the shared LLM retry wrapper."""

import pytest

from ingestion.utils import llm_retry


class _DummyRetryable(Exception):
    """Retryable stand-in for openai transport errors."""


def _stub_prompt():
    """Build a minimal prompt dict for tests."""
    return {
        "tools": [{"type": "function"}],
        "tool_choice": "required",
    }


def _make_llm(responses, record=None):
    """Build a fake LLM client that returns a sequence of responses."""
    state = {"idx": 0}

    def fake_call(**kwargs):
        if record is not None:
            record.append(kwargs)
        idx = state["idx"]
        state["idx"] += 1
        value = responses[idx]
        if isinstance(value, Exception):
            raise value
        return value

    return type(
        "FakeLLM",
        (),
        {"call": staticmethod(fake_call), "__doc__": "Fake."},
    )()


def test_call_with_retry_returns_parsed_result_on_first_attempt(monkeypatch):
    """Happy path: single call, parser returns, no retry."""
    monkeypatch.setattr(llm_retry.time, "sleep", lambda _s: None)
    llm = _make_llm([{"data": "ok"}])

    result = llm_retry.call_with_retry(
        llm,
        [{"role": "user", "content": "hi"}],
        _stub_prompt(),
        parser=lambda response: response["data"],
        stage="test_stage",
        context="test:single",
        max_retries=3,
        retry_delay=0.0,
    )

    assert result == "ok"


def test_call_with_retry_retries_on_value_error(monkeypatch):
    """Parser raises ValueError on first attempt, succeeds on second."""
    monkeypatch.setattr(llm_retry.time, "sleep", lambda _s: None)
    llm = _make_llm([{"data": "bad"}, {"data": "ok"}])
    attempt = [0]

    def parser(response):
        attempt[0] += 1
        if response["data"] == "bad":
            raise ValueError("structural failure")
        return response["data"]

    result = llm_retry.call_with_retry(
        llm,
        [{"role": "user", "content": "hi"}],
        _stub_prompt(),
        parser=parser,
        stage="test_stage",
        context="test:retry",
        max_retries=3,
        retry_delay=0.0,
    )

    assert result == "ok"
    assert attempt[0] == 2


def test_call_with_retry_retries_on_transport_error(monkeypatch):
    """Transport error on first attempt, valid response on second."""
    monkeypatch.setattr(llm_retry.time, "sleep", lambda _s: None)
    monkeypatch.setattr(
        llm_retry, "RETRYABLE_ERRORS", (_DummyRetryable, ValueError)
    )
    llm = _make_llm([_DummyRetryable("flake"), {"data": "ok"}])

    result = llm_retry.call_with_retry(
        llm,
        [{"role": "user", "content": "hi"}],
        _stub_prompt(),
        parser=lambda response: response["data"],
        stage="test_stage",
        context="test:transport",
        max_retries=3,
        retry_delay=0.0,
    )

    assert result == "ok"


def test_call_with_retry_retries_on_validator_failure(monkeypatch):
    """Validator raises ValueError on first attempt, passes on second."""
    monkeypatch.setattr(llm_retry.time, "sleep", lambda _s: None)
    llm = _make_llm([{"ids": ["1", "1"]}, {"ids": ["1"]}])

    def validator(parsed):
        if len(set(parsed)) != len(parsed):
            raise ValueError("duplicate ids")

    result = llm_retry.call_with_retry(
        llm,
        [{"role": "user", "content": "hi"}],
        _stub_prompt(),
        parser=lambda response: response["ids"],
        stage="test_stage",
        context="test:validator",
        max_retries=3,
        retry_delay=0.0,
        validator=validator,
    )

    assert result == ["1"]


def test_call_with_retry_raises_after_all_attempts_fail(monkeypatch):
    """All attempts fail with ValueError; final call raises."""
    monkeypatch.setattr(llm_retry.time, "sleep", lambda _s: None)
    llm = _make_llm([{"data": "bad"}, {"data": "bad"}])

    def parser(response):
        raise ValueError(f"bad response: {response}")

    with pytest.raises(ValueError, match="bad response"):
        llm_retry.call_with_retry(
            llm,
            [{"role": "user", "content": "hi"}],
            _stub_prompt(),
            parser=parser,
            stage="test_stage",
            context="test:exhausted",
            max_retries=2,
            retry_delay=0.0,
        )


def test_call_with_retry_does_not_retry_on_non_retryable(monkeypatch):
    """Non-retryable exception propagates without retrying."""
    monkeypatch.setattr(llm_retry.time, "sleep", lambda _s: None)
    llm = _make_llm([KeyError("surprise"), {"data": "ok"}])

    with pytest.raises(KeyError):
        llm_retry.call_with_retry(
            llm,
            [{"role": "user", "content": "hi"}],
            _stub_prompt(),
            parser=lambda response: response["data"],
            stage="test_stage",
            context="test:non_retryable",
            max_retries=3,
            retry_delay=0.0,
        )


def test_call_with_retry_appends_attempt_number_to_context(monkeypatch):
    """Each attempt's llm.call context is tagged with :attempt_N."""
    monkeypatch.setattr(llm_retry.time, "sleep", lambda _s: None)
    calls = []
    llm = _make_llm(
        [{"data": "bad"}, {"data": "ok"}],
        record=calls,
    )

    def parser(response):
        if response["data"] == "bad":
            raise ValueError("nope")
        return response["data"]

    llm_retry.call_with_retry(
        llm,
        [{"role": "user", "content": "hi"}],
        _stub_prompt(),
        parser=parser,
        stage="test_stage",
        context="test:context",
        max_retries=3,
        retry_delay=0.0,
    )

    contexts = [call["context"] for call in calls]
    assert contexts == ["test:context:attempt_1", "test:context:attempt_2"]


def test_call_with_retry_passes_stage_and_tools(monkeypatch):
    """LLM call receives the stage name, tools, and tool_choice."""
    monkeypatch.setattr(llm_retry.time, "sleep", lambda _s: None)
    calls = []
    llm = _make_llm([{"data": "ok"}], record=calls)

    llm_retry.call_with_retry(
        llm,
        [{"role": "user", "content": "hi"}],
        _stub_prompt(),
        parser=lambda response: response["data"],
        stage="test_stage",
        context="test:args",
        max_retries=1,
        retry_delay=0.0,
    )

    assert calls[0]["stage"] == "test_stage"
    assert calls[0]["tools"] == [{"type": "function"}]
    assert calls[0]["tool_choice"] == "required"


def test_call_with_retry_raises_when_retry_loop_never_runs(monkeypatch):
    """Defensive: max_retries=0 falls through and raises RuntimeError."""
    monkeypatch.setattr(llm_retry.time, "sleep", lambda _s: None)
    llm = _make_llm([{"data": "ok"}])

    with pytest.raises(
        RuntimeError, match="exited retry loop without a response"
    ):
        llm_retry.call_with_retry(
            llm,
            [{"role": "user", "content": "hi"}],
            _stub_prompt(),
            parser=lambda response: response["data"],
            stage="test_stage",
            context="test:zero",
            max_retries=0,
            retry_delay=0.0,
        )


def test_call_with_retry_linear_backoff_scales_with_attempt(monkeypatch):
    """Backoff duration is retry_delay * attempt (linear)."""
    sleeps: list[float] = []
    monkeypatch.setattr(llm_retry.time, "sleep", sleeps.append)
    llm = _make_llm([{"data": "bad"}, {"data": "bad"}, {"data": "ok"}])

    def parser(response):
        if response["data"] == "bad":
            raise ValueError("nope")
        return response["data"]

    llm_retry.call_with_retry(
        llm,
        [{"role": "user", "content": "hi"}],
        _stub_prompt(),
        parser=parser,
        stage="test_stage",
        context="test:backoff",
        max_retries=3,
        retry_delay=1.5,
    )

    assert sleeps == [1.5, 3.0]


def test_call_with_retry_skips_validator_when_none(monkeypatch):
    """validator=None is accepted and simply bypassed."""
    monkeypatch.setattr(llm_retry.time, "sleep", lambda _s: None)
    llm = _make_llm([{"data": "ok"}])

    result = llm_retry.call_with_retry(
        llm,
        [{"role": "user", "content": "hi"}],
        _stub_prompt(),
        parser=lambda response: response["data"],
        stage="test_stage",
        context="test:no_validator",
        max_retries=1,
        retry_delay=0.0,
        validator=None,
    )

    assert result == "ok"
