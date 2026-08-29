import json

import pytest

pytest.importorskip("opentelemetry.sdk.trace")

from o2a_eval.adapters.base import FrameworkAdapter
from o2a_eval.capture import attach as attach_mod


class FakeAgent:
    name = "fake_agent"
    output_key = "fake_output"
    input_keys = ["fake_input"]

    def __init__(self):
        self.session = {"fake_input": "hello"}

    def run(self):
        self.session[self.output_key] = "result-value"
        return "ok"


class FailingAgent(FakeAgent):
    def run(self):
        raise RuntimeError("agent blew up")


@pytest.fixture(autouse=True)
def _reset_attach_state():
    attach_mod._STATE.clear()
    attach_mod._STATE["attached"] = False
    yield
    attach_mod._STATE.clear()
    attach_mod._STATE["attached"] = False


def _adapter(cls) -> FrameworkAdapter:
    return FrameworkAdapter(
        base_class_path=f"{cls.__module__}.{cls.__qualname__}",
        get_session=lambda args, kwargs: getattr(args[0], "session", {}) if args else {},
    )


def test_attach_captures_spans_and_is_idempotent(monkeypatch):
    monkeypatch.setattr(FakeAgent, "run", FakeAgent.run, raising=False)
    adapter = _adapter(FakeAgent)
    store = attach_mod.attach(adapter, {"agent_dir": "x"}, evaluate_on_exit=False)

    agent = FakeAgent()
    result = agent.run()
    assert result == "ok"

    from opentelemetry import trace

    trace.get_tracer_provider().force_flush()
    spans = store.read_all()
    assert len(spans) >= 1
    span = spans[-1]
    assert span["attributes"]["o2a.agent.name"] == "fake_agent"
    assert span["attributes"]["o2a.session.keys_before"] == '["fake_input"]'

    # second attach() call must be a no-op that returns the same store
    store2 = attach_mod.attach(adapter, {"agent_dir": "x"}, evaluate_on_exit=False)
    assert store2 is store


def test_capture_failure_does_not_break_original_call(monkeypatch):
    adapter = _adapter(FakeAgent)

    def _boom(*a, **k):
        raise ValueError("capture broke")

    monkeypatch.setattr(attach_mod, "set_pre_attributes", _boom)
    attach_mod.attach(adapter, {"agent_dir": "x"}, evaluate_on_exit=False)

    agent = FakeAgent()
    assert agent.run() == "ok"  # original call still succeeds despite capture failure


def test_original_exception_propagates_with_error_status():
    adapter = _adapter(FailingAgent)
    store = attach_mod.attach(adapter, {"agent_dir": "x"}, evaluate_on_exit=False)

    agent = FailingAgent()
    with pytest.raises(RuntimeError):
        agent.run()

    from opentelemetry import trace

    trace.get_tracer_provider().force_flush()
    spans = store.read_all()
    assert any(s["status"] == "ERROR" for s in spans)
