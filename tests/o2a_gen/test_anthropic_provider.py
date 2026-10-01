"""Anthropic Batch API, per-call effort and caching marks, against local stand-ins."""

import http.server
import json
import threading

import pytest

from o2a_gen.llm import CallableClient, complete_json_many


class FakeBatches:
    """Serves the Message Batches endpoints on 127.0.0.1. ``responder(params)`` returns
    {"text", "stop_reason"} for a succeeded item or {"error": "..."} for an errored one."""

    def __init__(self, responder, polls_until_done=2):
        self.responder = responder
        self.submitted: list[list[dict]] = []
        self.gets = 0
        self.cancelled = False
        outer = self

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, payload, ctype="application/json"):
                data = payload.encode() if isinstance(payload, str) else json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or b"{}")
                if self.path.endswith("/cancel"):
                    outer.cancelled = True
                    return self._send({"id": "b1", "processing_status": "canceling"})
                outer.submitted.append(body["requests"])
                self._send({"id": "b1", "processing_status": "in_progress"})

            def do_GET(self):
                if self.path == "/results/b1":
                    lines = []
                    for item in reversed(outer.submitted[-1]):      # any order
                        r = outer.responder(item["params"])
                        if "error" in r:
                            res = {"type": "errored", "error": {"type": "api_error",
                                                                "message": r["error"]}}
                        else:
                            res = {"type": "succeeded", "message": {
                                "content": [{"type": "thinking", "thinking": ""},
                                            {"type": "text", "text": r["text"]}],
                                "stop_reason": r.get("stop_reason", "end_turn"),
                                "usage": {"input_tokens": 10, "cache_read_input_tokens": 90,
                                          "cache_creation_input_tokens": 0,
                                          "output_tokens": 5}}}
                        lines.append(json.dumps({"custom_id": item["custom_id"], "result": res}))
                    return self._send("\n".join(lines), "application/binary")
                outer.gets += 1
                done = outer.gets >= polls_until_done
                self._send({"id": "b1", "processing_status": "ended" if done else "in_progress",
                            "results_url": f"{outer.url}/results/b1" if done else None})

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()


@pytest.fixture
def api_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("O2A_ANTHROPIC_BATCH_POLL_SECONDS", "0")
    monkeypatch.delenv("O2A_ANTHROPIC_EFFORT", raising=False)
    monkeypatch.setattr("o2a_gen.providers.anthropic_provider._pause", lambda s: None)
    return monkeypatch


def _req(q, **kw):
    return {"model": "claude-opus-5-5", "system": "rules", "max_tokens": 400,
            "messages": [{"role": "user", "content": q}], **kw}


def test_batch_submits_cached_requests_and_maps_results_by_custom_id(api_env):
    from o2a_gen.providers.anthropic_provider import AnthropicError, batch
    fake = FakeBatches(lambda p: {"error": "overloaded"} if p["messages"][0]["content"][0]["text"] == "bad"
                       else {"text": p["messages"][0]["content"][0]["text"].upper()})
    api_env.setenv("ANTHROPIC_BASE_URL", fake.url)
    try:
        out = batch([_req("a", effort="low"), _req("bad"), _req("c")])
    finally:
        fake.close()
    assert out[0]["text"] == "A" and out[2]["text"] == "C"
    assert out[0]["cache_read_tokens"] == 90 and out[0]["input_tokens"] == 100
    assert isinstance(out[1], AnthropicError)
    sent = fake.submitted[0]
    assert [r["custom_id"] for r in sent] == ["r0", "r1", "r2"]
    p = sent[0]["params"]
    assert p["system"][0]["cache_control"] == {"type": "ephemeral"}       # shared prefix cached
    assert "cache_control" not in p["messages"][0]["content"][0]          # unique text is not
    assert p["output_config"] == {"effort": "low"} and "output_config" not in sent[1]["params"]
    assert p["max_tokens"] >= 16000 and "stream" not in p and "temperature" not in p


def test_batch_that_runs_too_long_is_cancelled(api_env):
    from o2a_gen.providers.anthropic_provider import AnthropicError, batch
    fake = FakeBatches(lambda p: {"text": "x"}, polls_until_done=10**6)
    api_env.setenv("ANTHROPIC_BASE_URL", fake.url)
    api_env.setenv("O2A_ANTHROPIC_BATCH_MAX_WAIT_MINUTES", "0")
    try:
        with pytest.raises(AnthropicError, match="cancelled"):
            batch([_req("a")])
    finally:
        fake.close()
    assert fake.cancelled


def test_chat_effort_wins_over_env_and_one_shot_prompt_is_not_cached(api_env):
    from o2a_gen.providers.anthropic_provider import chat

    from .fake_anthropic_server import FakeAnthropic
    fake = FakeAnthropic(lambda body: "ok")
    api_env.setenv("ANTHROPIC_BASE_URL", fake.url)
    api_env.setenv("O2A_ANTHROPIC_EFFORT", "high")
    try:
        chat(**_req("q", effort="low"))
        chat(**_req("q"))
    finally:
        fake.close()
    first, second = fake.bodies
    assert first["output_config"] == {"effort": "low"} and second["output_config"] == {"effort": "high"}
    assert "cache_control" not in first["messages"][0]["content"][0]
    assert first["system"][0]["cache_control"] == {"type": "ephemeral"}


def test_callable_client_passes_effort_only_to_chat_functions_that_take_it():
    seen = []

    def with_effort(*, model, system, messages, max_tokens, temperature, effort=None):
        seen.append(effort)
        return "ok"

    def without_effort(*, model, system, messages, max_tokens, temperature):
        return "ok"

    CallableClient(with_effort).complete(model="m", messages=[], effort="low")
    CallableClient(without_effort).complete(model="m", messages=[], effort="low")   # no TypeError
    assert seen == ["low"]


def test_complete_json_many_batches_then_retries_leftovers_with_chat():
    chat_calls, batches = [], []

    def chat(*, model, system, messages, max_tokens, temperature, effort=None):
        chat_calls.append(messages[-1]["content"])
        return json.dumps({"title": messages[-1]["content"]})

    def batch(requests):
        batches.append(requests)
        out = []
        for r in requests:
            q = r["messages"][0]["content"]
            if q == "broken":
                out.append({"text": "not json"})
            elif q == "declined":
                out.append({"text": "", "stop_reason": "refusal"})
            elif q == "failed":
                out.append(RuntimeError("overloaded"))
            else:
                out.append({"text": json.dumps({"title": q}), "cache_read_tokens": 7})
        return out

    client = CallableClient(chat, batch=batch, batch_min_requests=2)
    jobs = [(q, "sys", q) for q in ("one", "broken", "declined", "failed", "five")]
    got = complete_json_many(client, jobs, model="claude-opus-5-5", effort="low", workers=2)
    assert got == {q: {"title": q} for q in ("one", "broken", "declined", "failed", "five")}
    assert len(batches) == 1 and batches[0][0]["effort"] == "low"
    assert sorted(chat_calls) == ["broken", "declined", "failed"]       # only the leftovers
    u = client.usage.as_dict()
    assert u["batch_calls"] == 4 and u["cache_read_tokens"] == 14


def test_complete_json_many_small_groups_skip_the_batch():
    batches = []
    client = CallableClient(lambda **k: '{"summary": "x"}',
                            batch=lambda reqs: batches.append(reqs) or [], batch_min_requests=5)
    got = complete_json_many(client, [("a", "s", "p"), ("b", "s", "p")], model="m")
    assert set(got) == {"a", "b"} and not batches


def test_skill_tree_cards_go_through_one_batch_and_are_cached(tmp_path):
    from o2a_gen.catalog import CatalogDoc
    from o2a_gen.skilltree import make_cards
    batches = []

    def batch(requests):
        batches.append(requests)
        return [{"text": json.dumps({"title": "T", "one_line": "L", "keywords": ["k"]})}
                for _ in requests]

    client = CallableClient(lambda **k: "{}", batch=batch, batch_min_requests=2)
    docs = [CatalogDoc(id=f"tools/t{i}", kind="tool", name=f"t{i}", source="x", text=f"tool {i}")
            for i in range(4)]
    cards = make_cards(client, docs, "claude-opus-5-5", tmp_path / "cards.json", workers=2,
                       effort="low")
    assert all(v["title"] == "T" for v in cards.values())
    assert len(batches) == 1 and len(batches[0]) == 4 and batches[0][0]["effort"] == "low"
    make_cards(client, docs, "claude-opus-5-5", tmp_path / "cards.json", workers=2)
    assert len(batches) == 1                                             # unchanged: no calls


def test_anthropic_config_has_batch_off_and_low_compile_effort():
    from pathlib import Path

    from o2a_gen.config import load_config
    cfg = load_config(Path(__file__).resolve().parents[2] / "o2a_gen" / "gen_config.anthropic.yaml")
    assert "batch" not in cfg.llm              # off: batches can queue for an hour or more
    assert cfg.catalog["effort"] == "low"
