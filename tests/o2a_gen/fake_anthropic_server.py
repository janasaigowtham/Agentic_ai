"""A local stand-in for the Anthropic Messages API that streams like the real one.

``FakeAnthropic(responder)`` serves POST /v1/messages on 127.0.0.1. For each request
the responder receives the parsed body and returns one of:

  "text"                              -> streamed answer, stop_reason end_turn
  {"text": ..., "stop_reason": ...}   -> e.g. "refusal" or "max_tokens"
  {"status": 529, "body": "..."}      -> an HTTP error
  {"stream_error": "overloaded_error"} -> an error event in the middle of the stream
  {"text": ..., "slow": 0.2}          -> sends pings `slow` seconds apart before answering

Every request body is kept in ``.bodies``. No network access, no API key needed.
"""

from __future__ import annotations

import http.server
import json
import threading
import time


def _sse(event: str, data: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()


class FakeAnthropic:
    def __init__(self, responder):
        self.responder = responder
        self.bodies: list[dict] = []
        self._lock = threading.Lock()
        outer = self

        class H(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                with outer._lock:
                    outer.bodies.append(body)
                r = outer.responder(body)
                if isinstance(r, str):
                    r = {"text": r}
                if "status" in r:
                    payload = json.dumps({"type": "error", "error": {
                        "type": "api_error", "message": r.get("body", "")}}).encode()
                    self.send_response(r["status"])
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Connection", "close")
                self.end_headers()
                w = self.wfile
                w.write(_sse("message_start", {"type": "message_start", "message": {
                    "model": body["model"], "usage": {"input_tokens": 100,
                                                      "cache_read_input_tokens": 50,
                                                      "output_tokens": 1}}}))
                w.write(_sse("content_block_start", {"type": "content_block_start", "index": 0,
                                                     "content_block": {"type": "thinking",
                                                                       "thinking": ""}}))
                w.write(_sse("content_block_delta", {"type": "content_block_delta", "index": 0,
                                                     "delta": {"type": "thinking_delta",
                                                               "thinking": "not part of the answer"}}))
                w.write(_sse("content_block_stop", {"type": "content_block_stop", "index": 0}))
                for _ in range(int(r.get("pings", 3))):
                    time.sleep(float(r.get("slow", 0)))
                    w.write(_sse("ping", {"type": "ping"}))
                    w.flush()
                if r.get("stream_error"):
                    w.write(_sse("error", {"type": "error", "error": {
                        "type": r["stream_error"], "message": "try again"}}))
                    w.flush()
                    return
                text = r.get("text", "")
                w.write(_sse("content_block_start", {"type": "content_block_start", "index": 1,
                                                     "content_block": {"type": "text", "text": ""}}))
                for i in range(0, len(text), 200):
                    w.write(_sse("content_block_delta", {"type": "content_block_delta", "index": 1,
                                                         "delta": {"type": "text_delta",
                                                                   "text": text[i:i + 200]}}))
                w.write(_sse("content_block_stop", {"type": "content_block_stop", "index": 1}))
                w.write(_sse("message_delta", {"type": "message_delta", "delta": {
                    "stop_reason": r.get("stop_reason", "end_turn")},
                    "usage": {"output_tokens": max(1, len(text) // 4)}}))
                w.write(_sse("message_stop", {"type": "message_stop"}))
                w.flush()

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
