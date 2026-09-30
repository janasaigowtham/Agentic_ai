"""End-to-end smoke test of the web UI with no real API calls.

Starts a streaming stand-in for the Anthropic API (answered by SyntheticModel) and a
stand-in for DeepInfra embeddings, starts the real UI server pointed at them, uploads
the inputs exactly as the browser does, waits for the run, and checks the result.

    python -m tests.o2a_gen.smoke_ui \\
        --procedure AMBC1.md --tools tools.yaml --metadata metadata.yaml --syntax syntax.md

Without arguments it uses the test fixtures. Exit code 0 means every check passed.
"""

from __future__ import annotations

import argparse
import hashlib
import http.server
import io
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import threading
import time
import zipfile
from pathlib import Path

import httpx
import yaml

from .fake_anthropic_server import FakeAnthropic
from .synthetic_model import SyntheticModel

ROOT = Path(__file__).resolve().parents[2]
FIX = Path(__file__).parent / "fixtures" / "catalog_src"


def fake_deepinfra() -> tuple[http.server.HTTPServer, list]:
    calls: list = []

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append(body)
            rows = []
            for i, t in enumerate(body["input"]):
                v = [0.0] * 4096
                for w in re.findall(r"[a-z0-9_]+", t.lower()):
                    v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 4096] += 1.0
                n = sum(x * x for x in v) ** 0.5 or 1.0
                rows.append({"index": i, "embedding": [x / n for x in v]})
            payload = json.dumps({"data": rows}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, calls


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def run(procedure: Path, tools: list[Path], metadata: list[Path], syntax: list[Path],
        runs_dir: Path, timeout: float = 600) -> list[str]:
    """Returns the list of failed checks (empty = all passed)."""
    model = SyntheticModel()
    anthropic = FakeAnthropic(model)
    deepinfra, embed_calls = fake_deepinfra()
    port = free_port()
    env = {**os.environ, "ANTHROPIC_API_KEY": "smoke-test", "DEEPINFRA_API_KEY": "smoke-test",
           "ANTHROPIC_BASE_URL": anthropic.url,
           "DEEPINFRA_BASE_URL": f"http://127.0.0.1:{deepinfra.server_port}",
           "O2A_RUNS_DIR": str(runs_dir), "NO_PROXY": "127.0.0.1,localhost"}
    server = subprocess.Popen([sys.executable, "-m", "o2a_gen.webui", "--port", str(port)],
                              cwd=str(ROOT), env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True)
    base = f"http://127.0.0.1:{port}"
    fails: list[str] = []

    def check(ok, what):
        print(f"  {'PASS' if ok else 'FAIL'}  {what}")
        if not ok:
            fails.append(what)

    try:
        c = httpx.Client(base_url=base, timeout=30, trust_env=False)
        for _ in range(100):
            try:
                if c.get("/api/status").status_code == 200:
                    break
            except httpx.TransportError:
                time.sleep(0.1)
        page = c.get("/")
        check(page.status_code == 200 and "Generate workflow" in page.text, "UI page loads")
        st = c.get("/api/status").json()
        check(st["anthropic_key"] and st["deepinfra_key"], "server sees both API keys")

        files = [("procedure", (procedure.name, procedure.read_bytes()))]
        files += [("tools", (p.name, p.read_bytes())) for p in tools]
        files += [("metadata", (p.name, p.read_bytes())) for p in metadata]
        files += [("syntax", (p.name, p.read_bytes())) for p in syntax]
        r = c.post("/api/jobs", files=files, data={"prefix": "smoke", "pipeline_inputs": "",
                                                    "llm_agent_model": ""})
        check(r.status_code == 200, f"upload accepted ({r.status_code})")
        jid = r.json()["id"]

        t0, seen_stages = time.time(), []
        while time.time() - t0 < timeout:
            j = c.get(f"/api/jobs/{jid}").json()
            if not seen_stages or seen_stages[-1] != j["stage"]:
                seen_stages.append(j["stage"])
            if j["status"] != "running":
                break
            time.sleep(0.5)
        print(f"  run {jid}: {j['status']} in {time.time() - t0:.0f}s; stages {seen_stages}")
        check(j["status"] == "done", f"run finished ({j['status']}: {j.get('error')})")
        log = "\n".join(j["log"])
        if j["status"] != "done":
            print("\n".join(j["log"][-40:]))
            return fails
        for marker in ("[1/4] Cards", "[2/4] Embedding", "[3/4] Partitioning", "[4/4] Describing",
                       "[1/6]", "[2/6]", "[3/6]", "[4/6]", "[5/6]", "[6/6]"):
            check(marker in log, f"log shows stage {marker}")
        check("answered by claude-sonnet-5" in log, "declined card answered by the fallback model")
        check("Traceback" not in log, "no tracebacks in the log")

        # the YAMLs
        rep = j["report"]
        check(rep is not None and rep["agents"] == len(j["files"]) > 0,
              f"{len(j['files'])} YAML files, one per planned agent")
        classes = set()
        for p in syntax:
            from o2a_gen.catalog import discover_classes
            classes |= discover_classes(p.read_text(encoding="utf-8"))
        agents = {}
        for name in j["files"]:
            text = c.get(f"/api/jobs/{jid}/files/{name}").text
            agents[name[:-5]] = yaml.safe_load(text)
        check(all(a.get("agent_class") in classes for a in agents.values()),
              "every agent_class comes from the syntax document")
        subs = {s["name"] for a in agents.values() for s in a.get("sub_agents") or []}
        check(subs <= set(agents), "every sub_agent has its own YAML")
        roots = set(agents) - subs
        check(len(roots) == 1, f"one root agent ({sorted(roots)})")
        targets = {r.get("target_agent") for a in agents.values() for r in a.get("routes") or []}
        check(targets and targets <= set(agents), "every route targets an existing agent")
        check(all(c_["agent"] for c_ in rep["coverage"]), "every extracted step became an agent")
        if any("Required keys" in p.read_text(encoding="utf-8") for p in syntax):
            empty = sum(1 for g in rep["grounding"] if g["gaps"])
            check(empty > 0, f"placeholders reported as gaps ({empty} agents)")
        check(any(g["catalog_docs"] for g in rep["grounding"]), "agents written from found skills")
        check(not [g for g in rep["grounding"] if any("kept after" in w for w in g["warnings"])],
              "no agent needed its answer kept unchecked")

        # what reached the model
        bodies = anthropic.bodies
        check(all(b.get("stream") is True and "temperature" not in b for b in bodies),
              f"all {len(bodies)} model calls streamed, no temperature")
        ground = [b for b in bodies if "Agent to write:" in json.dumps(b)]
        check(any("The procedure's own wording for this step" in json.dumps(b) for b in ground),
              "agent writer gets the cited procedure lines verbatim")
        retried = [b for b in ground if len(b["messages"]) == 3]
        check(retried, "a bad answer was sent back with the error and corrected")
        check(model.counts.get("extract", 0) == 2, "broken extraction stream was retried")
        check(len(embed_calls) > 0, f"embeddings requested ({len(embed_calls)} calls)")

        z = zipfile.ZipFile(io.BytesIO(c.get(f"/api/jobs/{jid}/download").content))
        check(sum(n.endswith(".yaml") for n in z.namelist()) == len(j["files"]),
              "zip download holds every YAML")
        print(f"  model calls by type: {model.counts}")
    finally:
        server.terminate()
        server.wait(10)
        anthropic.close()
        deepinfra.shutdown()
    return fails


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--procedure", type=Path,
                    default=Path(__file__).parent / "fixtures" / "procedure_sample.md")
    ap.add_argument("--tools", type=Path, nargs="+", default=sorted((FIX / "tools").glob("*")))
    ap.add_argument("--metadata", type=Path, nargs="+", default=sorted((FIX / "metadata").glob("*")))
    ap.add_argument("--syntax", type=Path, nargs="+", default=sorted((FIX / "schema").glob("*")))
    a = ap.parse_args(argv)
    with tempfile.TemporaryDirectory() as tmp:
        fails = run(a.procedure, a.tools, a.metadata, a.syntax, Path(tmp) / "runs")
    print("\nALL CHECKS PASSED" if not fails else f"\n{len(fails)} CHECK(S) FAILED: {fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
