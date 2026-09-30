"""Web UI backend: upload the inputs, run o2a_gen, hand back the workflow YAMLs.

    python -m o2a_gen.webui              # http://127.0.0.1:8765

Each run gets a folder under ./runs/<id>/ holding the uploaded inputs, the compiled
skill tree, the generated YAMLs and the log. A run is the two CLI steps in a child
process (compile-catalog, then generate), so the UI server never blocks and the
log is exactly what the CLI prints. API keys are read from the environment only.
"""

from __future__ import annotations

import io
import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
import zipfile
from pathlib import Path

import yaml
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse

HERE = Path(__file__).resolve().parent
PKG_ROOT = HERE.parent.parent            # directory that contains the o2a_gen package
RUNS = Path(os.environ.get("O2A_RUNS_DIR", PKG_ROOT / "runs")).resolve()
BASE_CONFIG = Path(os.environ.get("O2A_GEN_CONFIG", HERE.parent / "gen_config.anthropic.yaml"))

# Stages shown in the UI, matched against the lines the CLI prints.
STAGES = [
    ("compile", "Compile inputs into skills", None),
    ("procedure", "Compile procedure into skills", "[1/6]"),
    ("extract", "Extract steps & choose agent classes", "[2/6]"),
    ("plan", "Plan the workflow", "[3/6]"),
    ("ground", "Write each agent from syntax + tools + metadata", "[4/6]"),
    ("write", "Save YAMLs", "[5/6]"),
    ("validate", "Check", "[6/6]"),
]

app = FastAPI(title="o2a_gen")
_jobs: dict[str, dict] = {}
_lock = threading.Lock()


def _safe_name(name: str) -> str:
    base = Path(name or "file").name
    return re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("._") or "file"


def _save(files: list[UploadFile], folder: Path) -> list[str]:
    folder.mkdir(parents=True, exist_ok=True)
    names = []
    for f in files or []:
        if not f or not f.filename:
            continue
        name = _safe_name(f.filename)
        (folder / name).write_bytes(f.file.read())
        names.append(name)
    return names


def _job_view(job: dict, log_tail: int = 400) -> dict:
    d = {k: job[k] for k in ("id", "status", "stage", "created", "finished", "error", "inputs",
                             "options")}
    log = job["dir"] / "log.txt"
    lines = log.read_text(encoding="utf-8", errors="replace").splitlines() if log.exists() else []
    d["log"] = lines[-log_tail:]
    d["stages"] = [{"id": s, "label": label} for s, label, _ in STAGES]
    out = job["dir"] / "out" / "workflow"
    d["files"] = sorted(p.name for p in out.glob("*.yaml")) if out.exists() else []
    d["agents"] = []
    for name in d["files"]:
        try:
            y = yaml.safe_load((out / name).read_text(encoding="utf-8")) or {}
        except yaml.YAMLError:
            y = {}
        subs = [s.get("name") if isinstance(s, dict) else s for s in y.get("sub_agents") or []]
        d["agents"].append({"file": name, "name": y.get("name") or name[:-5],
                            "agent_class": y.get("agent_class", ""), "children": subs,
                            "description": y.get("description", "")})
    report = job["dir"] / "out" / "workflow_report.json"
    d["report"] = json.loads(report.read_text()) if report.exists() else None
    return d


def _run(job: dict) -> None:
    jd: Path = job["dir"]
    log = (jd / "log.txt").open("a", encoding="utf-8")
    py = [sys.executable, "-u", "-m", "o2a_gen"]
    cmds = [
        ("compile", py + ["compile-catalog", "--config", str(jd / "config.yaml"),
                          "--src", str(jd / "catalog_src"), "--out", str(jd / "catalog_build")]),
        ("procedure", py + ["generate", "--config", str(jd / "config.yaml"),
                            "--procedure", str(jd / "procedure" / job["inputs"]["procedure"][0]),
                            "--catalog", str(jd / "catalog_build"),
                            "--out", str(jd / "out" / "workflow"), "--overwrite", "--workers", "6"]
         + (["--compare-with", str(jd / "compare")] if job["inputs"].get("compare") else [])),
    ]
    try:
        for stage, cmd in cmds:
            job["stage"] = stage
            log.write(f"$ o2a_gen {cmd[4]}\n")
            log.flush()
            proc = subprocess.Popen(cmd, cwd=str(PKG_ROOT), stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, bufsize=1,
                                    env={**os.environ, "PYTHONUNBUFFERED": "1"})
            job["pid"] = proc.pid
            for line in proc.stdout:
                log.write(line)
                log.flush()
                for sid, _, marker in STAGES:
                    if marker and line.strip().startswith(marker):
                        job["stage"] = sid
            code = proc.wait()
            if job.get("cancelled"):
                raise RuntimeError("stopped by the user")
            # generate exits 1 when the checks found errors; the YAMLs are still written.
            done = (jd / "out" / "workflow_report.json").exists()
            if code != 0 and not (stage == "procedure" and done):
                raise RuntimeError(f"{cmd[4]} exited with code {code}; see the log")
        job["status"], job["stage"] = "done", "done"
    except Exception as e:  # noqa: BLE001 - reported to the UI
        job["status"], job["error"] = "failed", str(e)
        log.write(f"\nFAILED: {e}\n")
    finally:
        job["finished"] = time.time()
        log.close()
        (jd / "job.json").write_text(json.dumps(
            {k: v for k, v in job.items() if k != "dir"}, indent=2, default=str))


def _load_old_jobs() -> None:
    if not RUNS.exists():
        return
    for jf in RUNS.glob("*/job.json"):
        try:
            job = json.loads(jf.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if job.get("status") == "running":
            job["status"], job["error"] = "failed", "server restarted during the run"
        job["dir"] = jf.parent
        _jobs[job["id"]] = job


_load_old_jobs()


@app.get("/")
def index():
    return FileResponse(HERE / "static" / "index.html")


@app.get("/api/status")
def status():
    cfg = yaml.safe_load(BASE_CONFIG.read_text())
    return {
        "anthropic_key": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "deepinfra_key": bool(os.environ.get("DEEPINFRA_API_KEY")),
        "llm": sorted(set((cfg.get("models") or {}).values()) - {""}),
        "embedding": (cfg.get("embedding") or {}).get("model"),
        "config": BASE_CONFIG.name,
    }


@app.post("/api/jobs")
def create_job(
    procedure: UploadFile = File(...),
    syntax: list[UploadFile] = File(...),
    tools: list[UploadFile] = File(...),
    metadata: list[UploadFile] = File(...),
    reference: list[UploadFile] = File(default=[]),
    compare: list[UploadFile] = File(default=[]),
    prefix: str = Form(""),
    pipeline_inputs: str = Form(""),
    llm_agent_model: str = Form(""),
):
    missing = [k for k, v in (("ANTHROPIC_API_KEY", os.environ.get("ANTHROPIC_API_KEY")),
                              ("DEEPINFRA_API_KEY", os.environ.get("DEEPINFRA_API_KEY"))) if not v]
    if missing:
        raise HTTPException(400, f"set {', '.join(missing)} in the terminal that runs the server")
    jid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    jd = RUNS / jid
    inputs = {
        "procedure": _save([procedure], jd / "procedure"),
        "syntax": _save(syntax, jd / "catalog_src" / "schema"),
        "tools": _save(tools, jd / "catalog_src" / "tools"),
        "metadata": _save(metadata, jd / "catalog_src" / "metadata"),
        "reference": _save(reference, jd / "catalog_src" / "reference"),
        "compare": _save([f for f in compare if f.filename.lower().endswith((".yaml", ".yml"))],
                         jd / "compare"),
    }
    for need in ("procedure", "syntax", "tools", "metadata"):
        if not inputs[need]:
            raise HTTPException(400, f"{need} file is required")

    cfg = yaml.safe_load(BASE_CONFIG.read_text())
    gen = cfg.setdefault("generation", {})
    prefix = re.sub(r"[^a-z0-9_]+", "_", prefix.strip().lower()).strip("_")
    gen["prefix"] = prefix
    gen["pipeline_inputs"] = [k.strip() for k in re.split(r"[,\s]+", pipeline_inputs) if k.strip()]
    cfg["models"]["llm_agent"] = llm_agent_model.strip()
    (jd / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    job = {"id": jid, "dir": jd, "status": "running", "stage": "compile", "created": time.time(),
           "finished": None, "error": None, "inputs": inputs,
           "options": {"prefix": prefix, "pipeline_inputs": gen["pipeline_inputs"],
                       "llm_agent_model": cfg["models"]["llm_agent"]}}
    with _lock:
        _jobs[jid] = job
    threading.Thread(target=_run, args=(job,), daemon=True).start()
    return {"id": jid}


def _job(jid: str) -> dict:
    job = _jobs.get(jid)
    if not job:
        raise HTTPException(404, "no such run")
    return job


@app.get("/api/jobs")
def list_jobs():
    jobs = sorted(_jobs.values(), key=lambda j: j["created"], reverse=True)[:20]
    return [{"id": j["id"], "status": j["status"], "created": j["created"],
             "procedure": (j["inputs"].get("procedure") or [""])[0]} for j in jobs]


@app.get("/api/jobs/{jid}")
def get_job(jid: str):
    return JSONResponse(_job_view(_job(jid)))


@app.post("/api/jobs/{jid}/cancel")
def cancel(jid: str):
    job = _job(jid)
    if job["status"] == "running" and job.get("pid"):
        job["cancelled"] = True
        try:
            os.kill(job["pid"], 15)
        except ProcessLookupError:
            pass
    return {"ok": True}


@app.get("/api/jobs/{jid}/files/{name}")
def get_file(jid: str, name: str):
    p = _job(jid)["dir"] / "out" / "workflow" / _safe_name(name)
    if not p.is_file():
        raise HTTPException(404, "no such file")
    return PlainTextResponse(p.read_text(encoding="utf-8"))


@app.get("/api/jobs/{jid}/download")
def download(jid: str):
    job = _job(jid)
    out = job["dir"] / "out"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted((out / "workflow").glob("*.yaml")):
            z.write(p, f"workflow/{p.name}")
        for extra in ("workflow_report.json", "workflow_steps.json"):
            if (out / extra).exists():
                z.write(out / extra, extra)
    buf.seek(0)
    name = f"o2a_workflow_{job['options'].get('prefix') or jid}.zip"
    return StreamingResponse(buf, media_type="application/zip",
                             headers={"Content-Disposition": f'attachment; filename="{name}"'})
