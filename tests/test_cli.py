import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
AGENT_DIR = REPO_ROOT / "tests" / "fixtures" / "agents"
TRACE_PATH = REPO_ROOT / "tests" / "fixtures" / "trace.jsonl"


def _run(*args):
    return subprocess.run(
        [sys.executable, "-m", "o2a_eval.cli", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )


def test_graph_no_missing_nodes():
    result = _run("graph", "--agent-dir", str(AGENT_DIR), "--pipeline", "pmi_ddn_pipeline")
    assert result.returncode == 0
    assert "<missing>" not in result.stdout
    assert "pipeline_hash" in result.stdout


def test_lint_only_static_flaws():
    result = _run("lint", "--agent-dir", str(AGENT_DIR), "--pipeline", "pmi_ddn_pipeline")
    assert result.returncode == 0
    assert "R-S3" in result.stdout or "N2" in result.stdout
    for runtime_flaw in ("D1", "D9", "N3", "O-R3", "G-R1", "R-R1"):
        assert runtime_flaw not in result.stdout


def test_score_catches_seeded_flaws_and_no_false_positives():
    result = _run(
        "score", str(TRACE_PATH),
        "--agent-dir", str(AGENT_DIR),
        "--pipeline", "pmi_ddn_pipeline",
        "--no-judge",
        "--latency-budget-s", "60",
    )
    assert result.returncode == 1
    for seeded in ("D1", "N3", "N2", "R-S3", "G-R1", "O-R3"):
        assert seeded in result.stdout, seeded
    assert "sequential_reorder" not in result.stdout
    assert "dead_output" not in result.stdout or "pmi_ddn_investor_score" in result.stdout
    assert "router_not_exhaustive" not in result.stdout or "pmi_ddn_investor_review_router" not in result.stdout
    assert "structural_drift" not in result.stdout


def test_score_json_no_pii():
    result = _run(
        "score", str(TRACE_PATH),
        "--agent-dir", str(AGENT_DIR),
        "--pipeline", "pmi_ddn_pipeline",
        "--no-judge",
        "--latency-budget-s", "60",
        "--json",
    )
    payload = json.loads(result.stdout)
    blob = json.dumps(payload)
    for pii in ["1234567890", "Jane Doe", "ABSOLUTE RULE", "SELECT"]:
        assert pii not in blob


def test_checks_lists_all_registered():
    result = _run("checks")
    assert result.returncode == 0
    assert "checks registered" in result.stdout


def test_score_fail_on_info_still_fails_on_sev3(tmp_path):
    result = _run(
        "score", str(TRACE_PATH),
        "--agent-dir", str(AGENT_DIR),
        "--pipeline", "pmi_ddn_pipeline",
        "--no-judge",
        "--fail-on", "INFO",
    )
    assert result.returncode == 1


def test_score_output_dir_writes_files(tmp_path):
    out_dir = tmp_path / "out"
    result = _run(
        "score", str(TRACE_PATH),
        "--agent-dir", str(AGENT_DIR),
        "--pipeline", "pmi_ddn_pipeline",
        "--no-judge",
        "--latency-budget-s", "60",
        "--output-dir", str(out_dir),
    )
    assert (out_dir / "scorecard.json").exists()
    assert (out_dir / "trace.summary.json").exists()
    scorecard = json.loads((out_dir / "scorecard.json").read_text())
    assert scorecard["status"] == "FAIL"


def test_lint_pipeline_defaults_and_fail_on():
    result = _run(
        "lint", "--agent-dir", str(AGENT_DIR), "--pipeline", "pmi_ddn_pipeline",
        "--fail-on", "INFO",
    )
    # only SEV-2/SEV-3 static flaws exist on this fixture, no SEV-1
    assert result.returncode == 1
