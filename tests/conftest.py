from pathlib import Path

import pytest

from o2a_eval.core.graph import load_pipeline
from o2a_eval.core.trace import build_run, load_jsonl

FIXTURES_DIR = Path(__file__).parent / "fixtures"
AGENT_DIR = FIXTURES_DIR / "agents"
PIPELINE_NAME = "pmi_ddn_pipeline"
TRACE_PATH = FIXTURES_DIR / "trace.jsonl"


@pytest.fixture(scope="session")
def graph():
    return load_pipeline(AGENT_DIR, PIPELINE_NAME)


@pytest.fixture(scope="session")
def run():
    if not TRACE_PATH.exists():
        from tests.fixtures.make_trace import build_spans
        import json

        with TRACE_PATH.open("w") as f:
            for span in build_spans():
                f.write(json.dumps(span) + "\n")
    records = load_jsonl(TRACE_PATH)
    return build_run(records)
