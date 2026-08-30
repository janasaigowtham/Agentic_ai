# Working example: a support-triage agent pipeline

The runnable companion to the course in `course/`. A small multi-agent
system that looks up an order, classifies a support message, routes to a
refund / technical / general specialist, pauses at a human-approval gate for
large refunds, drafts a reply, and is scored afterward by a toy evaluator.
See `course/10-capstone-case-study.md` for the full walkthrough.

## Run it

```bash
pip install -r requirements.txt
python run_pipeline.py
```

Runs four scenarios end to end and prints, for each: the execution trace
(which agents ran, in what order, how long they took), a scorecard (status,
overall score, any flaws found), and the final reply (when one was
produced).

## Test it

```bash
pytest
```

No network calls, no API key required — the pipeline runs against a
deterministic mock model by default (`llm_client.py`'s `MockLLM`).

## Files

| File | What it is | Course module |
|---|---|---|
| `pipeline.yaml` | the declarative pipeline definition — **edit this to change the system's shape**, never the engine | 5 |
| `agent_framework.py` | the generic execution engine: reads the `Node` tree, runs it, emits spans | 2, 5 |
| `tracing.py` | in-memory span capture | 7 |
| `eval.py` | a toy evaluator: structural / routing / efficiency / gate-integrity checks | 7 |
| `llm_client.py` | `MockLLM` (default, offline, deterministic) + `AnthropicLLM` (opt-in, real API) | 2, 6 |
| `tools.py` | example deterministic tools (order lookup, refund calculation) | 3 |
| `run_pipeline.py` | CLI: builds the engine, runs four scenarios, prints trace + scorecard | 9 |
| `conftest.py` | adds this directory to `sys.path` for the test suite | — |
| `tests/test_pipeline.py` | pytest suite: routing, gating, and evaluator behavior | 7 |

## Using a real model

By default nothing here calls out to a real LLM provider. To use one:

```bash
export AGENT_COURSE_LLM=anthropic
export ANTHROPIC_API_KEY=sk-...
pip install anthropic
python run_pipeline.py
```

`agent_framework.py` and `pipeline.yaml` do not change — only
`llm_client.build_llm()`'s return value does. That's the point: the engine
never knows or cares what's behind the `llm_agent` handler.
