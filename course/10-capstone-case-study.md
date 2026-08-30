# Module 10 — Capstone Case Study

This module walks the working example end to end, mapping every piece back
to the module that motivated it, then gives you extension exercises to make
the design your own.

Open these three files side by side: `course/example/pipeline.yaml`,
`course/example/agent_framework.py`, `course/example/eval.py`.

## The system, end to end

**The problem** (Module 1): triage a customer support message — figure out
what it's about, take the appropriate action, and reply — without a human
reading every message first.

**The shape** (Module 5): declared, not model-improvised. `pipeline.yaml`
fixes the graph: look up context, classify, route to exactly one specialist,
draft a reply. The only runtime decisions are *which* branch a router takes
and *what* an LLM step outputs — never *whether a step exists*.

**The engine** (Module 2's "declare vs. decide" fork, resolved): six generic
`agent_class` handlers in `agent_framework.py` — `llm_agent`, `tool_agent`,
`transform_agent`, `sequential_agent`, `router_agent`, `gate_agent`, plus a
top-level `orchestrator`. None of them know anything about refunds or
support tickets; the pipeline's actual meaning lives entirely in the YAML.
This is the "generalize by agent_class, not by pipeline name" principle from
`O2A_EVALUATOR_SPEC.md`, at a scale you can read in one sitting.

**State** (Module 4): one `session` dict, threaded through every step,
each step declaring `input_keys` and `output_key`. `lookup_order_step`
writes `order`; `calc_refund` reads it. No step reaches for a key it didn't
declare.

**Tools** (Module 3): `lookup_order` and `calculate_refund` in `tools.py` —
narrow, deterministic, return a plain dict/float, nothing the model could
misuse beyond its one declared purpose.

**Reasoning** (Module 2 and 6): two genuinely judgment-requiring nodes use
an LLM (`classify_intent`, and the three `draft_*_reply` nodes) — everything
else is deterministic code. `llm_client.py`'s `MockLLM` stands in for a real
model with deterministic, keyword-driven logic, specifically so the whole
system is reproducible offline; `AnthropicLLM` in the same file shows exactly
where a real provider call plugs in without touching `agent_framework.py` or
`pipeline.yaml` at all — proof that the engine doesn't care which model (or
whether a model) sits behind the `llm_agent` handler.

**Safety** (Module 8): `refund_gate` is a real gate, not a rubber stamp —
`run_pipeline.py`'s `approve_gate` policy rejects refunds over $1,000
outright, and adds a deliberate review delay for refunds over $100, so the
gate's timing reflects real review rather than an instant, meaningless
pass-through.

**Observability and evaluation** (Module 7): `tracing.py` captures one
`Span` per step; `eval.py` diffs what ran against what was declared and
produces a scorecard with severities — structural (`O-S3`: was the final
output ever produced), routing (`R-R1`: did exactly one branch run),
efficiency (`D1`: did anything run redundantly), and a gate-integrity check
(`G-R1`: did a gate that should have taken real review time actually take
any time at all).

## Run it and read the output

```bash
cd course/example
pip install -r requirements.txt
python run_pipeline.py
```

You'll see four scenarios: a small refund (auto-approved, clean scorecard),
a technical complaint (routes to `technical_flow`, `refund_flow` never
runs), an ambiguous refund-shaped complaint (Module 6's ambiguity policy in
action), and a large refund that the gate outright rejects (Module 8's gate
with teeth — the scorecard correctly reports `FAIL` because no final reply
was produced, which is the *correct* outcome for a rejected action, not a
bug).

Then:

```bash
pytest
```

Read `tests/test_pipeline.py`. Notice the tests don't call a real model or
hit a network — they assert against the mock's deterministic behavior,
exactly Module 7's "pin the non-deterministic parts to test the rest"
principle.

## Extension exercises

Ordered roughly easy → hard; each one exercises a specific module.

1. **Add a fourth intent** (e.g., "billing question") with its own branch
   and its own `draft_*_reply` node. (Module 5 — extend the declared graph,
   touch no engine code.)
2. **Add a new tool**, `check_loyalty_status(customer_id)`, and use its
   result to skip the gate for loyalty-tier customers on refunds up to
   $250. (Module 3 — contract design; Module 8 — think about whether this
   loosens the gate in a way that needs its own review.)
3. **Break the router on purpose** — change one route's `context_key` to a
   session key nothing produces — and confirm a new static check you write
   (mirroring `O2A_EVALUATOR_SPEC.md`'s `R-S1`) catches it *without running
   the pipeline at all*. (Module 7 — static vs. runtime checks.)
4. **Make the gate durable** — instead of `approve_gate` deciding
   synchronously in-process, serialize the session to a JSON file at the
   gate and write a separate `resume_pipeline.py` that loads it back and
   continues. (Module 4 and Module 9 — resumability and the async/gated
   deployment topology.)
5. **Add a judge-based check** — write a `check_reply_quality` in `eval.py`
   that calls `llm.complete()` with a rubric prompt (Module 7's judge
   pattern) and flags a reply that doesn't mention the customer's actual
   complaint. Decide, and justify, when it should be skipped (a structural
   SEV-1 already present, Module 7's skip rule).
6. **Swap in a real model.** Set `AGENT_COURSE_LLM=anthropic` and a real
   `ANTHROPIC_API_KEY`, rerun the scenarios, and see whether classification
   and drafting still behave sensibly on inputs the mock wasn't
   hand-written to handle — then decide what a judge check (exercise 5)
   would need to catch if they didn't.

## What you should be able to say now

Given any agentic system someone describes to you, you should be able to
ask the questions this course was built around: What's declared vs. decided
at runtime? What's the state contract between steps? Which actions need a
gate? How would you know, from a trace alone, that this ran correctly? That
last question is the one most systems in production today still can't
answer — which is exactly the gap `O2A_EVALUATOR_SPEC.md`, in this repo, was
written to close at production scale.
