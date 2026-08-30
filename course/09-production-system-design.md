# Module 9 — Production System Design

Everything so far has been about the pipeline itself. This module zooms out
to the system it runs inside: what surrounds an agentic pipeline in
production, and what changes about ordinary backend system design when one
of your "services" is a model call.

## Reference architecture

```
 client
   │
   ▼
 API gateway  ── auth, rate limiting, request validation
   │
   ▼
 orchestrator service  ── loads the declared pipeline (Module 5),
   │                       owns session lifecycle, resumability
   │
   ├──► deterministic workers  (tools, DB calls, transforms — Module 3/4)
   │
   ├──► model provider(s)  ── LLM calls, with routing/fallback across
   │                           providers or model sizes
   │
   ├──► gate/approval store  ── durable queue for paused, awaiting-human
   │                             pipeline runs (Module 8)
   │
   └──► observability pipeline  ── trace export, PII scrubbing,
                                    evaluator, scorecards (Module 7)
```

Nothing here is unique to agentic systems except the model-provider box and
the gate/approval store — everything else is the architecture you'd draw for
any stateful, multi-step backend system. That's deliberate: the goal of this
course is that an agentic system doesn't need a different design vocabulary,
just a few additions to the one you already have.

## Statelessness of workers, state in the store

Keep the orchestrator and workers themselves stateless between requests —
all pipeline state lives in the session store (Module 4), not in a worker
process's memory. This buys you the same thing it always buys: any worker
can pick up any run, workers can scale horizontally, and a crashed worker
loses no state, only the in-flight step (which should be safely retryable —
Module 3's idempotency requirement).

## Deployment topologies for long-running / gated flows

A pipeline with no gates can be a plain synchronous request/response call:
client calls, orchestrator runs the whole graph, response returns. A
pipeline with a gate (Module 8) cannot — it might pause for hours. Two
practical shapes:

- **Async job + webhook/poll.** Client kicks off a run, gets a run ID back
  immediately, and either polls for status or registers a webhook. The
  orchestrator persists session state at the gate and a separate approval
  action (a UI, an API call from a reviewer) resumes it later, loading state
  back from the store.
- **Queue-backed resume.** The paused run sits as a durable message/row;
  approval enqueues a "resume run X" event that a worker picks up. This is
  the same pattern as any long-running workflow system (Module 4's
  resumability requirement is what makes this possible at all) — if you've
  built a saga or a durable workflow engine before, this is the same
  problem.

Don't force a gated pipeline into a synchronous HTTP call with a long
timeout "because it's simpler" — it isn't simpler, it's a system that falls
over the first time a human takes longer than the timeout to review
something.

## Cost and latency budgets

Put a budget on every layer, not just the whole run:

- **Per-node model selection** (Module 6): don't run every node at your most
  expensive model by default.
- **Per-run cost ceiling** (Module 8): abort gracefully past a hard budget,
  don't let one pathological input consume unbounded spend.
- **Latency budget with attribution**: Module 7's observability should be
  able to answer "which step is eating the time" (the top-N-contributors
  evidence in a latency check), not just "the run was slow." Without
  attribution, latency regressions get "fixed" by guessing.
- **Fallback across providers/models**: if your primary model provider is
  degraded, a fallback (a different provider, or a smaller local model for
  low-stakes nodes) keeps the system available at reduced quality rather
  than fully down. Treat this like any other dependency failover — test it,
  don't assume it works the first time you need it.

## SLOs for a system with a non-deterministic component

You can still define SLOs; you just need axes beyond uptime:

- **Availability** — same as any service: is the orchestrator up, are model
  provider calls succeeding.
- **Structural correctness rate** — fraction of runs with zero SEV-1
  structural flaws (Module 7). This is measurable at 100% of runs, cheaply,
  with no model involved in the checking.
- **Output quality rate** — fraction of runs passing judge-based checks
  above threshold, sampled (judges are expensive; sample, don't check every
  run, once you trust the sampling is representative).
- **Time-to-resolution for gated runs** — how long human-approval steps
  actually take; a growing queue here is an incident even though nothing
  "crashed."

## Incident response

When an agentic system misbehaves in production, the trace (Module 7) is
your primary incident artifact — the same role a request trace plays in any
distributed system, plus the rendered prompt and session snapshot for the
step in question (handled per Module 8's PII rules). Build the habit of
pulling a specific run's trace as step one of any investigation, before
speculating about "the model" in the abstract — most incidents turn out to
be a specific node's contract violation (Module 7's structural checks), not
a mysterious model failure.

## A review checklist

Before calling an agentic system production-ready:

- [ ] Every step's contract is declared (`input_keys`/`output_key`,
      Module 4) and checkable without running a model
- [ ] Every router has an explicit catch-all (Module 5)
- [ ] Every tool's blast radius is sized to what's needed; nothing
      consequential fires without a gate (Module 8)
- [ ] Every control loop has a step/cost/time ceiling (Module 8)
- [ ] Structural checks run on every run; semantic (judge) checks are
      budgeted and sampled (Module 7)
- [ ] Traces are captured with an allow-list scrubber before persisting
      (Module 8)
- [ ] Gated flows are resumable from durable state, not in-process memory
      (Module 4, this module)
- [ ] There's a fallback path for model-provider degradation
- [ ] A specific run's trace can be pulled and read by an on-call engineer
      in under a minute

## Exercise

Take the architecture diagram above and mark which boxes your Module 5
pipeline design actually needs (not every system needs a gate/approval
store, for instance). Then identify the one box you're least confident you'd
get right on the first try — that's where to prototype first.
