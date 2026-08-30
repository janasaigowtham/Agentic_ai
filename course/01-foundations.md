# Module 1 — Foundations

## What makes a system "agentic"

A chatbot takes one input and produces one output. An **agentic** system takes
a goal, and then, on its own, decides a sequence of steps — which may include
calling tools, reading and writing state, asking clarifying questions, or
invoking other agents — until the goal is satisfied or it gives up. The model
call is one component in a larger control loop, not the whole system.

Three properties separate an agentic system from a single prompt call:

| Property | Plain LLM call | Agentic system |
|---|---|---|
| Steps | one | many, chosen at runtime |
| Action | text out | tool calls, side effects |
| State | none (or caller-managed) | the system owns and threads state across steps |
| Control flow | caller decides "what next" | the system (or a declared graph) decides |

None of this requires the model to be smarter — it requires *the system
around it* to be designed like any other multi-step, stateful, failure-prone
distributed system. That's the thesis of this course: agentic AI is a system
design discipline first, a prompting discipline second.

## The core loop

Almost every agentic architecture is a variation on one loop:

```
        ┌────────────┐
        │  PERCEIVE  │  read the goal, the environment, prior state
        └─────┬──────┘
              ▼
        ┌────────────┐
        │   REASON   │  decide the next step (an LLM call, usually)
        └─────┬──────┘
              ▼
        ┌────────────┐
        │     ACT    │  call a tool, hand off to another agent, or answer
        └─────┬──────┘
              ▼
        ┌────────────┐
        │   OBSERVE  │  capture the result into state
        └─────┬──────┘
              │  goal met? ── no ──┐
              ▼                    │
           done                    └──► back to REASON
```

Everything in this course is either: (a) how to implement one turn of this
loop well (Modules 2–4), (b) how to compose many such loops into a system
(Module 5–6), or (c) how to keep that system correct, safe, and observable
once it's running for real (Modules 7–9).

## Anatomy of an agent

Every agent, however implemented, has the same five parts:

- **Model** — the reasoning engine (an LLM call).
- **Instruction** — what it's for, expressed as a prompt/role, ideally as
  narrow and specific as a function's docstring.
- **Tools** — the actions it's allowed to take (§ Module 3).
- **Memory / state** — what it can read and must write (§ Module 4).
- **Control** — what decides whether this agent runs, and what runs next
  (§ Module 5). This is the part classic prompting tutorials skip and system
  design has to own.

## Agentic vs. non-agentic: worked examples

- **Not agentic**: "Summarize this document." One call, one output, no state
  carried forward, no decision about what to do next.
- **Agentic**: "Triage this support ticket." Requires looking up an order
  (a tool call), deciding a category (a reasoning step with a real branch),
  possibly waiting for a human, and producing a different downstream action
  depending on that decision. This is the working example for this course —
  see `course/example/pipeline.yaml`.
- **Borderline**: "Answer this question, using search if you need to."
  One decision point (search or not), but genuinely conditional and
  tool-using — agentic in miniature. Module 2 calls this pattern ReAct.

## Why this is a system-design problem

Once an LLM call can trigger a tool call, and a tool call can trigger another
LLM call, you have introduced everything system design already has language
for: a call graph, shared mutable state, partial failure, retries, latency
budgets, and the need for observability into a black box (the model's
"reasoning" is exactly as inspectable as a remote service's internals — which
is to say, not very, unless you instrument it). The rest of this course is
mostly: here is the standard system-design toolkit, here is what changes when
one node in the graph is non-deterministic.

## Exercise

Take a workflow you know well (an on-call runbook, an expense-approval
process, a customer-support escalation). Draw it as perceive → reason → act →
observe loops. Mark which steps are deterministic (would always do the same
thing given the same input) and which require judgment. The judgment steps
are where an LLM agent belongs; the deterministic steps should stay
deterministic code, not be "asked" of a model — a theme that recurs in every
module that follows.
