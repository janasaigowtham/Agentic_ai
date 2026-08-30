# System Design for Agentic AI

A from-scratch-to-production course on designing agentic AI systems: software
where an LLM doesn't just answer a prompt but perceives, reasons, calls tools,
maintains state, and takes multi-step action toward a goal, wired together
the way you'd design any other distributed system — with contracts, failure
modes, observability, and evaluation.

The course is organized as ten short modules, each building on the last, plus
one working, runnable example (`example/`) that gets extended chapter by
chapter conceptually and is fully implemented by Module 5.

## Who this is for

Engineers who can design a normal backend system (services, queues, state,
observability) and want the delta for when one of those "services" is an LLM
agent instead of deterministic code. No prior agent-framework experience
assumed.

## How to use it

Read modules 1 → 10 in order. Each one ends with a short exercise. From
Module 5 onward, sections point at concrete files in `example/` — open them
side by side. Run the example anytime:

```bash
cd course/example
pip install -r requirements.txt
python run_pipeline.py     # runs 4 end-to-end scenarios, prints traces + scores
pytest                      # the same behavior, asserted
```

No API key is required — the example ships with a deterministic mock model
so it's fully reproducible offline. Module 2 shows exactly where a real model
call would plug in.

## Modules

1. [Foundations](01-foundations.md) — what makes a system "agentic," the
   perceive → reason → act loop, why it's a system-design problem and not
   just a prompting problem.
2. [Agent Architectures](02-agent-architectures.md) — ReAct, Plan-and-Execute,
   Reflexion; single-agent control loops and where they break down.
3. [Tools & Function Calling](03-tools-and-function-calling.md) — designing
   tool contracts, side effects, retries, permissions.
4. [Memory & State](04-memory-and-state.md) — the session/state pattern,
   short- vs long-term memory, context budgets, resumability.
5. [Multi-Agent Orchestration](05-multi-agent-orchestration.md) — why and how
   to decompose one agent into a pipeline; sequential, router, parallel,
   hierarchical topologies. **The working example is introduced here.**
6. [Planning & Reasoning](06-planning-and-reasoning.md) — plan generation,
   re-planning, task graphs, handling ambiguity, cost/quality trade-offs.
7. [Reliability, Evaluation & Observability](07-reliability-evaluation-observability.md) —
   tracing agent steps, structural/routing/output checks, LLM-as-judge,
   regression testing for non-deterministic systems.
8. [Safety & Guardrails](08-safety-and-guardrails.md) — prompt injection,
   least-privilege tools, human-in-the-loop gates, PII handling, runaway-loop
   protection.
9. [Production System Design](09-production-system-design.md) — end-to-end
   architecture, scaling, deployment topologies for long-running/gated flows,
   cost control, SLOs, a review checklist.
10. [Capstone Case Study](10-capstone-case-study.md) — the full working
    example walked end to end, mapped back to every prior module, plus
    extension exercises.

## The working example at a glance

A small customer-support triage pipeline: it looks up an order, classifies
intent with an LLM, routes to a refund / technical / general specialist,
pauses at a human-approval gate for large refunds, drafts a reply, and is
scored by a small evaluator afterward.

```
support_triage_orchestrator
├─ lookup_order_step        (tool)         → order
├─ classify_intent          (LLM)          → intent
└─ intent_router            (router, by intent)
   ├─ refund_flow      (sequential)
   │  ├─ calc_refund        (transform)    → refund_amount
   │  ├─ refund_gate        (human gate)   → refund_approved
   │  └─ draft_refund_reply (LLM)          → final_reply
   ├─ technical_flow   (sequential) → draft_technical_reply (LLM) → final_reply
   └─ general_flow     (sequential) → draft_general_reply   (LLM) → final_reply
```

Everything about this pipeline's *shape* lives in `example/pipeline.yaml`,
not in code — the same declarative-agent pattern used by the production
pipelines this repo's `O2A_EVALUATOR_SPEC.md` was written to evaluate. That's
a deliberate choice you'll see argued for in Module 5.
