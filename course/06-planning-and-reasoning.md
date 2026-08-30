# Module 6 — Planning & Reasoning

Even inside a fully declared pipeline (Module 5), individual nodes still have
to make judgment calls: which branch does this message belong to, how should
this be worded, what's missing from this request. This module is about
designing *that* layer well — and about the harder case where the plan
itself, not just a node's output, has to be figured out at runtime.

## Levels of planning

- **No planning** — a single reasoning step decides directly (this course's
  `classify_intent`: one call, one categorical answer). Appropriate when the
  decision genuinely is one step.
- **Single-shot plan** — generate an ordered list of steps once, then execute
  it (Module 2's Plan-and-Execute). Appropriate when the step sequence is
  usually right but not fixed enough to hard-code as a graph.
- **Iterative re-planning** — after each step, reconsider whether the
  remaining plan still makes sense; revise if a step's result changes the
  picture (e.g., a lookup reveals the order doesn't exist — no refund
  amount to compute, replan to "explain and offer alternatives"). Necessary
  once actions can invalidate assumptions the plan was built on.
- **Task graphs / DAGs** — when steps have real dependencies but the order
  among independent branches doesn't matter, plan a graph instead of a list,
  and execute whatever's unblocked (this is the parallel topology from
  Module 5, generated rather than declared).

Prefer the cheapest level that's honest about the problem. A single
classification is not a reason to build a general planner; genuinely
open-ended troubleshooting is not a reason to force a fixed graph. Module 5's
"declare vs. decide at runtime" fork is the same decision restated for
planning specifically.

## Handling ambiguity

Real input is messier than your categories. Three legitimate strategies,
not mutually exclusive:

1. **Ask a clarifying question** instead of guessing, when the cost of
   guessing wrong is high and a human is present to answer. Adds a
   round-trip; worth it when a wrong guess routes to the wrong specialist
   entirely.
2. **Pick the safer branch and let a later step correct it.** This course's
   mock classifier resolves "please refund this, it's not working" (which
   plausibly matches both `refund` and `technical`) in favor of `refund` —
   a deliberate policy choice (a missed refund complaint is worse to lose
   than a slightly-imperfect first classification), not an accident. Naming
   the policy explicitly is the point; leaving the resolution order
   implicit is how silent misrouting bugs get shipped.
3. **Route to a general/human fallback** when confidence is genuinely low.
   This is why a router's catch-all branch (Module 5) matters beyond just
   "handling unmatched values" — it's also where low-confidence cases should
   land by design, not by accident.

## Chain-of-thought vs. tool-augmented reasoning

Asking a model to "think step by step" in free text improves quality on
reasoning-heavy tasks but produces an unstructured artifact — it's not
something downstream code can branch on. Tool-augmented reasoning (Module 2's
ReAct) produces structured intermediate results (a tool's return value) that
*can* be checked and branched on. When a decision needs to affect control
flow (which this course's `intent_router` does), prefer forcing a structured
output — a specific field, a fixed enum of values — over parsing prose. It's
also directly checkable: Module 7's router checks assume `chosen_target` is
a clean value, not something regex'd out of a paragraph.

## Cost, latency, and "how hard should this agent think"

More reasoning (a bigger model, a longer chain-of-thought, a Reflexion pass,
more ReAct turns) generally buys quality at the cost of latency and money.
Treat this as a per-node budget decision, not a global one:

- Cheap, low-stakes, high-volume nodes (classification, extraction) — small
  model, no extra reasoning passes, tight latency budget.
- Expensive, low-volume, high-stakes nodes (drafting a legal response,
  approving a large financial action) — can afford a bigger model, a
  self-critique pass, or a human gate (Module 8).

This is the same judgment call as choosing an instance size per
microservice based on its actual load and criticality — don't run every node
in the pipeline at the same (usually maximal) reasoning depth by default.

## Exercise

For your Module 5 graph, mark each node's ambiguity-resolution policy
explicitly: does it ask, guess-safe, or fall back? If a node has no stated
policy, that's the same bug as an unstated default branch in a router —
find one before an incident does.
