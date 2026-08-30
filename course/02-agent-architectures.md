# Module 2 — Agent Architectures

This module is about the control loop *inside* a single agent: given a goal,
how does one agent decide what to do next, turn after turn, before handing
off or answering? Module 5 covers composing multiple agents.

## ReAct: reason, then act, interleaved

The most common pattern. Each turn, the model produces a **thought** (why),
an **action** (a tool call or a final answer), and receives an
**observation** (the tool's result) before the next turn.

```
Thought: I need the customer's order total before I can compute a refund.
Action:  lookup_order(order_id="A1002")
Observation: {"order_id": "A1002", "total": 310.00, "item": "monitor"}
Thought: Total is $310, over the $100 auto-approval limit — route to review.
Action:  final_answer("route: refund_review")
```

Strengths: simple, debuggable (the thought trace is a readable audit log),
and naturally handles "I don't know yet, let me look" — the model doesn't
need the answer to exist in-context already.

Weaknesses: latency (one model round-trip per step) and no guarantee the
model's stated reasoning matches what it actually does — treat the thought
as a debugging aid, never as a verified fact (Module 7 covers checking
*behavior*, not the model's narration of it).

## Plan-and-Execute: separate planning from doing

Instead of deciding one step at a time, produce an entire plan up front, then
execute it (optionally re-planning if a step fails or the world changes).

```
Plan:
  1. Look up the order
  2. Classify the complaint
  3. If refund-eligible, compute the refund amount
  4. If refund > $100, request approval
  5. Draft a reply
Execute step 1 → step 2 → ...
```

Strengths: fewer expensive "what next" calls, a plan you can show a user or
log as an artifact before committing to action, and a natural point to
validate the plan against policy *before* anything runs.

Weaknesses: brittle under a plan that turns out to be wrong mid-execution —
you need explicit re-planning, or you're stuck executing a bad plan
faithfully. Good fit when the step sequence is usually predictable (which is
also, not coincidentally, when you should consider making it a **declared**
pipeline instead of a model-generated plan at all — see below).

## Reflexion / self-critique loops

The agent produces an attempt, a separate pass (the same or another model)
critiques it against the goal, and the agent revises. Useful for quality-
sensitive single outputs (code, long-form writing) where a second look
catches errors the first pass missed. Costs roughly 2x the latency/tokens for
a quality gain that has to be worth it — this is Module 7's "judge" pattern
applied inline instead of after the fact.

## The fork every design has to make: let the model decide the *shape*, or declare it?

This is the most consequential architectural decision in this course, and it
recurs at every scale:

- **Model decides at runtime** (ReAct, open-ended planning): maximally
  flexible, required when the step sequence genuinely can't be known ahead of
  time (open-ended research, novel troubleshooting). Costs: harder to test
  (the path taken varies run to run), harder to bound cost/latency, harder to
  audit.
- **Declared ahead of time** (a graph/YAML/state machine, with the model
  used *inside* specific nodes for judgment calls): the control flow is data
  you can read, lint, version, and unit test without calling a model at all.
  Costs: you have to have anticipated the shape.

Production systems converge on the declared shape far more often than
tutorials suggest, for the same reason production backends prefer explicit
state machines over "figure it out at runtime" logic: **testability**. A
declared graph lets you write `pytest` tests for routing logic without
mocking an LLM, and lets an evaluator (Module 7) check "did this run match
what was declared" — a question that is meaningless if nothing was declared.

The working example in this course takes the declared approach:
`course/example/pipeline.yaml` fixes the shape (lookup → classify → route →
specialist → gate → reply); the model is used only for the two genuinely
judgment-requiring nodes (classification, drafting). This is the pattern
you'll see argued for again in Module 5.

## Minimal ReAct loop, in pseudocode

```python
def run_agent(goal, tools, model, max_steps=8):
    history = [f"Goal: {goal}"]
    for _ in range(max_steps):                    # bound it — Module 8
        thought, action, args = model.decide(history, tools)
        if action == "final_answer":
            return args["answer"]
        observation = tools[action](**args)         # Module 3
        history.append(f"Thought: {thought}\nAction: {action}({args})\nObservation: {observation}")
    raise TimeoutError("agent did not converge within max_steps")
```

Note `max_steps`. An agent that decides its own next step can, in principle,
never stop. Every control loop in this course has an explicit ceiling —
this is not an edge case, it's a required part of the design (Module 8).

## Exercise

For the workflow you diagrammed in Module 1's exercise: would you implement
it as free-running ReAct, a fixed plan-and-execute, or a fully declared
graph? Justify it by how predictable the step sequence is, not by which
sounds more sophisticated — "declared" is usually the right, boring answer.
