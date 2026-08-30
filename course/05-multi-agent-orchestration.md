# Module 5 — Multi-Agent Orchestration

Module 2 asked "how does one agent decide its next step." This module asks
the system-design question one level up: how do you compose *several*
narrow agents into a pipeline, and who's in charge of the wiring? This is
also where the course's working example (`course/example/`) stops being a
diagram and becomes runnable code — open `pipeline.yaml` alongside this
module.

## Why decompose at all

You could build one enormous agent with every tool and a giant instruction
covering every case. It's usually worse, for reasons that are just distributed-
systems reasons in disguise:

- **Separation of concerns.** A classifier's instruction is one paragraph. A
  "do everything" agent's instruction is a wall of edge cases that gets
  harder to change safely as it grows — the same reason you don't want a
  10,000-line function.
- **Cost and latency control.** Route cheap, narrow requests to a cheap
  model; reserve expensive reasoning for the step that needs it. One
  monolithic agent pays the most-expensive-step's cost on every call.
  (Module 9 covers model routing/fallback in production.)
- **Testability.** A narrow agent's contract (Module 3) is small enough to
  write real test cases against. A monolith's behavior space is not.
- **Independent evolution.** You can swap the refund-drafting agent's prompt
  without touching classification, and verify only that one node changed
  (this is exactly what the `pipeline_hash` idea in `O2A_EVALUATOR_SPEC.md`
  is for: detecting that a specific sub-agent changed under an otherwise
  unchanged pipeline).

## Orchestration topologies

### Sequential

Steps run in a fixed order, each consuming the prior steps' outputs.
Simplest topology; use it whenever the step order genuinely doesn't depend
on runtime data.

```
A → B → C → D
```

In the example: `refund_flow` is a sequential chain — `calc_refund` →
`refund_gate` → `draft_refund_reply`.

### Router (decision)

One step evaluates conditions against session state and hands off to
exactly one of several branches. This is how you encode "the next step
depends on what we just learned" *without* letting the model freely improvise
the graph shape — the branches are declared; only which one fires is
runtime-determined.

```
             ┌─ refund_flow
classify → router ─ technical_flow
             └─ general_flow  (catch-all)
```

Two design rules that matter enough to be checks in Module 7:

- **Always have a catch-all branch**, or an explicit "no match" path. A
  router built only from `==` conditions with no default silently drops any
  input it didn't anticipate — a real failure mode, not a theoretical one.
  See `intent_router`'s `general_flow` route (`conditions: []`, lowest
  priority) in `pipeline.yaml`.
- **Priorities must be unambiguous.** If two routes can match the same input
  with the same priority, which one fires is an implementation accident, not
  a design decision.

### Parallel (fan-out / fan-in)

Independent steps run concurrently, then their outputs are joined. Use it
when steps don't depend on each other's output and latency matters (e.g.,
looking up order history and checking loyalty status at the same time). Not
used in this course's example (its steps are all dependent), but the same
session-dict pattern applies: each parallel branch writes a distinct key, and
a join step reads all of them.

### Hierarchical (orchestrator–worker)

A top-level orchestrator delegates whole sub-problems to specialized worker
agents (which may themselves be sequential/router pipelines), rather than
sequencing individual steps itself. This is a scaling pattern for when the
number of distinct "flows" grows large enough that one flat sequence becomes
unreadable — each worker gets tested and versioned independently, and the
orchestrator's job shrinks to "which worker handles this."

### Blackboard / shared-state

Multiple agents read and write a shared space without a fixed call order,
each contributing when its trigger condition is met. Powerful for open-ended
collaboration (several specialists converging on one answer) but the hardest
to test and reason about — there's no fixed graph to check against, only
runtime behavior. Use only when a fixed topology genuinely can't express the
problem; it forfeits most of the static-analysis benefits in Module 7.

## Control flow as data

The single biggest design decision in this module, previewed in Module 2:
should the graph's *shape* live in code, or in a declarative file the
engine interprets?

This course's example puts it in `pipeline.yaml`:

```yaml
name: support_triage_orchestrator
agent_class: orchestrator
output_key: final_reply
sub_agents:
  - name: lookup_order_step
    agent_class: tool_agent
    tool: lookup_order
    input_keys: [order_id]
    output_key: order
  - name: classify_intent
    agent_class: llm_agent
    instruction: "classify: {message}"
    input_keys: [message]
    output_key: intent
  - name: intent_router
    agent_class: router_agent
    ...
```

and a small, generic engine (`agent_framework.py`) that only knows how to
execute six `agent_class` values (`llm_agent`, `tool_agent`,
`transform_agent`, `sequential_agent`, `router_agent`, `gate_agent`) plus a
top-level `orchestrator`. Adding a new pipeline, or changing this one's
shape, means editing YAML — the engine doesn't change. This mirrors, at
small scale, the real production pattern `O2A_EVALUATOR_SPEC.md` describes:
"generalize by agent_class, not by pipeline name."

The payoff shows up immediately in the next two modules: Module 6's planning
discussion is about what happens when a step's *content* still needs
judgment even though its *position* in the graph is fixed; Module 7's
evaluator is only possible at all because there's a declared graph to
diff actual behavior against.

## Running the example now

```bash
cd course/example
pip install -r requirements.txt
python run_pipeline.py
```

Watch the printed trace: notice that for a "technical" complaint, only
`technical_flow`'s steps execute — `refund_flow` and `general_flow` never
run, because the router chose one branch. That's the router topology, live.

## Exercise

Take your Module 4 producer/consumer table and draw it as a graph. Which
edges are always taken (sequential), which are conditional (router), and are
any two steps truly independent of each other (a parallel opportunity)? If
your diagram has a branch with no catch-all, fix it now — it's the single
most common orchestration bug.
