# Module 4 — Memory & State

An agent without state can only do one step. Everything past a single tool
call requires a place to put what's been learned so far, and a rule for who
reads and writes what. This module is about designing that "database" for
your pipeline before you design the pipeline itself.

## The session dict pattern

The dominant pattern in production multi-agent systems (and the one used by
this course's example and by the `O2A_EVALUATOR_SPEC.md` pipelines this repo
was built to evaluate) is deceptively simple: one shared mutable dict, keyed
by name, threaded through every step.

```
session = {"order_id": "A1002"}
  → lookup_order_step  writes  "order"
  → classify_intent    writes  "intent"
  → intent_router       reads  "intent"          (decides, doesn't write data)
  → calc_refund         reads  "order"    writes  "refund_amount"
  → refund_gate         reads  "refund_amount"  writes "refund_approved"
  → draft_refund_reply  reads  "intent","refund_amount","order"  writes "final_reply"
```

Every step declares two contracts: **input_keys** (what it reads) and an
**output_key** (what it writes). This is the entire interface between steps —
there is no other channel. That constraint is what makes the system
analyzable: you can build a producer/consumer map (which step writes what,
which steps read it) purely from these declarations, without running
anything. Module 7's evaluator leans on exactly this map to catch a step
reading a key nobody wrote, or writing a key nobody reads.

## Why not just pass everything in the prompt?

You could skip declared keys and just dump the whole conversation/state into
every model call. Two things go wrong at scale:

- **Context cost.** Every step re-pays for tokens it doesn't need. A step
  that only needs `refund_amount` shouldn't also carry the full order
  history, the original message, and every prior model's reasoning.
- **No contract to check.** If a step just "sees everything," there's no way
  to answer "is this step using undeclared session state" or "did this
  step's prompt leak state it wasn't supposed to depend on" — a real failure
  mode: a model quietly starts relying on a field that happens to be present
  today and breaks when a pipeline change removes it. Declared `input_keys`
  turn that from a runtime surprise into a static check (see Module 7 §N3).

## Short-term vs. long-term memory

- **Short-term (session/context)**: the state for *this* run — what this
  course means by "session dict." Lives for the duration of one pipeline
  execution, then normally discarded (or archived for evaluation — with PII
  handling, see Module 8).
- **Long-term (external store)**: state that outlives one run — a customer's
  history, a vector store of past tickets for retrieval, a fact the system
  learned last week. This is read *into* the session at the start of a run
  (typically via a tool/retrieval step) and, if it should persist, written
  back out explicitly at the end. Don't conflate the two: a long-term store
  that gets silently mutated by every run's short-term scratch state becomes
  impossible to reason about.

## Context window budgeting

The context window is a finite, shared resource across every step that
touches the model. Three practical tactics, in order of preference:

1. **Pass keys, not history.** Give each LLM step exactly its declared
   `input_keys`, rendered fresh, instead of the accumulated transcript of
   every prior step. This is usually sufficient and is what the declared
   session-dict pattern gives you for free.
2. **Summarize instead of truncate**, when a genuine conversation history
   must be carried (e.g., a multi-turn chat agent). Truncation silently drops
   the oldest, possibly still-relevant, context; summarization compresses it
   under your control.
3. **Cap and monitor.** Track total session size and flag pipelines that
   accumulate unbounded state (Module 7's O-R4 check: session key count over
   a threshold at pipeline end) — an early warning that something is being
   written and never consumed, or that a loop is appending instead of
   overwriting.

## Idempotency and resumability

Long-running or gated pipelines (Module 8's human-approval gates; anything
that can be paused for hours) need their state to survive a restart. Two
requirements this implies:

- **The session must be serializable.** No live objects, DB connections, or
  closures in session values — only data. If a pipeline needs to persist
  across a gate, it should be able to write the session to a durable store
  (a row, a document) and reconstruct it later with no loss.
- **Steps before a resume point must be safe to have already run.** A
  resumed pipeline re-enters at the gate, not step 1 — but if your engine
  instead reruns the whole thing, every prior step needs to be idempotent
  (Module 3) or the resume will double-charge, double-send, or double-log.

## State as a state machine, not a free-for-all

The temptation with a shared dict is to let any step read or write any key.
Resist it. Two disciplines keep the session dict analyzable as the pipeline
grows:

- **Declare `input_keys` even when you could just reach into `session`
  directly.** The declaration is what makes the contract checkable.
- **Don't let two steps write the same key** unless one is explicitly a
  later override in a documented sequence. Two producers for one key is a
  correctness bug waiting for a race or an ordering change to surface it.

## Exercise

Take the pipeline you sketched in Module 1. List every piece of state that
flows between your steps as `key: producer → [consumers]`. Any key with zero
consumers, or a consumer that appears before its producer in your step
order, is a bug you just found for free — before writing a line of code.
This is precisely the static analysis Module 7 automates.
