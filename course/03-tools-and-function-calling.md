# Module 3 — Tools & Function Calling

Tools are how an agent has effects beyond generating text. Designing them
well is closer to API design than prompt design — get the contract wrong and
no amount of prompting fixes it.

## The mechanics

1. You declare a tool's name, description, and a JSON-schema of its
   arguments to the model.
2. The model, mid-generation, emits a structured call: `{"name": "lookup_order",
   "arguments": {"order_id": "A1002"}}` instead of (or alongside) prose.
3. Your code executes the *real* function — the model never runs it — and
   returns the result as a new message.
4. The model continues, now with that result in context.

The model only ever decides *whether and how* to call a tool. It never
executes anything. Every safety and reliability property in Modules 7–8
follows from keeping that boundary strict.

## Designing the contract

Treat each tool like a small internal API:

- **Name and description are the interface.** The model chooses tools by
  reading them, the same way a developer picks a library function by reading
  its docstring. A vague description ("handles orders") gets called at the
  wrong times or not called when it should be. Be as specific as a good
  function docstring: what it does, when to use it, what it returns.
- **Narrow scope beats a general one.** `lookup_order(order_id)` is easier
  for a model to use correctly than `run_query(sql)` — and the narrow one is
  also the one you can validate, rate-limit, and reason about the blast
  radius of. A general "run arbitrary SQL" tool pushes an entire injection
  and authorization surface onto model output (see Module 8).
- **Return structured, minimal results.** Return the fields the next step
  actually needs, not a full object dump — every extra field is tokens spent
  and a chance the model latches onto something irrelevant.
- **Declare idempotency.** Is calling this tool twice with the same
  arguments safe (a lookup) or not (charging a card)? Non-idempotent tools
  need either a dedup/idempotency-key layer or a gate before they run
  (Module 8) — never "trust the model not to call it twice."

## Deterministic vs. non-deterministic actions

This distinction matters enough to design around explicitly, and it recurs
in Module 7 as the axis an evaluator uses to decide *how* to check a step:

- **Deterministic tools** (a DB query, a calculator, a lookup): given the
  same input, always produce the same output. You can unit-test them with
  zero involvement of a model, and you should — bugs here are ordinary
  software bugs, not "the model got confused."
- **Non-deterministic steps** (anything that's actually an LLM call): the
  same input can produce a different output next time. These need the
  judgment-based checking Module 7 covers (LLM-as-judge, semantic rubrics) —
  a plain `assert` doesn't work at this layer.

A well-designed agentic system pushes as much as possible into the
deterministic category. If a "tool" is really just "ask the model to compute
something a function could compute exactly," replace it with the function —
it's cheaper, faster, and testable with a plain assertion instead of a judge.

## Error handling and retries

Tool calls fail like any external call: timeouts, 5xxs, malformed input the
model guessed at. Three rules:

- **Return errors as data, not exceptions that vanish.** Feed the error back
  to the model as an observation ("order not found") so it can recover
  (retry with corrected input, try another path, or say it can't proceed) —
  don't silently swallow it and let the model hallucinate a result.
- **Retry the deterministic call, not the model's decision to call it.**
  If a DB call times out, retry the DB call with backoff; don't re-run the
  whole reasoning step and hope the model calls it again.
- **Cap retries per tool per turn.** Combined with the step ceiling from
  Module 2, this bounds the worst case: a flaky tool can't turn into an
  unbounded loop.

## Permissions and sandboxing (preview of Module 8)

A tool is a capability grant. Before adding one, ask: what's the worst thing
this agent can do if the model calls this tool with adversarial or simply
wrong arguments? Read-only lookups are low risk. Anything that spends money,
deletes data, or sends external communication should sit behind a gate
(Module 8) rather than firing directly from a model's decision, no matter how
well-prompted.

## Example: a tool contract, worked

From this course's example (`course/example/tools.py`):

```python
def lookup_order(order_id: str) -> dict:
    """Look up an order by id. Returns {order_id, total, item};
    unknown ids return a zero-total placeholder rather than raising,
    so a bad id becomes a normal branch in the pipeline instead of a
    crash the model has to improvise around."""
```

Notice what's *not* here: no SQL, no arbitrary filters, no write path. The
tool is exactly as capable as the one thing this pipeline needs it to do.

## Exercise

Pick three tools you'd want for the workflow from Module 1. For each, write
the contract as a docstring (name, args, return shape, idempotency) before
writing any implementation. If you can't describe it in one tight paragraph,
it's probably two tools trying to be one.
