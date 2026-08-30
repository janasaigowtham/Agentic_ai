# Module 7 — Reliability, Evaluation & Observability

This is the module that separates a demo from a system you'd put in
production. Agentic pipelines fail in two qualitatively different ways, and
you need a different tool for each.

## Two failure classes

- **Structural / control-flow failures** — a wrong router branch, a step
  that read a session key nobody wrote, a duplicate tool call, a gate that
  didn't actually wait. These are ordinary software bugs. They have exact,
  checkable contracts (Modules 3–5 all declared these contracts on purpose)
  and **fail silently** — no exception, no error log, just a subtly wrong
  outcome. You do not need a model to catch them; you need a rule.
- **Semantic failures** — the output is well-formed and the control flow was
  correct, but the *content* is wrong: the drafted reply ignores the
  customer's actual complaint, or a step re-derives a number from text
  instead of using the structured value already sitting in session. These
  need judgment to detect — a rule can't tell you a reply "sounds off,"
  but another model call, given the right rubric, often can.

Conflating these two is the most common evaluation mistake: teams either
throw an expensive LLM judge at every check (slow, costly, and judges are
themselves non-deterministic — bad for catching a router bug that should be
a 100%-reproducible assertion), or they try to write regexes for semantic
quality (brittle, and misses genuine content problems). Match the tool to
the failure class.

## Observability: capture what actually happened

You cannot evaluate behavior you didn't record. The standard mechanism is
**tracing**: one span per agent step, capturing at minimum:

- which agent ran, and its declared `agent_class`
- the session keys present before the step (`keys_before`) and after
  (`keys_after` / `keys_written`)
- for an LLM step: the rendered prompt and which session keys it actually
  referenced
- for a router: every route it evaluated, which ones matched, and which one
  it chose
- duration, and whether it errored

This course's `course/example/tracing.py` implements a minimal version of
this — one `Span` per node, with `keys_before`/`keys_after` — deliberately
small enough to read end to end. Production systems export the same shape
over OpenTelemetry; see `O2A_EVALUATOR_SPEC.md` in this repo's root for a
fully worked, production-scale version of this idea (the `o2a.*` span
attribute namespace, PII-safe capture, and a much larger check suite) — this
module gives you the concepts that spec assumes you already have.

**The evaluator observes; it never drives.** It attaches to a run that would
have happened anyway and scores it afterward. An evaluator that can trigger
pipeline behavior is a second, untested code path with production
permissions — keep it strictly read-only.

## What to check, structurally

Every one of these is possible using only the declared graph (Module 5) and
the captured trace, with no model call:

- **Did everything that should have run, run?** Compare declared nodes
  against executed nodes — excluding nodes behind a router branch that
  wasn't chosen (those are *supposed* to be absent).
- **Did a router choose consistently with its own declared conditions?**
  Re-evaluate the conditions against the session state the trace captured;
  flag if the chosen branch isn't the one that should have matched.
- **Did any step run more than once with identical input?** Wasted cost at
  best (a redundant DB call), a real bug at worst.
- **Did a step read a session key nothing produced** (a silent `None`,
  masked if the step doesn't validate strictly)?
- **Did the pipeline actually produce its declared final output?**
- **Did a human-approval gate actually pause,** or did it resolve
  suspiciously instantly for a case that should have required review?

`course/example/eval.py` implements a small version of exactly these checks
(`O-S3`, `R-R1`, `D1`, `G-R1`) against the example pipeline — read it next to
this list; the check IDs match the naming convention `O2A_EVALUATOR_SPEC.md`
uses for the same categories at production scale.

## What to check, semantically: LLM-as-judge

For a step whose *output quality* matters (not just "did it run"), the
standard pattern is a second model call — the **judge** — given a rubric
built from the node's declared role, not free-form:

```
Declared role: draft a reply to a refund request
Declared inputs: intent, refund_amount, order
Input received: {...}
Output produced: {...}

Score 1-5: did the output fulfil its declared role? Did it use the
inputs it was given, or ignore/re-derive them? Is it complete?
Did it stay within its declared scope?
```

Grounding the rubric in what the node *declared* it would do (Module 3's
contract) turns "is this good" — an unanswerable, vibes-based question —
into "did this step do the specific job it was assigned," which a judge can
actually assess consistently.

Three judge-specific design rules, because a judge is itself a model call
with all of Module 2's non-determinism:

- **Budget it.** Cap judge calls per run; they're the most expensive check
  you have.
- **Skip it when structure already failed.** There's no point judging output
  quality for a pipeline that didn't execute as declared — a structural
  SEV-1 should suppress semantic checks downstream of it (this is exactly
  the skip rule `O2A_EVALUATOR_SPEC.md` §5.2 specifies).
- **Treat judge output as a hint, not a verdict**, until it's been validated
  against a batch of human-labeled examples. An unvalidated rubric can be
  confidently, consistently wrong.

## Severity, not just pass/fail

Not every flaw is a blocker. A useful scale (used throughout
`O2A_EVALUATOR_SPEC.md` and mirrored in `eval.py`):

| Severity | Meaning | Example |
|---|---|---|
| SEV-1 | Blocker — wrong or missing output | pipeline never produced its final answer |
| SEV-2 | Real problem | router chose a branch its own conditions don't support |
| SEV-3 | Design smell | a gate resolved suspiciously fast |
| INFO | Notable, not a problem | a check itself failed to run (report it, don't hide it) |

Aggregate per axis (structural, routing, efficiency, prompt quality, output
quality) rather than one flat score — a system can be structurally perfect
and semantically weak, or vice versa, and a single number hides which.

## Regression testing for a non-deterministic system

You can still write deterministic tests for a system with a model inside it,
by testing the parts that are deterministic and pinning the parts that
aren't:

- **Unit-test deterministic nodes directly** (tools, transforms, routers) —
  no model involved, ordinary `pytest`.
- **Seed known flaws into a fixture trace** and assert your checks catch
  them — and, just as important, assert they *don't* fire on a clean run
  (a false-positive suite is not optional; see `course/example/tests/`).
- **Use a deterministic mock model in tests**, a real one only in a smaller,
  separate "does the real model still behave" suite that runs less often and
  tolerates more flakiness. `course/example/llm_client.py`'s `MockLLM` exists
  for exactly this reason — it's not a toy, it's the standard way to make an
  agentic pipeline's control flow testable at all.

## Exercise

Run `python run_pipeline.py` in `course/example/` and read the printed
scorecards. Then open `course/example/tests/test_pipeline.py` and find the
test that deliberately forces a gate to resolve too fast — this is a seeded
flaw, exactly like the pattern above. Add one new seeded-flaw test of your
own (e.g., make the router misroute and assert `R-R1` fires).
