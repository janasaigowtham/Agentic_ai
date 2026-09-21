# Trajectory Review Harness — SDK Integration Specification
# Version: 0.1
# Companion to: TRAJECTORY_REVIEW_HARNESS_SPEC.md (core architecture, data model, stages)
# Purpose: how a target agent framework integrates with the harness — the adapter contract,
# the one-line attach() call, and the boundary between what ships inside the target
# application (the SDK) and what runs as a separate service (judgment).

---

## 1. PURPOSE AND SCOPE

The main spec defines what the harness does once it has a Trajectory. This document defines
how a Trajectory gets produced from a live target system in the first place, and how that
capture mechanism ships as something installable — an SDK — rather than something built
bespoke per target.

Precedent: this reuses the `FrameworkAdapter` pattern already built and shipped in the
`o2a_eval` package (`adapters/base.py`) — a dataclass of framework-specific callables plus a
single `attach(adapter, config)` call. That pattern solved the same problem for a different
capture target; this spec generalizes it for the trajectory review harness and documents the
concrete adapter needed for the real O2A/ADK repo.

---

## 2. NON-NEGOTIABLE INTEGRATION CONSTRAINTS

These extend §2 of the main spec with constraints specific to shipping this as an SDK inside
someone else's running application.

1. **Zero request-path latency on unobserved runs.** The overwhelming majority of runs are
   never flagged for review. The SDK's presence must cost those runs nothing beyond one
   cheap flag check — no synchronous capture work, no blocking I/O, no added round trip.
2. **Opt-in, single-record capture — never always-on.** Per the main spec's scope (§1): the
   harness never runs in batch. The SDK does not passively record every run "just in case";
   it activates capture only for the one record a human has already selected.
3. **The SDK never contains judgment logic.** Orient, the specialists, the Aggregator, and
   Fact-check are model calls that belong to the harness's own service (§3 of the main
   spec). The SDK's only job is getting a clean Trajectory out of the target system and
   submitting it — it must not embed any of the judgment pipeline in-process.
4. **A harness failure cannot affect a normal execution run.** If capture, queuing, or
   submission fails for any reason, the target system's run completes exactly as it would
   have with no SDK installed. Capture failures are swallowed and logged, never raised into
   the target application's own error path.
5. **One integration call per target framework, not one per target application.** The
   adapter is written once per framework (e.g. one ADK/O2A adapter serves every application
   built on that framework); the application itself only calls `attach()`.

---

## 3. THREE-LAYER ARCHITECTURE

```
┌─────────────────────────────────────────────────────────────┐
│  Target application (e.g. an O2A pipeline)                  │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Client SDK  (thin — ships inside the target process)│    │
│  │  - attach(adapter, config)                           │    │
│  │  - capture taps, per-record trigger check            │    │
│  │  - async queue + submit                              │    │
│  └───────────────────────┬───────────────────────────────┘   │
└──────────────────────────┼───────────────────────────────────┘
                            │  Trajectory (submitted, async)
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  Harness service  (separate process — runs the judgment)     │
│  Core: Orient → Specialists → Aggregator → Fact-check → Gate │
│  Own tables, own config, own CLI — per main spec §7          │
└─────────────────────────────────────────────────────────────┘
```

### 3.1 Core (framework-agnostic, service-side)

Everything already defined in `TRAJECTORY_REVIEW_HARNESS_SPEC.md` §4–§6: the data model, the
stage specifications, the specialist registry. Nothing in this layer references a specific
target framework. It runs as its own service, consuming submitted Trajectories.

### 3.2 Adapter (framework-specific, the extension point)

One implementation per target framework. Encodes exactly how to hook into that framework's
real execution model — which methods to wrap, what "before" and "after" mean for that
framework, and any known gaps (like the ADK plugin-bypass case, §6 below).

### 3.3 Client SDK (thin, ships inside the target application)

The only code that runs inside the target application's own process. Its job: install the
adapter's hooks, check whether the current record is flagged for observation, and — only if
so — capture and queue the resulting events for async submission to the harness service.
Contains no judgment logic and no persistent state beyond an in-memory queue.

---

## 4. THE ADAPTER CONTRACT

Mirrors `FrameworkAdapter` from `o2a_eval/adapters/base.py`, generalized for this harness.

```
Adapter:
  framework_name          string    — e.g. "adk"
  base_class_path          string    — the framework's base agent class, dotted path
  run_method                string    — the method that correctly invokes the plugin/callback
                                        chain (e.g. "run_async")
  bypassed_paths            list[string]?  — known methods on specific subclasses that skip
                                        run_method internally (e.g. decision_router_agent
                                        and iteration_agent calling _run_async_impl directly).
                                        Declared explicitly, not discovered at runtime — see §6.

  get_agent_name(agent)               -> string
  get_agent_class(agent)               -> string   — must match the target's own declared
                                                       agent_class field where one exists
  get_session_state(ctx)                -> dict     — the equivalent of state_delta
  get_declared_contract(agent)           -> dict?    — pointer to the agent's own declared
                                                       spec (e.g. YAML node), if the target
                                                       has one — see main spec §8.3
  resolve_query_or_template(agent, ctx)   -> string?  — pre-binding, params never resolved to
                                                       literal values (PII discipline, per
                                                       main spec's evidence-scrubbing rule)

  register_tap_a(callback)               — wires a before/after hook via the framework's own
                                            plugin or callback mechanism, where one exists
  register_tap_b(callback)               — wires a hook into the framework's top-level event
                                            stream, independent of tap A, as the fallback that
                                            covers bypassed_paths
```

An adapter that has no plugin/callback mechanism at all (`register_tap_a` is a no-op) still
functions — it just means every step in that framework gets Tap-B-only coverage: after-the-
fact state, no pre-execution snapshot. This must be surfaced to specialists the same way the
ADK adapter surfaces its own two bypassed classes (§6) — a known limitation stated plainly,
not a silent gap.

---

## 5. attach() — THE INTEGRATION CALL

```
attach(adapter: Adapter, config: AttachConfig) -> None

AttachConfig:
  harness_endpoint       string   — where captured Trajectories are submitted
  is_observed(record_id)  callable  — returns bool; the per-record trigger check (§7)
  queue_mode              enum(async, sync_test_only)  — sync_test_only exists for local
                                                          testing only, never for production
```

One call, made once at application startup, in the target application's own initialization
code — this is the "one line" integration point. Everything after that is the adapter's taps
firing during live execution and the SDK's queue draining asynchronously.

---

## 6. REFERENCE IMPLEMENTATION: THE O2A / ADK ADAPTER

Concrete instantiation of §4, grounded in the confirmed gap from the architecture review:

- `run_method`: `run_async` — the only method that fires
  `plugin_manager.run_before_agent_callback` / `run_after_agent_callback`.
- `bypassed_paths`: `decision_router_agent`, `iteration_agent` — both call
  `sub_agent._run_async_impl(...)` directly for their own sub-agents, skipping the plugin
  manager entirely.
- `register_tap_a`: a new passive `BasePlugin` subclass registered in `agent_executor.py`
  alongside the existing `RetryZeroTokenPlugin`. Fires for every class not in
  `bypassed_paths`.
- `register_tap_b`: the existing `async for event in runner.run_async(...)` loop already
  present in `agent_executor.py` — the adapter attaches a passive listener to it, which
  catches every step including ones from `bypassed_paths`, but only after-the-fact
  `state_delta`, never the pre-execution query/template Tap A would have captured.
- `get_declared_contract`: resolves against the target's own YAML-per-agent directory
  (per the original O2A spec), giving specialists the declared input_keys/output_key/query
  for the step being reviewed.

This is the adapter that ships as the default for any O2A/ADK-based target. A future target
on a different framework needs a new adapter written against §4's contract — it does not
reuse this one's internals, only its shape.

---

## 7. PER-RECORD TRIGGER ENFORCEMENT

The mechanism that makes constraint §2.1 (zero latency on unobserved runs) actually true,
not just stated:

1. `is_observed(record_id)` is checked once, synchronously, cheaply (a flag lookup — not a
   network call) at the point the adapter's taps would otherwise fire.
2. If false: the taps do not fire at all. The wrapped method calls through to the original
   framework behavior with no added work. This is the path taken by the overwhelming
   majority of runs.
3. If true: the taps fire, events are appended to an in-memory queue, and a background
   consumer drains that queue and submits the assembled Trajectory to the harness service
   asynchronously — never blocking the request path.
4. A capture failure at any point in steps 2–3 is caught and logged inside the SDK; it never
   propagates into the target application's own execution or error handling (constraint §2.4).

---

## 8. PACKAGING AND VERSIONING

- The client SDK and the adapter interface version independently from the harness service's
  core judgment pipeline — a target application should be able to upgrade its SDK without
  redeploying the harness service, and vice versa.
- Each adapter declares which version(s) of its target framework it's verified against (e.g.
  "ADK adapter, verified against `base_agent.py` as of the architecture review in §6" — a
  framework upgrade that changes `run_method`'s callback behavior invalidates that
  verification and requires re-review, not a silent assumption that it still holds).
- `bypassed_paths` is a declared, versioned list — not something the adapter tries to detect
  automatically. If the root-cause fix from the main spec's open items (making
  `decision_router_agent` / `iteration_agent` call `run_async` instead of
  `_run_async_impl`) ever ships, the adapter's `bypassed_paths` list is what gets updated —
  a one-line change, with the fix's own PR as the trigger to update it.

---

## 9. NON-GOALS

- Does not attempt framework-agnostic trace ingestion ("any trace shape in"). Every adapter
  is written and verified against one specific framework's real source, on purpose — see the
  generality-vs-depth tradeoff discussed when this SDK was first proposed. A wrong capture
  assumption here produces false evidence for every specialist downstream; that risk is
  judged worse than the convenience of a universal ingester.
- Does not define the harness service's own deployment (how it scales, where its tables
  live) — that's the main spec's and its own infrastructure's concern, not the SDK's.
- Does not run judgment client-side under any configuration. If latency from a separate
  service ever becomes a real constraint, that's a reason to revisit the harness service's
  deployment topology — not a reason to move Orient/Specialists/Aggregator into the SDK.
