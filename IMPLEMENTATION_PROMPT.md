# Build prompt: O2A pipeline evaluator

Paste this into Claude Code inside your repo. It is self-contained — the agent needs
no prior conversation. It builds the evaluator from scratch, in order, with tests.

---

````
Build a Python package called `o2a_eval` that evaluates our O2A agent pipelines.

## Context you need

Our pipelines are defined as a **tree of YAML files**, one file per agent, in a flat
directory. Agents reference each other **by name**, not by path:

    # pmi_ddn_pipeline.yaml
    name: pmi_ddn_pipeline
    version: '2.0'
    output_key: pmi_ddn_final_output
    agent_class: resumable_orchestrator
    sub_agents:
      - name: pmi_ddn_pre_process
      - name: pmi_ddn_extraction_decision_router
      - name: pmi_ddn_approval_gate
      - name: pmi_ddn_post_process

Sub-agents can themselves have `sub_agents`, arbitrarily deep. Every agent declares
`agent_class`. The classes we use:

| agent_class | kind | what it does |
|---|---|---|
| `LlmAgent` | NON-DETERMINISTIC | runs an `instruction:` prompt against a model |
| `database_agent` | deterministic | runs SQL from a `default_db_yaml` block |
| `slv_transformation_agent` | deterministic | `transform:` block, `$fn`/`$cond`/`$each` |
| `transformation_agent` | deterministic | same |
| `decision_router_agent` | control flow | picks one of `routes:` by condition + priority |
| `SequentialAgent` | control flow | runs `sub_agents` in order |
| `agent_gate` | control flow | pauses until an external event |
| `resumable_orchestrator` | control flow | top-level, resumable across gates |

State flows through a **session dict**. Every agent declares `output_key` (what it
writes) and `input_keys` (what it reads). Templates reference session with
`{{ key }}` or `{{ key[0].field }}` in transforms/SQL, and `{key}` inside LlmAgent
instruction blocks.

Representative node shapes:

    # database_agent — note default_db_yaml is an embedded YAML *string*
    name: pmi_ddn_fetch_msp_loan_data
    agent_class: database_agent
    output_key: pmi_ddn_msp_loan_data
    input_keys: [loan_number]
    default_db_yaml: |
      type: teradata
      connection:
        url: ${ENV:ECRM_DB_CONN}
      query: >
        SELECT M7.LN_NO, TRIM(M7.INV_CLASS_CODE) AS inv_class_code
        FROM MSP_LOAN_MASTER7_CS M7
        WHERE M7.LN_NO = LPAD(TRIM(CAST(:loan_number AS VARCHAR(10))), 10, '0')

    # slv_transformation_agent
    name: pmi_ddn_icmp_required_flag
    agent_class: slv_transformation_agent
    input_keys: [pmi_ddn_msp_loan_data]
    output_key: pmi_ddn_icmp_required_flag
    strict: false
    transform:
      pmi_ddn_icmp_required_flag:
        $cond:
          if:
            left: '{{ pmi_ddn_msp_loan_data[0].letter_effective_date }}'
            op: not_null
          then: Y
          else: N

    # decision_router_agent
    name: pmi_ddn_icmp_decision_router
    agent_class: decision_router_agent
    output_key: routing_decision
    routes:
      - conditions:
          - context_key: pmi_ddn_icmp_required_flag
            operator: eq
            value: Y
        target_agent: pmi_ddn_icmp_process
        priority: 10
      - conditions:
          - context_key: pmi_ddn_icmp_required_flag
            operator: eq
            value: N
        target_agent: pmi_ddn_no_icmp_path_handler
        priority: 20
    sub_agents:
      - name: pmi_ddn_icmp_process
      - name: pmi_ddn_no_icmp_path_handler

Traces already flow to Arize over OTel. We can also run pipelines locally.

## What to build

An evaluator that compares **declared intent** (YAML) against **actual behaviour**
(captured spans) and scores both halves of the pipeline. Deterministic agents have
exact contracts, so they get rule-based checks with no model. `LlmAgent` gets a
mixture of static prompt checks and judge-based semantic checks.

### Non-negotiable design constraints

1. **Observer, never driver.** The evaluator must never invoke the pipeline. Our
   runner already does that. The evaluator attaches, watches, scores on exit.
2. **One framework-specific file.** All coupling to our runner lives in
   `adapters/base.py`. Every other module reads only `Run` (spans) and
   `PipelineGraph` (YAML) and must be testable with fixtures alone.
3. **Captured data contains PII.** Session snapshots and rendered prompts hold loan
   numbers and borrower data. They must not survive the process. See PII section.
4. **The judge is on-prem.** Refuse non-local endpoints in code, not documentation.
5. **Checks are pure functions.** Signature `(Run, PipelineGraph, dict) -> list[Flaw]`.
   A check that raises must not sink the run.
6. **Generalise by agent_class, not by pipeline name.** A new use case with the same
   classes must work with zero new code — only optional config.

## Package layout

    o2a_eval/
      __init__.py           attach, load_pipeline, build_run, build_scorecard exports
      cli.py                score | lint | graph | checks subcommands
      core/
        models.py           Severity, Flaw, Node, Route, PipelineGraph, Span, Run, Scorecard
        graph.py            recursive multi-file YAML loader, producer/consumer maps
        trace.py            span dicts -> Run, tree reconstruction
        registry.py         @register decorator, check metadata
        rollup.py           run all checks, gate, score axes
      checks/
        __init__.py         imports every module so registration happens
        structural.py       D9 D5 D6 Q-R2 T-S7
        hygiene.py          D1 D2 N3 O-R3 O-R4
        database.py         S1-S8 R1 R5
        router.py           R-S1 R-S2 R-S3 R-S5 R-S7 R-R1
        prompts.py          N1a-N1d N2
        lifecycle.py        G-S1 G-R1 G-R2 O-S1 O-S3 Q-S1 Q-S3
        judge.py            LocalJudgeClient, N4, N5
      capture/
        ephemeral.py        EphemeralTraceStore
        attach.py           attach(), install_enrichment()
      adapters/
        base.py             FrameworkAdapter — THE integration seam
      report/
        scrubber.py         allow-list PII scrubber, scorecard/summary writers
        terminal.py         rich renderer with plain fallback
    tests/
      fixtures/agents/*.yaml
      fixtures/make_trace.py
      test_evaluator.py

## Data models (core/models.py)

`Severity(IntEnum)`: INFO=0 < SEV_3=1 < SEV_2=2 < SEV_1=3, with `.label` returning
"SEV-1" etc. and a `.parse()` classmethod. Ordering must work for comparisons.

`Flaw`: type, severity, node, description, fix, evidence dict, check_id, agent_class.
`evidence` may hold PII and gets scrubbed before persist.

`Node`: name, agent_class, source_path, yaml_hash, description, instruction,
output_key, input_keys, strict, raw, sub_agent_names, children, path (hierarchical
tuple), routes, query, db_type, db_url, transform, model. Add a `.kind` property
returning "non_deterministic" | "control_flow" | "deterministic" based on
agent_class, and `.walk()` yielding self then descendants.

`PipelineGraph`: name, version, root, nodes (name->Node), producers (session key ->
producing node name), consumers (session key -> [consuming node names]),
pipeline_hash. Helpers: `by_class()`, `deterministic()`, `non_deterministic()`.

`Span`: span_id, trace_id, parent_id, name, start_ns, end_ns, status, attributes,
events, children. Add **typed properties** that read `o2a.*` attributes so no check
ever does raw dict lookups: `agent_name`, `agent_class`, `output_key`, `input_keys`,
`keys_before`, `keys_after`, `keys_written`, `keys_read`, `input_value`,
`output_value`, `rendered_prompt`, `session_snapshot`, `template_references`,
`evaluated_routes`, `chosen_target`, `query_hash`, `row_count`, `duration_s`,
`errored`. JSON-string attributes get a `jattr()` helper that tolerates malformed JSON.

`Run`: trace_id, pipeline_name, pipeline_hash, spans, root, by_id, by_agent.
Helpers: `duration_s`, `spans_of_class()`, `llm_spans()`, `ordered()`, `spans_for()`.

`Scorecard`: pipeline, trace_id, pipeline_hash, duration_s, axes dict, overall,
flaws, judge_calls_used/budget, span_count, plus `.status` (PASS/FLAGGED/FAIL from
worst severity) and `.max_severity`.

## YAML loader (core/graph.py)

1. Index the whole agent directory by each file's `name:` field — linkage is by name.
2. Walk from the root pipeline, resolving `sub_agents[].name` against the index.
   A name with no file becomes a Node with `agent_class="<missing>"` so a check can
   flag it. Guard against cycles with a seen-set.
3. `default_db_yaml` may be a nested dict OR an embedded YAML string — handle both.
4. Some nodes use singular `input_key` instead of `input_keys` — merge them.
5. Build `producers` from every `output_key`.
6. Build `consumers` from `input_keys` PLUS regex-extracted template references in
   `transform`, `instruction`, and router `context_key`s. Two regexes:
   `{{ key }}` (double brace, capture the root identifier before any `.` or `[`) and
   `{key}` (single brace, negative lookarounds so it doesn't match doubles).
   Recurse through nested dicts/lists.
7. `pipeline_hash` = sha256 of all reachable node file hashes sorted and joined. This
   is how we detect a sub-agent edit under an unchanged pipeline file.

Also export `extract_template_keys(any_structure) -> set[str]` and
`extract_sql_params(query) -> set[str]` (matches `:param`).

## Registry (core/registry.py)

`@register(check_id, *, axis, stage, needs_judge=False)` decorator storing CheckMeta
with the function's first docstring line as its description. Axes:
`structural | efficiency | routing | prompt_quality | output_quality`. Stages:
`static | runtime | cross_trace`. `all_checks()` returns them ordered static ->
runtime -> cross_trace, then by id.

## Rollup (core/rollup.py)

Run every registered check. Skip a check when:
- it's in `checks_disabled`, or `checks_enabled` exists and it isn't in it
- it needs a judge and `config["_judge"]` is None
- **the run has no spans and the check's stage is runtime/cross_trace** (static-lint
  mode — this is easy to forget and produces nonsense flags)
- a structural SEV-1 already fired and the check is a judge-based semantic check
  (no point judging output quality of a pipeline that didn't execute as declared)

Wrap each check in try/except; on exception emit an INFO `check_error` flaw rather
than propagating.

Axis score = `max(0, 1 - sum(penalty))` with penalties INFO 0.0, SEV-3 0.05,
SEV-2 0.15, SEV-1 0.40. Overall = weighted average, defaults structural 0.30,
output_quality 0.25, efficiency 0.20, routing 0.15, prompt_quality 0.10,
overridable via `config["axis_weights"]`.

## Checks to implement

Each spec below is: id | axis | stage | what it detects | severity.

### structural.py
- **D9** structural runtime — declared node never fired, or an undeclared span
  appeared. **Exclude router branch targets from "missing"** — they're conditionally
  executed by design. Missing = SEV-1, unexpected = SEV-2.
- **D5** structural runtime — SequentialAgent children missing (SEV-1) or reordered
  (SEV-2). **Dedupe repeated children before the order comparison** — a child running
  twice is D1's concern, not an ordering fault.
- **Q-R2** structural runtime — a child span errored but later siblings still ran.
  SEV-1.
- **D6** structural runtime — a transform's `{{ key }}` refs weren't in
  `keys_before`. Discard `item` (the `$each` loop variable). SEV-1 when
  `strict:false` (silent null), SEV-2 otherwise. Skip when the span carries no
  session capture.
- **T-S7** structural static — `strict:false` on a transform agent. SEV-3 audit flag.

### hygiene.py
- **D1** efficiency runtime — same agent name + same hashed `input.value` ran 2+
  times in one trace. Skip control-flow classes. SEV-2 for `database_agent`
  (real cost), SEV-3 otherwise. Report wasted seconds.
- **D2** efficiency runtime — an executed node's `output_key` has no consumer.
  **Exclude**: routers/gates/orchestrators (their output goes to control flow), the
  pipeline's own `output_key`, direct children of the root, and the last node in
  declared order — those are terminal and consumed by the caller.
- **N3** prompt_quality runtime — for each LlmAgent span, keys in
  `template_references ∪ keys_read` that are NOT in that node's `input_keys` but ARE
  in `graph.producers`. SEV-2. **Compare against `input_keys`, not the consumers map**
  — the consumers map is built from instruction refs, so comparing against it can
  never flag anything. Report which other LLM spans also read the key.
- **O-R3** efficiency runtime — trace duration over `config["latency_budget_s"]`.
  SEV-1. Evidence must name the top 3 leaf spans by duration with pct-of-total.
- **O-R4** efficiency runtime — session key count at pipeline end over threshold
  (default 25). SEV-3.

### database.py
Use `sqlglot` if importable; degrade to regex when absent rather than failing.
- **S1** structural static — query doesn't parse for its declared dialect. SEV-1.
- **S2** structural static — `:params` not in `input_keys` (SEV-1); declared
  `input_keys` the query never references (SEV-3).
- **S5** structural static — `{{ }}` interpolated inside SQL instead of bound
  params. Injection risk. SEV-1.
- **S6** structural static — connection URL not sourced from `${ENV:...}`. SEV-1.
- **S8** structural static — DELETE/UPDATE/INSERT/TRUNCATE/DROP/ALTER in a
  `database_agent`. SEV-1.
- **S3** efficiency static — `SELECT *`. SEV-2.
- **S4** efficiency static — no WHERE, LIMIT or QUALIFY. Skip tables whose names
  contain `_lookup`/`_config`/`_ref`/`_dim`/`_codes` — reference tables are
  legitimately unbounded. SEV-2.
- **S7** efficiency static — CTEs + joins + windows + subqueries over threshold
  (default 6). SEV-3.
- **R1** efficiency runtime — `row_count == 0`. Scan other nodes' transform and
  instruction text for `{output_key}[0]`; if any downstream indexes into element
  zero it's SEV-1, else SEV-2.
- **R5** efficiency runtime — `row_count` over threshold (default 1000). SEV-2.

### router.py
- **R-S1** routing static — a `context_key` no node produces. SEV-1.
- **R-S2** routing static — `target_agent` with no YAML (SEV-1), or not listed in
  `sub_agents` (SEV-2).
- **R-S3** routing static — every route uses `eq` and there's no conditionless
  catch-all, so null/unexpected values match nothing. SEV-2. **Do not flag routers
  that use non-eq operators** — a `neq` route is effectively a default.
- **R-S5** routing static — two routes share a priority. SEV-2.
- **R-S7** routing static — a `sub_agents` entry no route targets. SEV-3.
- **R-R1** routing runtime — from `evaluated_routes` (each with a `matched` bool) and
  `chosen_target`: zero matched = SEV-1 `router_no_match`; chosen != lowest-priority-
  number among matched = SEV-1 `router_misdecision`; more than one matched = SEV-3
  `router_multi_match`.

### prompts.py
- **N1a** prompt_quality static — instruction references a key nothing produces and
  that isn't in `input_keys`. SEV-1. Use a prefix heuristic to avoid flagging prose
  words caught by the single-brace regex.
- **N1b** prompt_quality static — instruction over token threshold (default 2000,
  approximate as `len//4`). SEV-3.
- **N1c** prompt_quality static — instruction over ~300 tokens with fewer than 2
  section markers (`##`, "context", "rules", "output", "step", ...). SEV-3.
- **N1d** prompt_quality static — the node's `output_key` is consumed by a transform
  or router (so it gets parsed), but the instruction never mentions json / output
  format / return / schema. SEV-2.
- **N2** prompt_quality static — pairwise 8-gram overlap between LlmAgent
  instructions over threshold (default 0.25 of the smaller set). SEV-3. Report the
  ratio and a sample shared phrase.

### lifecycle.py
- **G-S1** structural static — gate's `input_key` not produced upstream. SEV-1.
- **G-R1** structural runtime — gate span shorter than `gate_min_pause_s`
  (default 0.5) — it isn't gating. SEV-3.
- **G-R2** structural runtime — gate waited past `gate_max_wait_s` (default 86400).
  SEV-2.
- **O-S1** structural static — a consumer appears before its producer in declared
  walk order. SEV-1.
- **O-S3** structural static — pipeline `output_key` nothing produces. SEV-1.
- **Q-S1** structural static — a node with `agent_class == "<missing>"`; report which
  parents reference it. SEV-1.
- **Q-S3** structural static — the same child listed twice in one `sub_agents`. SEV-2.

### judge.py
`LocalJudgeClient(endpoint, model, allowed_hosts, max_calls)`:
- **Raise `JudgeEndpointError` in `__init__`** if the hostname isn't in
  `{localhost, 127.0.0.1, ::1, 0.0.0.0}` and doesn't end in `.internal` or `.local`.
  This is the load-bearing PII guard — it must fail before any prompt is constructed.
- `complete_json()` strips ``` fences and falls back to a `{...}` regex extract.
- Track `calls_used` against `max_calls`; expose `.exhausted`.

- **N5** output_quality runtime, needs_judge — per LlmAgent span, judge input/output
  against a rubric built from the node's `description`, `input_keys`, `output_key`,
  and declared downstream consumers. Score 1–5 on role_fulfillment,
  input_utilization, output_completeness, scope_adherence, plus a closed-enum
  `failure_mode`. Flag when worst axis ≤3 or failure_mode != none. SEV-1 at ≤2,
  SEV-2 at 3. On judge exception emit an INFO `judge_error` flaw and continue.
- **N4** prompt_quality runtime, needs_judge — given the session snapshot's field
  *names and types* (not values) and the rendered prompt, ask whether the prompt
  instructs the model to re-derive something already structured in session. SEV-2.

## Capture (capture/)

`EphemeralTraceStore(mode)` where mode is `memory` (default) or `encrypted_temp`:
- memory: spans in a list, never on disk
- encrypted_temp: also append Fernet-encrypted lines to a `mkstemp` file chmod 0600,
  key generated at init and held **only in memory**
- `cleanup()`: clear every span dict, clear the list, shred the temp file
  (random bytes -> zeros -> fsync -> unlink), zero the key. Idempotent. Returns a
  summary dict.
- Register cleanup on `atexit` AND handlers for SIGINT/SIGTERM/SIGHUP that clean up
  then re-raise with the previous handler. Tolerate being constructed off the main
  thread (signal.signal raises there — catch and rely on atexit).
- `add()` after cleanup must be a no-op.

`attach(adapter, config, evaluate_on_exit=True)`:
1. Create the store.
2. Get the current TracerProvider. If it isn't an SDK provider, install one. **Add**
   a BatchSpanProcessor with an exporter that writes serialized spans into the store
   — do not replace or disturb the existing Arize exporter.
3. `install_enrichment(adapter)` — monkey-patch the adapter's base class run method
   with a wrapper that opens a span, sets pre-attributes, calls the original,
   records exceptions with ERROR status, sets post-attributes. Mark the wrapper with
   `_o2a_wrapped` so it's idempotent. **Every attribute-setting call must be wrapped
   in try/except** — a capture failure must never break the user's pipeline run.
4. Register an atexit hook that force-flushes, builds the Run, loads the graph,
   scores, renders, and writes. Wrap the whole thing so evaluation failure can't
   break process exit.

`adapters/base.py` — `FrameworkAdapter` dataclass with `base_class_path`,
`run_method`, and callables: `get_session(args, kwargs)`, `get_agent_name(agent)`,
`get_agent_class(agent)`, `get_output_key`, `get_input_keys`,
`render_prompt(agent, session)`, `get_routes(agent, session)`,
`get_query_info(agent, session, result)`, plus class-name tuples and a
`capture_full_fidelity` flag (False omits prompts/snapshots for production).
Provide sensible defaults for all of them. Then two functions,
`set_pre_attributes` and `set_post_attributes`, that write the `o2a.*` namespace:

    o2a.agent.name / .class / .output_key / .input_keys
    o2a.session.keys_before / .keys_after / .keys_written
    o2a.pipeline.name / .hash
    o2a.llm.rendered_prompt / .session_snapshot / .template_references   (LlmAgent)
    o2a.router.evaluated_routes / .chosen_target                          (router)
    o2a.db.query / .query_hash / .row_count                               (database)

Truncate large attributes at 64KB.

## PII scrubbing (report/scrubber.py)

**Allow-list, not deny-list.** Allow-lists fail closed: a new evidence key a future
check adds is dropped until explicitly declared safe. A deny-list fails open and
leaks. This is the single most important design decision in the file.

- `SAFE_EVIDENCE_KEYS` — counts, scores, thresholds, durations, and *name-like*
  fields (agent names, session KEY names, check ids). Never value-bearing fields.
- `BLOCKED_EVIDENCE_KEYS` — `input_value`, `output_value`, `rendered_prompt`,
  `session_snapshot`, `query`, `judge_reason`, `reason`, `snippet`, `prompt`,
  `value`, `values`, `row`, `rows`, `data`, `axes`, `sample_shared_phrase`.
  Blocked at every nesting level, not just the top.
- `scrub_evidence()` keeps only allow-listed keys, truncates strings to 200 chars,
  caps list/dict size, limits recursion depth, and records what it dropped under
  `_dropped_fields` so a check author can see their key was rejected.
- `scorecard_to_safe_dict()` and `trace_summary()` (structural span facts: agent,
  class, duration, status, session KEY names, template refs, router target, row
  count — never values).

## Terminal report (report/terminal.py)

Use `rich` when importable, plain text otherwise. Show status, overall, duration,
pipeline hash, judge budget; per-axis bar + score; then flaws ordered by severity
with check_id, type, node, description, fix, and a short evidence preview.

## CLI (cli.py)

- `o2a-eval score <trace.jsonl> --agent-dir --pipeline [--output-dir] [--json]
  [--no-judge] [--latency-budget-s] [--fail-on SEV-1]`
- `o2a-eval lint --agent-dir --pipeline` — static checks only, no trace. Construct an
  empty Run; the rollup's has-trace guard handles the rest.
- `o2a-eval graph --agent-dir --pipeline` — print the parsed tree with a marker per
  kind and the output_key per node, then node counts and pipeline_hash.
- `o2a-eval checks` — list every registered check grouped by stage.

Exit non-zero when max severity >= `--fail-on`, so this can gate CI.

## Tests (tests/test_evaluator.py)

Build fixture YAMLs modelling a realistic pipeline (an orchestrator, two
SequentialAgents, two routers with branch targets, a gate, three database_agents,
several transforms, two LlmAgents whose instructions share a repeated
"NEVER fabricate" block). Then a `make_trace.py` generating a span JSONL that
**deliberately seeds these flaws**, so the checks have something to find:

- one database_agent called twice with identical input        -> D1
- an LlmAgent injecting a key absent from its `input_keys`    -> N3
- both LlmAgent instructions repeating a block                -> N2
- an eq-only router                                            -> R-S3
- a gate completing in ~4ms                                    -> G-R1
- total duration far over a test budget                        -> O-R3

Cover at minimum:
- graph: name-based resolution, embedded db yaml string, route parsing, producer and
  consumer maps, stable pipeline_hash, unknown pipeline raises
- trace: tree reconstruction, typed span properties, router attributes
- each check family: the seeded flaw IS caught, and the specific false positives
  called out above are NOT produced (no `sequential_reorder` from a repeated child,
  no `dead_output` for terminal/router outputs, no `router_not_exhaustive` on a
  `neq` router, no structural drift on a valid run)
- judge gating: judge checks absent without a client; `LocalJudgeClient` raises on
  an external endpoint and accepts localhost and `*.internal`
- rollup: axis keys, score bounds, clean run passes, a check raising produces
  `check_error` without zeroing the score
- **scrubber (load-bearing)**: parametrized test that every blocked key never
  persists; unknown keys dropped and reported; nested blocked keys removed; a full
  scorecard contains no seeded PII string (loan number, borrower name, prompt text,
  table name); trace_summary keeps key NAMES but not values
- **ephemeral store (load-bearing)**: memory purge, idempotent cleanup, add-after-
  cleanup no-op, encrypted temp is unreadable at rest and shredded on cleanup, 0600
  permissions

## Build order

Implement and test in this order; each step should leave the suite green.

1. `core/models.py` + `core/graph.py`, with fixture YAMLs. Verify `o2a-eval graph`
   prints the right tree and `pipeline_hash` is stable.
2. `core/trace.py` + `make_trace.py`. Verify tree reconstruction and span properties.
3. `core/registry.py` + `core/rollup.py` with one trivial check, to prove the loop.
4. `checks/hygiene.py` D1 first — cheapest real signal.
5. `checks/structural.py`, then `router.py`, then `database.py`, then `prompts.py`,
   then `lifecycle.py`. Test each as you go; expect to find false positives and
   calibrate.
6. `report/scrubber.py` with its full test class **before** anything writes to disk.
7. `report/terminal.py` + `cli.py`.
8. `capture/ephemeral.py` with its full test class.
9. `adapters/base.py` + `capture/attach.py`.
10. `checks/judge.py` last — everything else must work without a model.

## Acceptance criteria

- `pytest` green, with the scrubber and ephemeral-store classes passing in full
- `o2a-eval graph` tree matches the fixture YAML structure, no `<missing>` nodes
- `o2a-eval lint` runs with no trace and emits only static flaws
- `o2a-eval score` on the seeded fixture catches every seeded flaw and produces none
  of the listed false positives
- `--json` output contains no seeded PII string
- Package installs via `pip install -e ".[all]"` and exposes an `o2a-eval` entrypoint
- `sqlglot`, `rich`, `cryptography` all optional — the package works without each

Report when done: the check inventory as a table, test count, and any check you
could not implement faithfully with the reason.
````

---

## Notes on running this

Give the agent the real YAML directory if you can — fixtures modelled on your actual
pipeline surface calibration bugs that toy fixtures don't. Two false positives in the
reference implementation (`dead_output` on terminal outputs, `sequential_reorder` from
a duplicated child) only appeared once the fixtures matched real structure.

Expect the agent to need a calibration pass after step 5. That's normal — the first
version of a check is usually too eager.
