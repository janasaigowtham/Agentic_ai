# O2A Pipeline Evaluator — Technical Specification
# Version: 1.0
# Audience: coding agent (Claude Code, Cursor, etc.)
# Purpose: implement the evaluator from this file alone, no other context needed

---

## 1. PROBLEM STATEMENT

O2A pipelines are defined as a tree of YAML files, one per agent, in a flat
directory. They mix deterministic agents (exact contracts, rule-checkable) with
non-deterministic agents (LlmAgent, requires a judge). Deterministic agents
fail silently — a wrong router branch, a null from a missed session key, a
redundant DB call — without any runtime error. LlmAgent failures are semantic —
wrong output, scope drift, re-deriving structured data from text. Neither is
caught today.

The evaluator compares declared intent (YAMLs) against actual behaviour (OTel
spans), scores both, flags flaws with severity and fix suggestions, and writes
a PII-safe scorecard to disk. It runs locally, triggered by running a pipeline
in the terminal. It must never break a pipeline run.

---

## 2. PIPELINE SHAPE (what the evaluator reads)

### 2.1 YAML schema

Every agent lives in its own YAML file. Fields used by the evaluator:

```
name: string                     # matches the filename (stem), used for cross-references
agent_class: string              # see §2.2
version: string                  # optional
description: string              # free text; becomes judge rubric for LlmAgent
output_key: string               # session key this agent writes
input_keys: list[string]         # session keys this agent reads (also singular input_key)
input_key: string                # merged with input_keys on load
strict: bool                     # default true; false = silent null on missing refs
sub_agents:                      # ordered list of child references
  - name: string                 # resolved by name against the directory index
instruction: |                   # LlmAgent prompt template (may be multi-line)
  ...{session_key}...
transform: dict                  # slv_transformation_agent / transformation_agent
  $fn / $cond / $each: ...
routes:                          # decision_router_agent
  - conditions:
      - context_key: string
        operator: eq | neq | not_null | gt | lt | ...
        value: any
    target_agent: string
    priority: int                # lower number = higher priority
    metadata: dict
default_db_yaml: |               # database_agent; may be a nested dict OR an embedded YAML string
  type: postgres | teradata | ...
  connection:
    url: string                  # should be ${ENV:VAR}, not a literal credential
  query: >
    SELECT ... WHERE col = :param_name
model: string                    # LlmAgent model identifier
```

### 2.2 Agent classes

| agent_class | kind | contract |
|---|---|---|
| `LlmAgent` | NON_DETERMINISTIC | runs `instruction` against a model |
| `database_agent` | DETERMINISTIC | executes SQL, writes rows to `output_key` |
| `slv_transformation_agent` | DETERMINISTIC | evaluates `transform` block |
| `transformation_agent` | DETERMINISTIC | same as above |
| `decision_router_agent` | CONTROL_FLOW | evaluates `routes`, hands off to one target |
| `SequentialAgent` | CONTROL_FLOW | runs `sub_agents` in order |
| `agent_gate` | CONTROL_FLOW | pauses until external resume signal |
| `resumable_orchestrator` | CONTROL_FLOW | top-level, supports resume across gates |

Classification rule:
- NON_DETERMINISTIC = {"LlmAgent"}
- CONTROL_FLOW = {"resumable_orchestrator","SequentialAgent","decision_router_agent","agent_gate"}
- DETERMINISTIC = everything else

### 2.3 Session state

All inter-agent communication happens through a session dict. An agent reads from
`input_keys`, may reference keys via templates in transform/instruction/SQL, and
writes to `output_key`. There is no other interface between agents.

Template reference styles:
- `{{ key }}` or `{{ key[0].field | filter }}` (double-brace, in transform/SQL)
- `{key}` (single-brace, in LlmAgent instruction blocks)
- `:param` (bound parameter in SQL)

### 2.4 OTel span attributes (o2a.* namespace)

The evaluator reads these from captured spans. The capture hook (§7) writes them.

```
o2a.agent.name          string   agent's declared name
o2a.agent.class         string   agent_class value
o2a.agent.output_key    string   declared output_key or ""
o2a.agent.input_keys    JSON     list[string]
o2a.session.keys_before JSON     sorted list of session keys at span start
o2a.session.keys_after  JSON     sorted list at span end
o2a.session.keys_written JSON    keys added by this span
o2a.session.keys_read   JSON     keys actually accessed (if instrumented)
o2a.pipeline.name       string
o2a.pipeline.hash       string   sha256[:16] of all YAML hashes combined

# LlmAgent only (local capture, full fidelity):
o2a.llm.rendered_prompt    string   fully substituted prompt (may be 64KB)
o2a.llm.session_snapshot   JSON     full session dict at span start
o2a.llm.template_references JSON    list of {key} referenced in instruction

# decision_router_agent only:
o2a.router.evaluated_routes JSON    list[{target,priority,conditions,matched:bool}]
o2a.router.chosen_target    string

# database_agent only:
o2a.db.query        string   rendered query (local only)
o2a.db.query_hash   string   hash of rendered query
o2a.db.row_count    int
```

---

## 3. DATA MODELS (core/models.py)

### 3.1 Severity

```python
class Severity(IntEnum):
    INFO  = 0   # notable, not a problem
    SEV_3 = 1   # design smell
    SEV_2 = 2   # real problem
    SEV_1 = 3   # blocker

    @property
    def label(self) -> str: ...          # "INFO" | "SEV-3" | "SEV-2" | "SEV-1"
    @classmethod
    def parse(cls, s: str) -> Severity:  # "SEV-1" -> SEV_1 etc.
```

Ordering: INFO < SEV_3 < SEV_2 < SEV_1. IntEnum comparisons must work.

### 3.2 Flaw

```python
@dataclass
class Flaw:
    type: str                    # snake_case identifier e.g. "redundant_execution"
    severity: Severity
    node: str                    # agent name where the flaw was found
    description: str             # human-readable one-liner
    fix: str | None              # suggested remediation
    evidence: dict[str, Any]     # MAY contain PII — scrubbed before persist
    check_id: str = ""           # set by rollup from CheckMeta
    agent_class: str = ""

    def to_dict(self, include_evidence=True) -> dict: ...
```

### 3.3 Node

```python
@dataclass
class Node:
    name: str
    agent_class: str
    source_path: str             # absolute path to YAML file
    yaml_hash: str               # sha256[:16] of the file bytes
    description: str
    instruction: str             # LlmAgent prompt template (raw, unsubstituted)
    output_key: str | None
    input_keys: list[str]        # merged input_keys + input_key
    strict: bool                 # default True
    raw: dict                    # full parsed YAML
    sub_agent_names: list[str]   # from sub_agents[].name
    children: list[Node]         # resolved after directory index is built
    path: tuple[str, ...]        # hierarchical e.g. ("pmi_ddn_pipeline","pmi_ddn_pre_process")
    routes: list[Route]
    query: str | None
    db_type: str | None
    db_url: str | None
    transform: dict | None
    model: str | None

    @property
    def kind(self) -> str:
        # "non_deterministic" | "control_flow" | "deterministic"

    @property
    def is_llm(self) -> bool: ...

    @property
    def qualified(self) -> str:          # ".".join(self.path)

    def walk(self) -> Iterator[Node]:    # yield self then all descendants
```

### 3.4 Route

```python
@dataclass
class Route:
    target_agent: str
    priority: int = 0
    conditions: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def context_keys(self) -> list[str]:   # [c["context_key"] for c in conditions]
```

### 3.5 PipelineGraph

```python
@dataclass
class PipelineGraph:
    name: str
    version: str
    root: Node
    nodes: dict[str, Node]               # name -> Node (all reachable)
    producers: dict[str, str]            # session key -> node name
    consumers: dict[str, list[str]]      # session key -> [node names]
    pipeline_hash: str

    def by_class(self, agent_class: str) -> list[Node]: ...
    def deterministic(self) -> list[Node]: ...
    def non_deterministic(self) -> list[Node]: ...
    def downstream_of(self, node_name: str) -> list[Node]: ...
```

### 3.6 Span

```python
@dataclass
class Span:
    span_id: str
    trace_id: str
    parent_id: str | None
    name: str
    start_ns: int
    end_ns: int
    status: str = "OK"
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[dict] = field(default_factory=list)
    children: list[Span] = field(default_factory=list)

    def attr(self, key, default=None): ...     # raw attribute lookup
    def jattr(self, key, default=None): ...    # JSON-string attribute, tolerates malformed

    @property def duration_s(self) -> float: ...
    @property def agent_name(self) -> str: ...          # o2a.agent.name, fallback to span name
    @property def agent_class(self) -> str: ...
    @property def output_key(self) -> str: ...
    @property def input_keys(self) -> list[str]: ...
    @property def keys_before(self) -> list[str]: ...
    @property def keys_after(self) -> list[str]: ...
    @property def keys_written(self) -> list[str]: ...
    @property def keys_read(self) -> list[str]: ...
    @property def input_value(self) -> Any: ...         # "input.value" attribute
    @property def output_value(self) -> Any: ...
    @property def rendered_prompt(self) -> str: ...
    @property def session_snapshot(self) -> dict: ...
    @property def template_references(self) -> list[str]: ...
    @property def evaluated_routes(self) -> list[dict]: ...
    @property def chosen_target(self) -> str | None: ...
    @property def query_hash(self) -> str | None: ...
    @property def row_count(self) -> int | None: ...
    @property def errored(self) -> bool: ...
```

No check should ever call `span.attributes[...]` directly. All access goes through
the typed properties.

### 3.7 Run

```python
@dataclass
class Run:
    trace_id: str
    pipeline_name: str
    pipeline_hash: str
    spans: list[Span]
    root: Span | None = None
    by_id: dict[str, Span] = field(default_factory=dict)
    by_agent: dict[str, list[Span]] = field(default_factory=dict)

    @property def duration_s(self) -> float: ...
    def spans_of_class(self, agent_class: str) -> list[Span]: ...
    def llm_spans(self) -> list[Span]: ...
    def ordered(self) -> list[Span]: ...           # by start_ns
    def spans_for(self, agent_name: str) -> list[Span]: ...
```

### 3.8 Scorecard

```python
@dataclass
class Scorecard:
    pipeline: str
    trace_id: str
    pipeline_hash: str
    duration_s: float
    axes: dict[str, float]         # one key per axis in §5.1
    overall: float
    flaws: list[Flaw]              # ordered worst-first
    judge_calls_used: int = 0
    judge_calls_budget: int = 0
    span_count: int = 0

    @property def status(self) -> str:        # "PASS" | "FLAGGED" | "FAIL"
    @property def max_severity(self) -> Severity: ...
```

STATUS rule: FAIL if any SEV_1, FLAGGED if any SEV_2 or SEV_3, PASS otherwise.

---

## 4. YAML LOADER (core/graph.py)

### 4.1 Directory indexing

```python
def index_directory(agent_dir: str | Path) -> dict[str, Node]:
```

- Glob `*.yaml` and `*.yml` in the directory (non-recursive).
- Parse each with `yaml.safe_load`. Skip files without a `name:` field.
- Key the result by the `name:` field, not by filename.

### 4.2 Node parsing

```python
def _node_from_spec(spec: dict, path: Path) -> Node:
```

- `input_keys`: merge `input_keys` list and singular `input_key` string; dedupe.
- `default_db_yaml`: may be a nested dict or an embedded YAML string — attempt
  `yaml.safe_load` if it's a string. Extract `query`, `type`, `connection.url`.
- `routes`: parse each entry's `conditions`, `target_agent`, `priority`, `metadata`.
- `sub_agent_names`: from `sub_agents[].name` or bare string entries.
- `yaml_hash`: `sha256(path.read_bytes()).hexdigest()[:16]`

### 4.3 Graph construction

```python
def load_pipeline(agent_dir: str | Path, pipeline_name: str) -> PipelineGraph:
```

1. Call `index_directory`.
2. Look up `pipeline_name` in the index; raise `ValueError` if absent.
3. Walk from the root: for each `sub_agent_name`, look up in the index. If not
   found, create a placeholder `Node(agent_class="<missing>")` and add it to the
   index. Attach as a child. Guard cycles with a seen-set (don't recurse into a
   node we're currently processing — just attach without descending).
4. Assign `node.path = parent_path + (node.name,)` during the walk.
5. Build `producers`: for every reachable node with an `output_key`, map
   `output_key -> node.name`.
6. Build `consumers`: for every reachable node, collect reads from:
   - `input_keys`
   - `extract_template_keys(node.transform)` — all `{{ key }}` and `{key}` refs
   - `extract_template_keys(node.instruction)`
   - route `context_keys`
   Map each key to the consuming node name.
7. `pipeline_hash`: sha256 of all reachable node `yaml_hash` values, sorted and
   joined, hexdigest[:16].

### 4.4 Helpers

```python
def extract_template_keys(blob: Any) -> set[str]:
```

Walk any nested structure (dict, list, string). Apply two regexes:
- `{{ key }}` style: `r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)"` — captures the root
  identifier before any `.`, `[`, or `|`.
- `{key}` style: `r"(?<!\{)\{([A-Za-z_][A-Za-z0-9_]*)\}(?!\})"` — negative
  lookarounds so it doesn't match doubles.
Return the union.

```python
def extract_sql_params(query: str) -> set[str]:
```

Match `r":([A-Za-z_][A-Za-z0-9_]*)"`. Return the set of param names.

---

## 5. CHECK SYSTEM

### 5.1 Registry (core/registry.py)

```python
AXES = ("structural", "efficiency", "routing", "prompt_quality", "output_quality")

@dataclass
class CheckMeta:
    id: str
    name: str
    axis: str            # must be in AXES
    stage: str           # "static" | "runtime" | "cross_trace"
    fn: CheckFn          # (Run, PipelineGraph, dict) -> list[Flaw]
    needs_judge: bool
    description: str     # first line of the function's docstring

def register(check_id, *, axis, stage, needs_judge=False):
    # decorator; validates axis and stage

def all_checks() -> list[CheckMeta]:
    # static first, then runtime, then cross_trace; within each group sort by id
```

### 5.2 Rollup (core/rollup.py)

```python
def build_scorecard(run: Run, graph: PipelineGraph, config: dict) -> Scorecard:
```

**Skip conditions for a check:**
1. `checks_enabled` in config and check_id not in it
2. `checks_disabled` in config and check_id in it
3. `needs_judge=True` and `config.get("_judge")` is None
4. `run.spans` is empty and stage is "runtime" or "cross_trace" (static-lint mode)
5. A structural SEV_1 flaw already exists and the check has `needs_judge=True`
   and axis in {"output_quality", "prompt_quality"} — no point judging semantics of
   a pipeline that didn't execute as declared

**Exception handling:** wrap each check call in try/except; on exception emit a
`Flaw(type="check_error", severity=Severity.INFO, ...)` and continue.

**Scoring:**
```
SEVERITY_PENALTY = {INFO: 0.0, SEV_3: 0.05, SEV_2: 0.15, SEV_1: 0.40}

axis_score = max(0.0, 1.0 - sum(penalty(f) for f in axis_flaws))

DEFAULT_WEIGHTS = {
    "structural": 0.30,
    "output_quality": 0.25,
    "efficiency": 0.20,
    "routing": 0.15,
    "prompt_quality": 0.10,
}
overall = sum(axis_scores[a] * weights[a] for a in AXES) / sum(weights[a] for a in AXES)
```

Weights overridable via `config["axis_weights"]`.

---

## 6. CHECKS

Signature of every check:
```python
def check_name(run: Run, graph: PipelineGraph, config: dict) -> list[Flaw]:
    """First line becomes CheckMeta.description."""
```

### 6.1 structural.py

**D9** — axis=structural, stage=runtime
What: every declared node fired; no undeclared span appeared.
How:
- `declared` = all node names where `agent_class != "<missing>"`
- `executed` = all `span.agent_name` values
- `conditional` = all `target_agent` values across every router's routes, plus
  all descendants of those targets — these are legitimately absent when not routed to
- `missing` = (declared − executed) − conditional → SEV_1
- `unexpected` = executed − declared − {""} → SEV_2
Evidence: missing list, unexpected list.

**D5** — axis=structural, stage=runtime
What: SequentialAgent children all present, in declared order.
How: for each SequentialAgent span, collect `child.agent_name` from `span.children`.
- `missing` = declared children absent from actual list → SEV_1 per missing child
- Order check: **dedupe** the actual list first (a child appearing twice is D1's
  concern, not an ordering fault). Compare the deduped intersection to the declared
  intersection in declared order → SEV_2 if different.
Evidence: declared list, actual list, missing list / declared_order vs actual_order.

**Q-R2** — axis=structural, stage=runtime
What: a child errored but later siblings still ran.
How: for every span with children, iterate children; if child `errored` and there
are subsequent siblings, flag. → SEV_1.
Evidence: parent name, errored child name, sibling names after it.

**D6** — axis=structural, stage=runtime
What: transform's `{{ key }}` refs were absent from session when it ran.
How: for each transform-class node with a transform block, extract refs with
`extract_template_keys(node.transform)`. Discard `"item"` (the `$each` loop var).
For each span of that node, if `span.keys_before` is non-empty, compute
`unresolved = sorted(refs − set(keys_before))`. If non-empty:
- `strict=False` → SEV_1 (silent null)
- `strict=True` → SEV_2
Evidence: unresolved list, available keys, strict value.
Skip if `span.keys_before` is empty (session not captured for this span).

**T-S7** — axis=structural, stage=static
What: `strict=False` audit — silences errors.
How: every transform-class node with `strict=False` → SEV_3.
Evidence: source_path.

### 6.2 hygiene.py

**D1** — axis=efficiency, stage=runtime
What: same agent ran twice with identical input in one trace.
How: for each span (skip CONTROL_FLOW classes), compute `(span.agent_name,
sha256(str(input_value)).hexdigest()[:16])`. Group. Any group size ≥ 2:
- `database_agent` → SEV_2 (real cost)
- others → SEV_3
Evidence: call_count, input_hash, wasted_seconds (sum of durations beyond the first).

**D2** — axis=efficiency, stage=runtime
What: a node's output_key is never consumed.
How: for each executed node with an `output_key`:
- Skip if `agent_class` in CONTROL_FLOW (routing decisions consumed by orchestrator)
- Skip if `output_key == graph.root.output_key` (pipeline terminal output)
- Skip if node is a direct child of root and last in root.sub_agent_names (terminal)
- Skip if declared consumers exist (any entry in `graph.consumers[output_key]`
  other than the node itself)
Then confirm at runtime: if no other span's `keys_read` or `template_references`
contains the key → SEV_3.
Evidence: output_key, downstream_checked count.

**N3** — axis=prompt_quality, stage=runtime
What: LlmAgent prompt injects a session key not in its `input_keys`.
How: for each LlmAgent span:
- `used` = `set(span.template_references) | set(span.keys_read)`
- `declared` = `set(node.input_keys)` for the matching node
- `undeclared` = sorted keys in `(used − declared)` that exist in `graph.producers`
- **Compare against `input_keys`, not the consumers map.** The consumers map is
  built from instruction refs, so anything in the instruction is already in
  consumers — comparing against it can never flag anything. This is a critical
  correctness constraint.
For each undeclared key, note which other LLM spans also read it → SEV_2.
Evidence: session_key, produced_by, declared_consumers, actual_llm_readers, undeclared.

**O-R3** — axis=efficiency, stage=runtime
What: pipeline duration exceeds configured latency budget.
How: if `config.get("latency_budget_s")` is set and `run.duration_s` exceeds it:
Top 3 leaf spans (no children) by duration → SEV_1.
Evidence: duration_s, budget_s, top_contributors list with agent/class/duration_s/pct_of_total.

**O-R4** — axis=efficiency, stage=runtime
What: session key count at pipeline end exceeds threshold (default 25).
How: final span's `keys_after` (fall back to `keys_before`) length.
→ SEV_3 if exceeded.
Evidence: final_key_count, threshold, keys list.

### 6.3 database.py

Use `sqlglot` when importable; fall back to regex when not. Never raise because
sqlglot is absent.

Reference table hint strings: `("_lookup", "_config", "_ref", "_dim", "_codes")`.

**S1** — axis=structural, stage=static
What: SQL fails to parse.
Condition: `sqlglot.parse_one(query, dialect=dialect)` raises.
→ SEV_1. Evidence: error string[:300], dialect.
Skip entire check when sqlglot is not importable.

**S2** — axis=structural, stage=static
What: `:params` vs `input_keys` mismatch.
How: `bound = extract_sql_params(node.query)`, `declared = set(node.input_keys)`.
- `bound − declared` → SEV_1 `query_param_undeclared`
- `declared − bound − extract_template_keys(node.query)` → SEV_3 `query_input_key_unused`
Evidence: bound_params, declared_input_keys, undeclared/unused lists.

**S5** — axis=structural, stage=static
What: `{{ }}` interpolated into SQL (injection risk).
How: `extract_template_keys(node.query)` non-empty → SEV_1.
Evidence: interpolated_refs list.

**S6** — axis=structural, stage=static
What: connection URL inlined (not `${ENV:...}`).
How: `node.db_url` exists and does not contain `"${ENV:"` and does not contain `"${"`.
→ SEV_1. Evidence: has_env_ref=False.

**S8** — axis=structural, stage=static
What: database_agent issues a mutating statement.
How: `re.search(r"\b(DELETE|UPDATE|INSERT|TRUNCATE|DROP|ALTER)\b", query, re.I)`.
→ SEV_1. Evidence: statement keyword.

**S3** — axis=efficiency, stage=static
What: `SELECT *`.
How: `re.search(r"SELECT\s+\*", query, re.I)` → SEV_2.
Evidence: output_key.

**S4** — axis=efficiency, stage=static, requires sqlglot
What: no WHERE / LIMIT / QUALIFY against a likely fact table.
How: parse, find all `exp.Table` nodes; skip if any table name contains a reference
table hint; skip if the AST contains `exp.Where`, `exp.Limit`, or `exp.Qualify`.
→ SEV_2. Evidence: tables list.

**S7** — axis=efficiency, stage=static, requires sqlglot
What: high structural complexity.
How: `score = len(CTEs) + len(Joins) + len(Windows) + len(Subqueries)`.
Threshold: `config.get("query_complexity_threshold", 6)`.
→ SEV_3 if exceeded. Evidence: score, ctes, joins, windows, subqueries, threshold.

**R1** — axis=efficiency, stage=runtime
What: query returned 0 rows while a downstream node indexes `[0]`.
How: for each database_agent span where `span.row_count == 0`:
Scan other nodes' transform text and instruction for `f"{output_key}[0]"`. If any
match → SEV_1 else → SEV_2.
Evidence: output_key, downstream_indexers list, row_count.

**R5** — axis=efficiency, stage=runtime
What: result set over row threshold.
Threshold: `config.get("db_row_count_threshold", 1000)`.
→ SEV_2. Evidence: row_count, threshold.

### 6.4 router.py

**R-S1** — axis=routing, stage=static
What: a `context_key` no agent produces.
How: for each router's routes, collect all `context_keys`; any not in
`graph.producers` → SEV_1.
Evidence: missing_keys, known_producers.

**R-S2** — axis=routing, stage=static
What: `target_agent` missing or not in `sub_agents`.
How:
- `graph.nodes.get(target)` is None or agent_class=="<missing>" → SEV_1
- target not in `node.sub_agent_names` (when sub_agents is non-empty) → SEV_2
Evidence: target, priority, sub_agents list.

**R-S3** — axis=routing, stage=static
What: eq-only routes leave null/unexpected values unmatched.
How: collect operators across all route conditions. If every operator is `"eq"` AND
there is no route with empty `conditions` (catch-all) → flag → SEV_2.
**Do not flag routers that use any non-eq operator** (neq, not_null, gt, lt, etc.).
A neq route is effectively a default. This exclusion is critical.
Evidence: context_keys, covered_values (the eq values enumerated).

**R-S5** — axis=routing, stage=static
What: two routes share a priority.
How: `Counter(r.priority for r in node.routes)`. Any count > 1 → SEV_2.
Evidence: duplicate_priorities dict, routes list.

**R-S7** — axis=routing, stage=static
What: a `sub_agents` entry no route targets.
How: `set(node.sub_agent_names) − {r.target_agent for r in node.routes}` → SEV_3.
Evidence: orphans list, routed_targets list.

**R-R1** — axis=routing, stage=runtime
What: router chose the wrong branch, or no branch matched.
How: from `span.evaluated_routes` and `span.chosen_target`:
- zero routes matched → SEV_1 `router_no_match`
- more than one matched → check if chosen is the one with the lowest priority number
  among matched; if not → SEV_1 `router_misdecision`; if yes → SEV_3 `router_multi_match`
- exactly one matched and chosen != its target → SEV_1 `router_misdecision`
Evidence: chosen, expected, matched_routes, evaluated_routes.

### 6.5 prompts.py

Token count approximation throughout: `max(1, len(text) // 4)`.

Section hint strings: `("##", "###", "context", "instruction", "rules", "output",
"format", "step", "input data", "evaluation", "critical")`.

**N1a** — axis=prompt_quality, stage=static
What: instruction references a session key nothing produces.
How: `extract_template_keys(node.instruction)` minus `graph.producers` minus
`node.input_keys`. Apply prefix heuristic to avoid prose false positives: only flag
keys that start with a known pipeline prefix OR that appear in `node.raw.get("input_keys", [])`.
Recommended heuristic: only flag if the key contains `"_"` (snake_case suggests a
session key, not a prose word caught by the single-brace regex) → SEV_1.
Evidence: missing_refs list, known_producers list.

**N1b** — axis=prompt_quality, stage=static
What: instruction over token threshold.
Threshold: `config.get("prompt_token_threshold", 2000)`.
→ SEV_3. Evidence: approx_tokens, threshold, char_count.

**N1c** — axis=prompt_quality, stage=static
What: long instruction with no section structure.
Condition: `approx_tokens >= 300` AND fewer than 2 section hint matches.
→ SEV_3. Evidence: section_hits, approx_tokens.

**N1d** — axis=prompt_quality, stage=static
What: output parsed downstream but no output format stated.
How: `output_key` consumers include any `transformation_agent` or
`decision_router_agent`. Instruction must mention at least one of
`("json", "output format", "return", "respond with", "schema")` (case-insensitive).
→ SEV_2. Evidence: output_key, parsing_consumers list.

**N2** — axis=prompt_quality, stage=static
What: repeated instruction block across LlmAgent prompts.
How: pairwise 8-gram overlap.
```
def ngrams(tokens, n=8): return {tuple(tokens[i:i+n]) for i in range(max(0,len(tokens)-n+1))}
tokens = re.findall(r"\w+", text.lower())
overlap_ratio = len(ngrams_a & ngrams_b) / min(len(ngrams_a), len(ngrams_b))
```
Threshold: `config.get("instruction_overlap_threshold", 0.25)`.
Skip pairs where either instruction has < 50 tokens.
→ SEV_3. Evidence: node_a, node_b, overlap_ratio, shared_ngrams count.

### 6.6 lifecycle.py

**G-S1** — axis=structural, stage=static
What: gate's `input_key` not produced upstream.
How: `[k for k in node.input_keys if k not in graph.producers]` → SEV_1.
Evidence: missing list.

**G-R1** — axis=structural, stage=runtime
What: gate never paused.
Threshold: `config.get("gate_min_pause_s", 0.5)`.
→ SEV_3. Evidence: duration_s, threshold_s.

**G-R2** — axis=structural, stage=runtime
What: gate waited past ceiling.
Threshold: `config.get("gate_max_wait_s", 86400)`.
→ SEV_2. Evidence: wait_s, ceiling_s.

**O-S1** — axis=structural, stage=static
What: consumer appears before producer in declared walk order.
How: build walk-order list from `graph.root.walk()`. For every `(key, producer)` in
`graph.producers`, for every consumer of that key, check
`position[consumer] < position[producer]` → SEV_1.
Evidence: session_key, producer, consumer_position, producer_position.

**O-S3** — axis=structural, stage=static
What: pipeline `output_key` nothing produces.
Condition: `graph.root.output_key` exists and not in `graph.producers` → SEV_1.
Evidence: declared_output_key, available_keys list.

**Q-S1** — axis=structural, stage=static
What: `sub_agents` references a name with no YAML file.
How: any node with `agent_class == "<missing>"`.
Evidence: referenced_by list (parents that name this node).
→ SEV_1.

**Q-S3** — axis=structural, stage=static
What: same child listed twice in one `sub_agents`.
How: `Counter(node.sub_agent_names)`, any count > 1 → SEV_2.
Evidence: duplicates list, sub_agents list.

### 6.7 judge.py

**LocalJudgeClient**

```python
class JudgeEndpointError(RuntimeError): pass

class LocalJudgeClient:
    ALLOWED_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}

    def __init__(self, endpoint, model, allowed_hosts=None, timeout=120, max_calls=10):
        # CRITICAL: check the hostname at construction time.
        # If it's not local, raise JudgeEndpointError BEFORE building any prompt.
        # Allowed: anything in ALLOWED_HOSTS, anything ending in ".internal" or ".local".
        # Do not catch this exception — it must propagate to the caller.
        ...

    @property
    def exhausted(self) -> bool: return self.calls_used >= self.max_calls

    def complete(self, prompt: str) -> str:
        # POST to self.endpoint with {"model", "messages": [{"role":"user","content":prompt}],
        # "temperature": 0}. Use urllib.request (no external deps).
        # Increment calls_used. Raise RuntimeError if exhausted before the call.
        ...

    def complete_json(self, prompt: str) -> dict:
        # Call complete(), strip ```json ... ``` fences, parse JSON.
        # Fallback: regex extract first {...} if full parse fails.
        ...
```

**N5** — axis=output_quality, stage=runtime, needs_judge=True

Rubric prompt template:
```
Node: {node}
Declared role: {role}
Declared inputs: {inputs}
Declared output key: {output_key}
Downstream consumers: {downstream}

Input received:
{input_value}

Output produced:
{output_value}

Score 1-5 on each axis with one sentence reason:
1. role_fulfillment: did output fulfil the declared role?
2. input_utilization: did it use declared inputs or re-derive them?
3. output_completeness: is output well-formed for downstream consumers?
4. scope_adherence: did it stay within declared role?

Also return failure_mode, one of:
none | wrong_role | incomplete_output | ignored_input | scope_leak | hallucinated_output | other

Return strict JSON only:
{"role_fulfillment":{"score":int,"reason":str},"input_utilization":{"score":int,"reason":str},
"output_completeness":{"score":int,"reason":str},"scope_adherence":{"score":int,"reason":str},
"failure_mode":str}
```

Truncate `input_value` and `output_value` to 4000 chars each. Stop if `judge.exhausted`.
On judge exception → INFO `judge_error` flaw, continue.
Flag if worst axis score ≤ 3 OR failure_mode not in {"none", None}.
- score ≤ 2 → SEV_1
- score == 3 → SEV_2
Evidence: axes dict, failure_mode, mean_score.

**N4** — axis=prompt_quality, stage=runtime, needs_judge=True

Describe session fields without exposing values:
```python
for key, val in snapshot.items():
    if isinstance(val, dict):
        desc = f"{key}: object with keys {sorted(val)[:15]}"
    elif isinstance(val, list) and val and isinstance(val[0], dict):
        desc = f"{key}: list[{len(val)}] of objects with keys {sorted(val[0])[:15]}"
    else:
        desc = f"{key}: {type(val).__name__}"
```

Rubric prompt:
```
Session already contains these structured fields:
{field_descriptions}

Prompt sent to model:
{rendered_prompt[:6000]}

Does the prompt instruct the model to extract, parse, derive or infer any value
that already exists as a structured field above?

Return strict JSON only:
{"bypasses": bool, "fields": [str], "reason": str}
```

Flag if `bypasses == True` and `fields` is non-empty → SEV_2.
Evidence: re_derived_fields, available_field_names (keys only, not values), judge_reason (truncated).

---

## 7. CAPTURE (capture/)

### 7.1 EphemeralTraceStore (capture/ephemeral.py)

```python
class EphemeralTraceStore:
    def __init__(self, mode: str = "memory", temp_dir: str | None = None):
        # mode: "memory" | "encrypted_temp"
        # encrypted_temp requires the `cryptography` package
        ...

    def add(self, span: dict[str, Any]) -> None:
        # memory: append to self.spans list
        # encrypted_temp: also Fernet-encrypt and append to temp file
        # no-op after cleanup()

    def read_all(self) -> list[dict]: ...
    def __len__(self) -> int: ...

    def cleanup(self) -> dict:
        # 1. Clear every span dict in self.spans, then clear the list
        # 2. If temp file: overwrite with random bytes, then zeros, fsync, unlink
        # 3. Zero the Fernet key bytes
        # 4. Set _cleaned = True
        # 5. Idempotent: if already cleaned, return {"already_clean": True}
        # Returns: {"spans_purged": int, "temp_shredded": bool, "mode": str}
```

**Cleanup registration (in `__init__`):**
```python
atexit.register(self.cleanup)
for signame in ("SIGINT", "SIGTERM", "SIGHUP"):
    sig = getattr(signal, signame, None)
    if sig is None: continue
    try:
        prev = signal.signal(sig, self._signal_cleanup)
        self._prev_handlers[sig] = prev
    except (ValueError, OSError):
        pass  # off main thread — atexit covers normal exit
```

`_signal_cleanup(signum, frame)`: call cleanup(), restore previous handler, re-raise.

**encrypted_temp specifics:**
- `fd, path = tempfile.mkstemp(prefix="o2a_", suffix=".enc", dir=temp_dir)`
- `os.chmod(path, 0o600)`
- Encryption: `Fernet.generate_key()` at init; cipher = `Fernet(key)`.
- Each span: `cipher.encrypt(json.dumps(span, default=str).encode())` + `\n`.
- Data unreadable at rest even if the file survives a SIGKILL (key died with process).

**Shred algorithm:**
```python
size = path.stat().st_size
with open(path, "r+b") as f:
    f.write(secrets.token_bytes(size)); f.flush(); os.fsync(f.fileno())
    f.seek(0)
    f.write(b"\x00" * size); f.flush(); os.fsync(f.fileno())
path.unlink()
```

### 7.2 OTel exporter

```python
class EphemeralExporter(SpanExporter):
    def __init__(self, store: EphemeralTraceStore): ...
    def export(self, spans) -> SpanExportResult:
        for s in spans:
            store.add(_serialize(s))
        return SpanExportResult.SUCCESS
    def shutdown(self): pass   # cleanup owned by the store, not the exporter
    def force_flush(self, timeout_millis=30000) -> bool: return True
```

Serialization:
```python
def _serialize(s) -> dict:
    ctx = s.get_span_context()
    return {
        "trace_id": format(ctx.trace_id, "032x"),
        "span_id": format(ctx.span_id, "016x"),
        "parent_id": format(s.parent.span_id, "016x") if s.parent else None,
        "name": s.name,
        "start_ns": s.start_time,
        "end_ns": s.end_time,
        "status": s.status.status_code.name if s.status else "UNSET",
        "attributes": dict(s.attributes or {}),
        "events": [{"name":e.name,"attributes":dict(e.attributes or {}),"ts":e.timestamp}
                   for e in (s.events or [])],
    }
```

### 7.3 Enrichment hook (capture/attach.py)

```python
def install_enrichment(adapter: FrameworkAdapter) -> None:
    module, cls_name, base_cls = _import_class(adapter.base_class_path)
    original = getattr(base_cls, adapter.run_method)
    if getattr(original, "_o2a_wrapped", False): return  # idempotent

    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        session = {}
        try:
            session = adapter.get_session(args, kwargs) or {}
        except Exception:
            pass
        keys_before = set(session.keys())
        name = adapter.get_agent_name(self)

        with tracer.start_as_current_span(name) as span:
            try: set_pre_attributes(span, self, session, adapter)
            except Exception: pass
            try:
                result = original(self, *args, **kwargs)
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise
            try: set_post_attributes(span, self, session, keys_before, result, adapter)
            except Exception: pass
            return result

    wrapped._o2a_wrapped = True
    setattr(base_cls, adapter.run_method, wrapped)
```

**CRITICAL:** every `set_pre_attributes` and `set_post_attributes` call is wrapped
in try/except. A capture failure must never raise into the user's code.

### 7.4 attach()

```python
_STATE: dict = {"attached": False}

def attach(adapter, config=None, evaluate_on_exit=True) -> EphemeralTraceStore:
    if _STATE["attached"]: return _STATE["store"]
    config = dict(config or {})
    store = EphemeralTraceStore(mode=config.get("storage_mode","memory"))

    provider = trace.get_tracer_provider()
    if not isinstance(provider, TracerProvider):
        provider = TracerProvider()
        trace.set_tracer_provider(provider)
    provider.add_span_processor(BatchSpanProcessor(EphemeralExporter(store)))

    install_enrichment(adapter)
    _STATE.update(attached=True, store=store, config=config)
    if evaluate_on_exit:
        atexit.register(_on_exit)
    return store

def _on_exit():
    try:
        trace.get_tracer_provider().force_flush()
        store = _STATE["store"]
        if not len(store): return
        config = _STATE["config"]
        run = build_run(store.read_all())
        graph = load_pipeline(config["agent_dir"], config.get("pipeline") or run.pipeline_name)
        card = build_scorecard(run, graph, config)
        render_scorecard(card)
        if config.get("output_dir"):
            write_scorecard(card, run, config["output_dir"])
    except Exception as e:
        print(f"[o2a-eval] evaluation skipped: {type(e).__name__}: {e}")
```

### 7.5 FrameworkAdapter (adapters/base.py)

```python
MAX_ATTR_BYTES = 65536

@dataclass
class FrameworkAdapter:
    base_class_path: str         # "mypackage.agents.BaseAgent"
    run_method: str = "run"

    # All callables have safe defaults; override for your framework
    get_session: Callable = field(default=lambda args, kwargs:
        getattr(args[0], "session", {}) if args else {})
    get_agent_name: Callable = field(default=lambda agent:
        getattr(agent, "name", type(agent).__name__))
    get_agent_class: Callable = field(default=lambda agent:
        type(agent).__name__)
    get_output_key: Callable = field(default=lambda agent:
        getattr(agent, "output_key", None))
    get_input_keys: Callable = field(default=lambda agent:
        list(getattr(agent, "input_keys", []) or []))
    render_prompt: Callable = field(default=lambda agent, session: None)
    get_routes: Callable = field(default=lambda agent, session: [])
    get_query_info: Callable = field(default=lambda agent, session, result: {})

    llm_class_names: tuple = ("LlmAgent",)
    router_class_names: tuple = ("decision_router_agent",)
    db_class_names: tuple = ("database_agent",)
    pipeline_name: str = ""
    pipeline_hash: str = ""
    capture_full_fidelity: bool = True   # False = omit prompts/snapshots

def set_pre_attributes(span, agent, session, adapter) -> None: ...
def set_post_attributes(span, agent, session, keys_before, result, adapter) -> None: ...
```

`set_pre_attributes` writes:
- Always: `o2a.agent.*`, `o2a.session.keys_before`, `o2a.pipeline.*`
- LlmAgent + `capture_full_fidelity`: `o2a.llm.rendered_prompt` (truncated 64KB),
  `o2a.llm.session_snapshot`, `o2a.llm.template_references` (from agent if available)
- Router: `o2a.router.evaluated_routes`

`set_post_attributes` writes:
- Always: `o2a.session.keys_after`, `o2a.session.keys_written`
- `output.value` if `capture_full_fidelity`
- Router: `o2a.router.chosen_target`
- DB: `o2a.db.query_hash`, `o2a.db.row_count`, `o2a.db.query` (if full fidelity)

---

## 8. PII MODEL (report/scrubber.py)

### 8.1 Design principle

**Allow-list, not deny-list.** A new evidence key a check author adds is silently
dropped until it appears in `SAFE_EVIDENCE_KEYS`. A deny-list fails open (a new key
passes through until noticed). An allow-list fails closed (a new key is blocked until
explicitly declared safe). This asymmetry is the load-bearing safety property.

### 8.2 Allow-list

```python
SAFE_EVIDENCE_KEYS = {
    # Counts, scores, durations, thresholds — never values
    "call_count", "wasted_seconds", "duration_s", "budget_s", "row_count",
    "threshold", "approx_tokens", "char_count", "section_hits", "overlap_ratio",
    "shared_ngrams", "score", "ctes", "joins", "windows", "subqueries",
    "final_key_count", "downstream_checked", "declared_count", "wait_s",
    "ceiling_s", "threshold_s", "consumer_position", "producer_position",
    "pct_of_total", "mean_score", "db_row_count_threshold", "sample_size",
    "fire_rate", "spans_purged", "temp_shredded", "already_clean",
    # Names — agent names, session KEY names (never key values)
    "missing", "unexpected", "declared", "actual", "declared_order",
    "actual_order", "unresolved", "available", "session_key", "produced_by",
    "producer_class", "declared_consumers", "actual_llm_readers", "undeclared",
    "output_key", "keys", "missing_keys", "known_producers", "target",
    "sub_agents", "orphans", "routed_targets", "context_keys", "covered_values",
    "duplicate_priorities", "routes", "matched", "chosen", "expected",
    "matched_routes", "session_keys", "node_a", "node_b", "missing_refs",
    "parsing_consumers", "referenced_by", "duplicates", "bound_params",
    "declared_input_keys", "unused", "tables", "interpolated_refs",
    "downstream_indexers", "re_derived_fields", "available_field_names",
    "failure_mode", "top_contributors", "agent", "class",
    "declared_output_key", "available_keys", "input_keys", "strict",
    "dialect", "statement", "has_env_ref", "mode", "priority", "error",
}
```

### 8.3 Block-list (belt-and-suspenders for deeply nested evidence)

```python
BLOCKED_EVIDENCE_KEYS = {
    "input_value", "output_value", "rendered_prompt", "session_snapshot",
    "query", "sample_shared_phrase", "judge_reason", "reason", "axes",
    "snippet", "prompt", "value", "values", "row", "rows", "data",
}
```

### 8.4 scrub_evidence()

```python
def scrub_evidence(evidence: dict) -> dict:
    out = {}
    dropped = []
    for k, v in (evidence or {}).items():
        if k in BLOCKED_EVIDENCE_KEYS or k not in SAFE_EVIDENCE_KEYS:
            dropped.append(k)
            continue
        out[k] = _scrub_value(v)
    if dropped:
        out["_dropped_fields"] = sorted(dropped)
    return out

def _scrub_value(v, depth=0):
    if depth > 4: return "<nested>"
    if isinstance(v, (int, float, bool)) or v is None: return v
    if isinstance(v, str): return v[:200]
    if isinstance(v, (list, tuple)):
        return [_scrub_value(x, depth+1) for x in v[:50]]
    if isinstance(v, dict):
        return {str(k)[:80]: _scrub_value(val, depth+1)
                for k, val in list(v.items())[:50]
                if k not in BLOCKED_EVIDENCE_KEYS}
    return str(v)[:200]
```

### 8.5 scorecard_to_safe_dict()

Structural fields only: pipeline, trace_id, pipeline_hash, duration_s, span_count,
status, overall, axes, judge calls, and scrubbed flaws (check_id, type, severity
label, node, agent_class, description[:500], fix[:500], scrubbed evidence).

### 8.6 trace_summary()

Per-span: span_id, parent_id, agent name, agent_class, duration_s, status,
output_key, session_keys_before (KEY NAMES only), session_keys_written, template_references,
router_chosen_target, row_count. Never: rendered prompt, session snapshot, input/output
values, DB query text, any value from the session dict.

---

## 9. REPORTING

### 9.1 Terminal (report/terminal.py)

Import `rich` optionally; fall back to plain text if not installed.

Rich output order:
1. `console.rule(f"o2a-eval · {pipeline} · {trace_id[:12]}")`
2. Panel: status (color-coded), overall score, duration, span count, pipeline hash,
   judge calls (if budget > 0)
3. Per-axis table: name, bar chart `(█ * int(score*10))░...`, score as float
4. Flaws section: for each flaw in order, SEV label (colored), check_id, type, node,
   description, fix, evidence preview (3 key=value pairs, truncated to 160 chars)

Status colors: PASS=green, FLAGGED=yellow, FAIL=red.
Severity colors: SEV_1=bold red, SEV_2=yellow, SEV_3=cyan, INFO=dim.

### 9.2 Writers (report/scrubber.py)

```python
def write_scorecard(card: Scorecard, run: Run, output_dir: str | Path) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    sc = out / "scorecard.json"
    sc.write_text(json.dumps(scorecard_to_safe_dict(card), indent=2))
    (out / "trace.summary.json").write_text(json.dumps(trace_summary(run), indent=2))
    return sc
```

---

## 10. CLI (cli.py)

Entry point: `o2a-eval = "o2a_eval.cli:main"`.

### Subcommands

**score**
```
o2a-eval score <trace.jsonl>
  --agent-dir DIR        required
  --pipeline NAME        optional; defaults to run.pipeline_name
  --config PATH          yaml or json config file
  --output-dir DIR       write scorecard.json + trace.summary.json
  --json                 print scorecard as JSON instead of rich output
  --no-judge             skip judge-based checks
  --latency-budget-s N   override latency threshold
  --fail-on LEVEL        INFO|SEV-3|SEV-2|SEV-1 (default SEV-1)
```
Exit code: 0 if max_severity < fail_on, 1 otherwise.

**lint** (static only — no trace)
```
o2a-eval lint
  --agent-dir DIR   required
  --pipeline NAME   required
  --config PATH
  --fail-on LEVEL
```
Construct `Run(trace_id="static", pipeline_name=..., pipeline_hash=..., spans=[])`.
The rollup's has-trace guard skips all runtime checks automatically.

**graph**
```
o2a-eval graph
  --agent-dir DIR   required
  --pipeline NAME   required
```
Print the resolved tree recursively. Prefix each node with a kind marker:
`◆` non-deterministic, `▸` control flow, `·` deterministic. Append `→ output_key`
if present. Indentation = 2 spaces per depth level. After the tree: node counts
(total, deterministic, non-deterministic) and pipeline_hash.

**checks**
No arguments. Print all registered checks grouped by stage (STATIC / RUNTIME /
CROSS_TRACE). Per check: id, axis (padded), description, "[judge]" marker if needed.
Final line: "N checks registered".

---

## 11. PACKAGE

### 11.1 pyproject.toml

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "o2a-eval"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = ["pyyaml>=6.0"]

[project.optional-dependencies]
sql    = ["sqlglot>=20.0"]
rich   = ["rich>=13.0"]
crypto = ["cryptography>=41.0"]
otel   = ["opentelemetry-sdk>=1.20","opentelemetry-api>=1.20"]
all    = ["sqlglot>=20.0","rich>=13.0","cryptography>=41.0",
          "opentelemetry-sdk>=1.20","opentelemetry-api>=1.20"]
dev    = ["pytest>=7.0"]

[project.scripts]
o2a-eval = "o2a_eval.cli:main"

[tool.setuptools.packages.find]
include = ["o2a_eval*"]
```

### 11.2 Module init exports

`o2a_eval/__init__.py` must export:
`attach`, `FrameworkAdapter`, `load_pipeline`, `build_run`, `load_jsonl`,
`build_scorecard`, `Scorecard`, `Flaw`, `Severity`, `Run`, `PipelineGraph`.

`o2a_eval/checks/__init__.py` must import all check modules so decorators fire:
`from . import structural, hygiene, database, router, prompts, lifecycle, judge`.

---

## 12. TESTS

### 12.1 Fixture YAMLs (tests/fixtures/agents/*.yaml)

Model a realistic pipeline with at minimum:
- A `resumable_orchestrator` root with 4 sub_agents
- A `SequentialAgent` pre-process with 6+ children
- Two `database_agent` nodes (different dialects if possible)
- At least 3 transform nodes (two `slv_transformation_agent`, one `transformation_agent`)
- Two `decision_router_agent` nodes — one using only `eq` operators, one using `neq`
- One `agent_gate`
- A `SequentialAgent` post-process with two `LlmAgent` nodes whose instructions
  share a repeated block (e.g., "ABSOLUTE RULE: NEVER fabricate...")

### 12.2 Fixture trace (tests/fixtures/make_trace.py)

Executable script: `python make_trace.py <output.jsonl>`.

Deliberately seeds these flaws:
```
D1   — one database_agent runs twice with identical input_value
N3   — pmi_ddn_verdict_synthesizer references pmi_ddn_msp_loan_data
       but its input_keys only declares pmi_ddn_answer_q25527
N2   — both LlmAgent instructions contain a shared "ABSOLUTE RULE" block
R-S3 — at least one router uses only eq operators
G-R1 — agent_gate span has duration < 0.01s
O-R3 — total trace duration = 101s; test uses budget of 60s
```

Each span must have:
- `o2a.agent.name`, `o2a.agent.class`, `o2a.session.keys_before` attributes
- LlmAgent spans must have `o2a.llm.template_references` and `o2a.llm.session_snapshot`
- Router spans must have `o2a.router.evaluated_routes` and `o2a.router.chosen_target`
- DB spans must have `o2a.db.query_hash` and `o2a.db.row_count`
- `parent_id` set correctly so tree reconstruction works

### 12.3 Test cases

**Graph (TestGraph)**
- Name-based multi-file resolution: all nodes present, no `<missing>`
- Non-deterministic classification: only LlmAgent nodes
- Producer map correctness (specific key → specific producer)
- Consumer map includes instruction refs (qa agent consumes msp_loan_data because
  its instruction block references `{pmi_ddn_msp_loan_data}`)
- DB YAML parsed from embedded string: db_type, query content, connection url
- Route parsing: count, target_agents, context_keys
- Pipeline hash is stable (two loads of same dir give same hash)
- Unknown pipeline raises ValueError

**Extractors (TestExtractors)**
- `{{ key }}` extraction from string
- `{{ key[0].field }}` → extracts only the root key
- `{single_brace}` extraction
- Nested dict/list structure
- `:param` SQL extraction

**Trace (TestTrace)**
- Tree reconstruction: root has expected number of direct children
- Span typed properties: agent_class, output_key, template_references,
  session_snapshot (value access), chosen_target, evaluated_routes matched count
- Run duration matches fixture

**Deterministic checks (TestDeterministicChecks)**
- D1: database_agent duplicate → flagged; control-flow not flagged
- D2: terminal output NOT flagged; router output NOT flagged; genuinely unused key IS flagged
- D5: no `sequential_reorder` when a child merely ran twice (dedupe must work)
- G-R1: gate no-op flagged
- O-R3: latency breach flagged; evidence names top contributor correctly

**Router checks (TestRouterChecks)**
- R-S3: eq-only router flagged; neq router NOT flagged
- R-R1: no misdecision on a correctly-routed trace
- R-S2: no false positives on valid targets

**Prompt checks (TestPromptChecks)**
- N3: seeded undeclared injection flagged with correct node and key
- N3: qa_q25527 NOT flagged (msp_loan_data IS in its input_keys)
- N2: shared block flagged; overlap_ratio > threshold
- N1a: no false positives on valid instruction refs

**Database checks (TestDatabaseChecks)**
- S5: no injection flag when using `:param` binding
- S3: no SELECT * flag when columns are named
- S6: no credentials flag when using `${ENV:...}`
- Injection flag when `{{ key }}` appears in SQL

**Judge gating (TestJudgeGating)**
- N4/N5 absent from flaws when `_judge` is not in config
- `LocalJudgeClient("https://api.openai.com/...")` raises `JudgeEndpointError`
- `LocalJudgeClient("http://localhost:8080/v1/chat/completions", "m")` succeeds
- `LocalJudgeClient("http://llm.corp.internal/...", "m")` succeeds

**Rollup (TestRollup)**
- Scorecard has exactly the 5 expected axis keys
- `0.0 <= overall <= 1.0`
- Status == "FAIL" when any SEV_1 exists
- Empty run → `max_severity < SEV_1` (static-lint mode, runtime checks suppressed)
- A check raising an exception → `check_error` INFO flaw, overall > 0

**Scrubber — load-bearing safety tests (TestScrubber)**

These tests are the primary safety guarantee. They must pass before anything
writes to disk.

```python
@pytest.mark.parametrize("key", [
    "input_value", "output_value", "rendered_prompt", "session_snapshot",
    "query", "judge_reason", "reason", "data", "rows", "sample_shared_phrase",
])
def test_blocked_keys_never_persist(self, key):
    out = scrub_evidence({key: "loan 1234567890 borrower Jane Doe SSN 123-45-6789"})
    assert key not in out
    assert "1234567890" not in json.dumps(out)

def test_unknown_key_dropped_and_reported(self):
    out = scrub_evidence({"call_count": 2, "new_field": "sensitive data"})
    assert out["call_count"] == 2
    assert "new_field" not in out
    assert "new_field" in out["_dropped_fields"]

def test_nested_blocked_keys_removed(self):
    out = scrub_evidence({"top_contributors": [{"agent": "a", "query": "SELECT ssn"}]})
    assert "ssn" not in json.dumps(out)

def test_full_scorecard_no_pii(self, run, graph):
    card = build_scorecard(run, graph, {"latency_budget_s": 60})
    blob = json.dumps(scorecard_to_safe_dict(card))
    # These strings appear in the fixture trace but must not appear in the scorecard
    for pii in ["1234567890", "Jane Doe", "ABSOLUTE RULE", "SELECT"]:
        assert pii not in blob

def test_trace_summary_key_names_not_values(self, run):
    summary = trace_summary(run)
    blob = json.dumps(summary)
    assert "pmi_ddn_msp_loan_data" in blob   # key NAME is safe
    assert "1234567890" not in blob           # key VALUE is not
    assert "ABSOLUTE RULE" not in blob        # prompt text is not
```

**Ephemeral store — load-bearing safety tests (TestEphemeralStore)**

```python
def test_memory_purge():
    store = EphemeralTraceStore()
    store.add({"span_id": "a", "attributes": {"secret": "loan 1234567890"}})
    assert len(store) == 1
    result = store.cleanup()
    assert result["spans_purged"] == 1
    assert len(store) == 0

def test_cleanup_idempotent():
    store = EphemeralTraceStore()
    store.add({"span_id": "a"})
    store.cleanup()
    assert store.cleanup()["already_clean"] is True

def test_add_after_cleanup_noop():
    store = EphemeralTraceStore()
    store.cleanup()
    store.add({"span_id": "b"})
    assert len(store) == 0

def test_encrypted_temp_unreadable_at_rest(tmp_path):
    pytest.importorskip("cryptography")
    store = EphemeralTraceStore(mode="encrypted_temp", temp_dir=str(tmp_path))
    store.add({"span_id": "a", "attributes": {"v": "loan 1234567890"}})
    path = store._temp_path
    raw = path.read_bytes()
    assert b"1234567890" not in raw   # encrypted at rest

def test_encrypted_temp_shredded_on_cleanup(tmp_path):
    pytest.importorskip("cryptography")
    store = EphemeralTraceStore(mode="encrypted_temp", temp_dir=str(tmp_path))
    path = store._temp_path
    store.add({"span_id": "a"})
    store.cleanup()
    assert not path.exists()

def test_encrypted_temp_permissions(tmp_path):
    pytest.importorskip("cryptography")
    store = EphemeralTraceStore(mode="encrypted_temp", temp_dir=str(tmp_path))
    try:
        assert oct(store._temp_path.stat().st_mode)[-3:] == "600"
    finally:
        store.cleanup()
```

### 12.4 Acceptance criteria

All of the following must be true before the implementation is complete:

1. `pytest` exits 0 with all scrubber and ephemeral-store tests passing
2. `o2a-eval graph --agent-dir tests/fixtures/agents --pipeline <root>` prints the
   correct tree with no `[<missing>]` nodes
3. `o2a-eval lint --agent-dir tests/fixtures/agents --pipeline <root>` runs without
   error and emits only static-stage flaws
4. `o2a-eval score tests/fixtures/trace.jsonl --agent-dir tests/fixtures/agents
   --pipeline <root> --no-judge --latency-budget-s 60` catches all 6 seeded flaws
   and produces none of the listed false positives (sequential_reorder, dead_output
   on terminal/router, router_not_exhaustive on the neq router, structural_drift on
   a valid run)
5. `--json` output on the above command contains no seeded PII string
6. `pip install -e ".[all]"` succeeds and `o2a-eval --help` works
7. Removing sqlglot, rich, cryptography each individually does not break the test
   suite (checks degrade or skip, they do not raise)

---

## 13. BUILD ORDER

Implement in this sequence. Each step must leave the test suite green.

1. `core/models.py` — all dataclasses, no external deps
2. `core/graph.py` + fixture YAMLs — verify `o2a-eval graph` tree output
3. `core/trace.py` + `tests/fixtures/make_trace.py` — verify tree reconstruction
4. `core/registry.py` + `core/rollup.py` with a single trivial check
5. `checks/hygiene.py` D1 only — cheapest signal, proves the check loop
6. `checks/structural.py` — test for false positives after each check
7. `checks/router.py` — test neq exclusion in R-S3 specifically
8. `checks/database.py` — test with and without sqlglot importable
9. `checks/prompts.py` — test N3 against-input_keys logic
10. `checks/lifecycle.py`
11. `report/scrubber.py` + safety tests — must pass before step 12
12. `report/terminal.py` + `cli.py`
13. `capture/ephemeral.py` + safety tests
14. `adapters/base.py` + `capture/attach.py`
15. `checks/judge.py` — requires no model; test endpoint guard only
16. Full acceptance criteria run

---

## 14. KNOWN LIMITS AND EXCLUSIONS

- Cross-trace checks (dead branch distribution, latency baselines across runs) are
  not in scope for v1. `stage="cross_trace"` is reserved in the registry.
- The judge rubrics for N4/N5 are unvalidated until run against 50–100 human-labelled
  spans. Treat their output as a hint, not a verdict, until validation is done.
- R-R1 degrades gracefully if the framework only exposes the winning route: if
  `evaluated_routes` contains no per-route `matched` flags, the check emits nothing
  rather than guessing.
- On modern SSDs, the shred algorithm (overwrite + zeros) does not guarantee physical
  erasure due to wear-levelling. It is a best-effort measure; the encrypted_temp
  design treats the key as the primary protection.
- `install_enrichment` requires the framework to have a single base class. If agents
  inherit from multiple unrelated base classes, the adapter must be called with each
  base class path separately.
