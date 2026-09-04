# Trajectory Review Harness — Specification
# Version: 0.1 (handoff draft)
# Audience: engineering team or coding agent implementing/enhancing this in a target system
# Purpose: a portable spec of the architecture design settled on for reviewing an agent's
# full execution trajectory — its plan, tool calls, and queries — and turning that review
# into a root-caused, human-approved recommendation.

---

## 1. PURPOSE

Given a completed (or in-progress) execution trajectory of an agentic pipeline, produce:
1. A full-detail account of what happened at every step, judged against what was declared.
2. A root-caused explanation when something went wrong — not just "step 4 looked off," but
   "step 4 is why step 9 failed."
3. A concrete, proposed recommendation tied to that root cause.
4. None of it applied automatically. A human approves or declines before anything changes.

This harness never drives the target pipeline. It watches a trajectory that already exists
or is being produced by a system outside this spec's scope.

---

## 2. DESIGN PRINCIPLES (non-negotiable)

1. **Observer, never driver.** The harness must never invoke or influence the target
   pipeline. It reads a trajectory; it does not produce one.
2. **No rule engine in the judgment path.** Every claim made about the trajectory — a step
   was inefficient, a plan was violated, an escalation should have happened — is made by a
   model, not a deterministic check. Plain code exists only as control flow: sequencing
   stages, deciding which specialist fires, gating on approval. Control flow forms no
   opinion about the trajectory it's moving.
3. **No stage waits on a shared schedule.** Every stage and every specialist is
   event-triggered — it fires the instant the specific fact it depends on exists. Some
   facts exist early (one step just finished); some exist only once the trajectory itself
   completes. Neither is "waiting" — both are a trigger firing when its condition is met.
4. **Exhaustive, not selective, analysis.** Every step gets the same depth of scrutiny
   regardless of how minor it looks going in — nothing is pre-judged as unimportant before
   it's actually been examined.
5. **Exhaustive, not selective, reporting.** No cap on findings, no dropping findings to
   keep a report short. Root-cause-linked findings surface first for triage; every other
   finding stays in the report, in full detail, underneath.
6. **Nothing acts without a human.** A recommendation is a proposal. It sits behind a
   human approval gate before anything is applied to a live system. A decline is held and
   fed back as data — never silently discarded, never auto-retried against the live system.
7. **No implicit model memory.** No stage retrains or fine-tunes on trajectory review
   outcomes. The only improvement mechanism is external: disagreements get logged, a human
   reviews the pattern, a human revises a rubric. See §5.6.

---

## 3. CORE VOCABULARY

- **Trajectory** — the full, ordered record of what an agent actually did to complete one
  task: every step, every tool call, every decision, start to outcome.
- **Step** — one unit of the trajectory: a tool call and its result, or one decision point
  (a delegation, a gate, an escalation, a plan branch).
- **Declared contract** — what a step was *supposed* to do, per the target system's own
  spec (e.g. a YAML node's declared inputs, outputs, query, or routing options), when the
  target system has one. Optional — see §8.3.
- **Case summary** — the one artifact Stage 01 produces and every downstream stage reads
  instead of re-reading the raw trajectory: the goal, the plan actually followed, the
  outcome, and a map of each step's type.
- **Verdict** — one specialist's finding about one step (or a small set of related steps).
- **Root-cause chain** — the aggregator's trace from a bad outcome back through the
  verdicts to the step that actually caused it.
- **Recommendation** — a proposed fix tied to a root cause, never applied automatically.

---

## 4. DATA MODEL

Field lists below are a minimum contract, not a storage schema. The implementing system
chooses its own transport/persistence format.

### 4.1 Trajectory / Step

```
Trajectory:
  trajectory_id        string
  source_pipeline      string   — identifier of the target system/pipeline
  source_spec_ref      string?  — pointer to the declared spec directory, if one exists
  steps                list[Step], in execution order
  started_at           timestamp
  completed_at          timestamp?  — null while still in progress
  outcome               any

Step:
  step_id               string
  step_type             string   — open-ended; initial set: tool_call | delegation |
                                    gate_permission | escalation | plan_node
  agent_name            string   — the declared node/agent that executed this step
  input                 any
  output                any
  started_at            timestamp
  completed_at          timestamp
  parent_step_id         string?  — for delegated/nested steps
  declared_contract_ref  string?  — pointer to this step's declared spec, if available
```

### 4.2 Case Summary (Stage 01 output)

```
CaseSummary:
  trajectory_id     string
  goal              string   — the task as stated
  plan_declared     any?     — the declared plan, when a spec exists
  plan_actual       list[step_id]  — the order steps actually ran in
  outcome           any
  step_type_map     dict[step_id, step_type]
  orient_notes      list[string]  — things noticed at a glance; not verdicts, pointers only
```

### 4.3 Verdict (Stage 02 output)

```
Verdict:
  verdict_id      string
  specialist      string   — which specialist produced this (see §6)
  step_ids        list[step_id]
  trigger_event   string   — the fact that caused this specialist to fire
  finding         string
  severity        enum(none, minor, significant, critical)
  evidence        any      — must not carry PII if persisted; scrub before storage
```

### 4.4 Recommendation (Stage 03 output)

```
Recommendation:
  recommendation_id     string
  trajectory_id         string
  tied_to_root_cause     bool
  root_cause_chain       list[step_id]   — ordered, cause to effect
  description            string
  proposed_fix           string
  status                 enum(proposed, approved, declined, held)
```

### 4.5 Gate Decision

```
GateDecision:
  recommendation_id   string
  decision             enum(approved, declined)
  reviewer             string
  note                 string?
  timestamp             timestamp
```

### 4.6 Calibration Record

```
CalibrationRecord:
  case_id           string
  trajectory_id      string
  harness_verdict     any
  human_label         any
  agreement           bool
  reviewed_by         string
  timestamp            timestamp
```

---

## 5. STAGE SPECIFICATIONS

### 5.1 Stage 01 — Orienting agent

| | |
|---|---|
| Trigger | Trajectory available for review |
| Reads | Raw trajectory; declared contract per touched step, if available |
| Writes | Case Summary (§4.2) |
| Model tier | Strongest available — no exceptions |
| Note | Single point of failure. Nothing downstream re-derives its work; an error here propagates uncaught. Never run this on a cost-optimized tier. |

### 5.2 Stage 02 — Specialists (event-triggered fan-out)

General contract, applies to every specialist in §6:
- Fires the instant its trigger fact exists — never on a shared schedule with other
  specialists, never held for a "batch."
- Only invoked at all if its step type occurs in this trajectory.
- Fires once per occurrence of its step type (a trajectory with two delegation steps
  triggers the delegation specialist twice, independently).
- Reads: Case Summary + only the step(s) relevant to it + that step's declared contract,
  if available. Never the full raw trajectory.
- Writes: zero or more Verdicts (§4.3).

### 5.3 Stage 03 — Aggregator

| | |
|---|---|
| Trigger | Trajectory review complete — no further Verdicts expected |
| Reads | Case Summary, all Verdicts for this trajectory |
| Writes | Root-cause chain; full finding set (every Verdict retained — §7); Recommendation(s) |
| Model tier | Strongest available |
| Note | The only stage doing cross-step reasoning. A recommendation must reference the specific step/tool call/query involved — not a generic fix. |

### 5.4 Stage 03.5 — Fact-check

| | |
|---|---|
| Trigger | Aggregator output produced |
| Reads | Original prompt/task, trajectory outcome, aggregator's report + recommendation(s) |
| Writes | Confidence-checked report, or a routing signal back to Stage 03 |
| Model requirement | A genuinely different model lineage than the aggregator — not just a different prompt on the same model. Exists to catch the aggregator's own blind spots; a same-model reviewer tends to share them and just agree. |

### 5.5 Gate — Human approval

| | |
|---|---|
| Type | Not a model. A checkpoint. |
| Reads | Fact-checked report + recommendation(s) |
| Writes | GateDecision (§4.5) per recommendation |
| Rule | Never applies a recommendation automatically, regardless of confidence. A decline is held, not discarded, and MUST become a CalibrationRecord candidate (§5.6). |

### 5.6 Stage 04 — Calibration

| | |
|---|---|
| Trigger | (a) scheduled baseline, weekly by default; (b) immediately on any change that could introduce drift — a specialist rubric edit, a model swap, a change to the target pipeline's own declared spec |
| Reads | A working set of human-labeled trajectories (recommended 40–60, refreshed periodically so it doesn't go stale as the target system's real behavior legitimately shifts); recent Verdicts/Recommendations; declined GateDecisions |
| Writes | Agreement rate; drift flag; candidate rubric revisions |
| Reviewer | A human who knows the target system, rotating — never the Fact-check agent grading its own calibration |
| Note | Agreement-rate computation is arithmetic, not a model call — it is not a judgment about the target agent, it's measuring the harness against ground truth. A model is invoked only to characterize *why* a disagreement happened, and only on the disagreeing cases. A cluster of declines around one specialist or one recommendation type is what should trigger a rubric review — not any single decline restarting a trajectory's evaluation. |

---

## 6. SPECIALIST REGISTRY (initial set — extensible, see §8.1)

| Specialist | Judges | Trigger | Tier |
|---|---|---|---|
| Tool-call | Necessity, efficiency, correct use of results downstream. For query/DB steps: also join necessity and filter placement, not just gross patterns like SELECT * | Per step, the instant it completes | Two-pass — cheap pass catches the obvious cases (no filter, unused columns); escalates to the strong tier only for ambiguous semantic calls (is this join actually needed) |
| Delegation | Sound decomposition; was the delegate's output verified or blindly trusted | The step immediately after the delegation returns (not the delegation step itself — that fact isn't knowable yet) | Strong |
| Gate / permission | Was risk classified correctly in the moment | Per step, the instant it completes | Lean |
| Escalation | Should this step have escalated to a human, or was continuing correct | Per step, the instant it completes | Strong — this is a genuinely ambiguous call in both directions and asymmetric to get wrong |
| Plan-fidelity | Did execution match the declared order; was a declared step skipped entirely | Per step, live, for ordering violations; on trajectory completion, for "was something skipped" (that fact only exists once nothing else is coming) | Lean |

---

## 7. RECOMMENDATION POLICY

- **No cap. No dropping.** Every finding produced by every specialist stays in the final
  report, in full detail. A run that produces more findings does not lose any of them to
  keep the report short.
- **Root-cause-linked findings surface first**, for triage — this is an ordering rule, not
  a filter. Everything else remains fully present underneath.
- **Every step gets equal analytical depth**, decided by nothing except its type — never
  a lighter pass because a step looked unimportant before being examined.

---

## 8. EXTENSION POINTS

### 8.1 Adding a specialist

Add one when the target system's action space includes a genuinely new *category* of
judgment — not a new tool, not a new YAML node. Rule of thumb: if the new specialist would
end up checking mostly what an existing one already checks (necessity, correct use of
results), it isn't a new category; extend the existing specialist's rubric instead.

New specialists plug in without touching Stage 03, Stage 03.5, the gate, or Stage 04 —
those stages consume "however many verdicts arrived," generically, by construction.

### 8.2 Model tier policy

Tiers in §6 are a recommendation, not a hard requirement. The implementing system may
apply its own cost/quality policy per specialist — the one constraint that should not be
relaxed is Stage 01 and Stage 03 staying on the strongest available tier; they are the two
single points of failure in this design (§2.1, §5.1, §5.3).

### 8.3 Declared-spec integration

This spec assumes the target system can supply, per step, a declared contract: inputs,
outputs, a tool/query definition, routing options — e.g. a YAML-per-agent pipeline in the
style of O2A_EVALUATOR_SPEC.md. If the target system has no declared-spec layer, Stage 01's
case summary is built from the trajectory alone, and specialists lose the "declared vs.
actual" comparison — they can still judge a step in isolation, but with materially less
grounding, and several specialists (plan-fidelity especially) lose most of their value.
Wiring in declared-spec access is the single highest-leverage enhancement for a target
system that currently lacks one.

---

## 9. EXPLICIT NON-GOALS

- Does not define exact prompts or rubric text for any specialist — left to the
  implementing system, informed by the judgment focus column in §6.
- Does not define a storage or transport format — §4 is a minimum field contract, not a
  schema for a specific database or message bus.
- Does not define how the target pipeline is run or instrumented — this harness assumes a
  Trajectory (§4.1) already exists or is being produced by a system outside this spec's
  scope (§2.1).
- Does not include model fine-tuning or weight updates. The only improvement path is
  Stage 04 surfacing a human-reviewed rubric revision (§5.6) — never automatic, never
  silent.
