# O2A agent syntax (test fixture)

Illustrative stand-in for the real O2A runtime schema document. Values below are generic
placeholders, not taken from any production pipeline.

## Common fields

Every agent is one YAML file named after the agent.

- `name` (string, required): matches the file name.
- `agent_class` (string, required): one of the classes below.
- `description` (string): what the agent does.
- `output_key` (string): session key the agent writes.
- `input_keys` (list of strings): session keys the agent reads.
- `sub_agents` (list of `{name}`): children, for container classes.

Session values are referenced as `{{ key }}` in transforms and SQL, and as `{key}` in
LlmAgent instructions.

## database_agent

Runs one SQL query and writes the rows (a list of dicts) to `output_key`.

- `default_db_yaml` (string, required): an embedded YAML document with
  - `type`: database dialect, e.g. `teradata` or `postgres`
  - `connection.url`: always `${ENV:VAR_NAME}`
  - `query`: SQL; bind session values as `:param_name`
- `input_keys`: must list every bound `:param_name`.

## slv_transformation_agent

Evaluates a `transform` block and writes the result to `output_key`.

- `transform` (mapping, required): `{<output_key>: <expression>}`. Expressions may use
  `$cond` (`if: {left, op, right}`, `then`, `else`), `$fn` and `$each`.
- `strict` (bool, default true): when false, missing references evaluate to null.

## transformation_agent

Same fields as slv_transformation_agent.

## LlmAgent

Runs an instruction against a model.

- `instruction` (string, required): the prompt, with `{key}` placeholders.
- `model` (string, required): model identifier.
- `tools` (list of tool names): tools the model may call.

## decision_router_agent

Evaluates `routes` in priority order and hands off to one target.

- `routes` (list, required): each has `conditions` (list of `{context_key, operator, value}`,
  operators `eq`, `neq`, `not_null`, `is_null`, `gt`, `lt`), `target_agent`, `priority`
  (lower runs first). A route with no conditions is the default.
- `sub_agents`: must list every `target_agent`.

## agent_gate

Pauses the pipeline until an external event resumes it.

- `resume_event` (string, required): name of the event that resumes the pipeline.
- `timeout_hours` (number): optional timeout.

## SequentialAgent

Runs `sub_agents` in order.

## resumable_orchestrator

Top-level agent; runs `sub_agents` in order and can resume across gates.

- `version` (string).
