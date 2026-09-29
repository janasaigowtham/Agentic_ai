# o2a_gen: procedure document → O2A agent YAMLs

Reads a review procedure, plans the agent workflow, looks up the real tables, existing
agents, tools and reference docs each step needs, and writes one O2A YAML per agent. It then
validates the result and, if you have a hand-built version, scores the generated set against it.

Every LLM call goes through **one file you control** (`providers/tachyon_provider.py`), so it
runs on Tachyon with whatever models you choose.

```
procedure.md ──► 1 extract steps ──► 2 plan agents ──► 3 ground each step ──► 4 write YAMLs ──► 5 validate
                 (LLM)               (no LLM)          (navigate catalog +      (one per agent)   (static checks)
                                                        LLM writes fields)
                                                              ▲
catalog_src/ ──► compile-catalog (Corpus2Skill) ──────────────┘
 agents/ tables/ tools/ reference/
```

## Where Corpus2Skill is used

The procedure itself is small and ordered, so it is read directly. The **catalog** is the part
that is too large to paste into a prompt: every table definition, existing agent YAML, tool and
guideline. `compile-catalog` runs the vendored Corpus2Skill compiler (`third_party/corpus2skill`,
unmodified) over it to build a navigable skill tree. During generation, `navigator.py` browses
that tree for each step, the way Corpus2Skill's serve step does, but locally: nothing is uploaded
and no server-side code execution is used.

## Setup

```bash
pip install -r o2a_gen/requirements.txt
cp o2a_gen/gen_config.example.yaml gen_config.yaml
```

1. **Connect Tachyon.** Edit the three `TODO(tachyon)` blocks in
   `o2a_gen/providers/tachyon_provider.py`: build the client, make a chat call, and (optionally)
   an embeddings call. Nothing else in the code knows about Tachyon. Read keys from the
   environment, never from the file.
2. **Choose models.** In `gen_config.yaml`, set each role under `models:` to a Tachyon model ID.
   `llm_agent` is the value written into `model:` of generated LlmAgent YAMLs.
3. **Choose embeddings.** `embedding.provider: local` runs sentence-transformers on your
   machine (`pip install sentence-transformers`; point `model` at an internal mirror if
   Hugging Face is blocked). `provider: llm` uses the `embed` function in the Tachyon file instead.

## Build the catalog (once, and again when it changes)

```
catalog_src/
  agents/      existing O2A agent YAMLs from any pipeline, the best examples to copy
  tables/      .sql DDL, .csv (table,column,type,description), .md notes (which connection
               env var each table uses), or .yaml/.json table specs
  tools/       tool definitions (.yaml/.json with a name, or .md)
  reference/   policies, investor guidelines, SOP excerpts (.md/.txt)
```

```bash
python -m o2a_gen compile-catalog --config gen_config.yaml --src catalog_src --out catalog_build
```

Put only non-PII reference material in the catalog. Its text is sent to the LLM when compiling.

## Generate

```bash
python -m o2a_gen generate --config gen_config.yaml \
    --procedure procedures/pmi_ddn.md --catalog catalog_build --out generated/pmi_ddn
```

Procedures can be `.md`, `.txt`, `.docx` (`pip install python-docx`) or `.pdf` (`pip install pypdf`).
Omit `--catalog` to generate from the procedure alone (no grounding).

Output:

| File | What it is |
|---|---|
| `generated/pmi_ddn/*.yaml` | one agent per file, each headed by the procedure step and lines it came from |
| `generated/pmi_ddn_steps.json` | the steps as extracted from the procedure, to check the reading was right |
| `generated/pmi_ddn_report.json` | step → agent coverage, which catalog docs grounded each agent, gaps the model reported, validation findings, LLM usage |

Exit code is 1 if validation found errors.

## Validate and compare

```bash
python -m o2a_gen validate generated/pmi_ddn --inputs loan_number
python -m o2a_gen compare --gold pipelines/pmi_ddn --generated generated/pmi_ddn
```

`validate` also works on hand-written pipelines. It checks:
- every referenced agent has a YAML, and each file is named after its agent
- every key an agent reads was written earlier on its path; keys written in only some router
  branches produce a warning
- LlmAgent `{refs}` and transform `{{ refs }}` are declared in `input_keys`
- SQL `:params` are declared, and connection URLs are `${ENV:VAR}`, never literals
- router targets match `sub_agents`, priorities are unique, and there is a default route

`compare` matches agents by name, then by output_key and class, then by similar name, and reports
coverage, class/key agreement, input-key overlap, and whether each database agent reads the same
tables. Run it against the existing PMI DDN YAMLs to measure the generator before trusting it on
a new procedure.

## How each step type becomes an agent

| Procedure step | agent_class | Grounded with |
|---|---|---|
| look up / pull data | `database_agent` | table definitions, connection env var, similar database agents |
| derive / flag / format | `slv_transformation_agent` | similar transform agents (for `$cond`/`$fn`/`$each` syntax) |
| review / judge / write | `LlmAgent` | guidelines, similar LlmAgents, tools from the catalog |
| if … otherwise … | `decision_router_agent` + one target per branch | the upstream flag |
| supervisor approval | `agent_gate` | runtime fields copied from existing gates |
| a phase of several steps | `SequentialAgent` | |
| the whole procedure | `resumable_orchestrator` | |

Each model answer is checked before it is accepted. For example, it must reference only session
keys that exist at that point, bind SQL values as `:params`, name only catalog tools, and include
one route per branch. A failed answer is sent back to the model with the error, for up to 3 tries.

Fields the procedure can't supply (timeouts, gate resume events, retries) can be set per class
under `generation.agent_templates`, or copied from similar catalog agents by the model.

## Tests

```bash
python -m pytest tests/o2a_gen -q
```

The tests use a scripted fake model (`tests/o2a_gen/fake_tachyon.py`), so they run offline. They
cover the real Corpus2Skill compile, navigation, validation retries, YAML output, and a full run
compared against a hand-built PMI DDN-style set in `tests/o2a_gen/fixtures/gold`. The fixture
procedure and catalog are illustrative, not real policy.

## Limits

- Generated YAMLs are a draft for review, not a replacement for sign-off.
- Very long procedures may exceed the extraction call's output limit; split them by phase.
- The `agent_gate` and tool fields depend on your runtime's schema. Supply them with templates or
  good catalog examples.
