# o2a_gen: procedure document → O2A agent YAMLs

Architecture and a detailed end-to-end walkthrough: open [`docs/architecture.html`](docs/architecture.html) in a browser.

Generates the agent YAMLs for a review workflow from four inputs:

| Input | What it gives the generator |
|---|---|
| **Procedure** | the steps, their order, the decisions and the approvals |
| **Agent syntax** | the O2A runtime schema of each `agent_class`: its fields and rules |
| **Metadata** | tables, columns and connections the lookups use |
| **Tools** | tools that LlmAgents may call (plus optional reference docs: policies, guidelines) |

Existing agent YAMLs are **not** an input. They are used only after generation, to measure
how close the generated set is (`--compare-with`, or `compare`).

```
                      agent syntax ┐
 metadata, tools, reference docs ──┴─► compile-catalog ─► skill tree (browsable folders)
                                                               │
 procedure ─► 1 extract steps ─► 2 plan agents ─► 3 write each agent ─► 4 YAMLs ─► 5 validate ─► 6 compare
              (LLM)              (no LLM)         (browse the tree for          (static checks   (only if existing
                                                   metadata/tools + the class's  + agent syntax)  YAMLs are given)
                                                   syntax section, then LLM)
```

Every LLM call goes through **one file you control** (`providers/tachyon_provider.py`), so it
runs on Tachyon with the models you choose.

## The technique (compile, then navigate)

This follows the Corpus2Skill approach: instead of pasting everything into one prompt or using
a vector search, the inputs are **compiled** into a folder tree with a summary at each level, and
the model **navigates** it (read a folder summary, open a sub-folder, fetch a document) to find
what each step needs.

- `compile-catalog` builds the tree from the agent syntax, metadata, tools and reference docs.
- While writing each agent, `navigator.py` browses the tree for that step's tables, connection
  and tools. It sends one JSON action per turn, so any chat model works, and nothing is uploaded.
- The **agent-syntax section for the agent's class is always included** directly (not left to
  navigation), since every agent must follow it. The validator also checks every generated field
  against it.

Two interchangeable engines build the tree (`catalog.engine`):

- **`native` (default): `o2a_gen/skilltree.py`.** Written for this project, with no third-party
  code. Top folders are fixed by kind (`agent-syntax`, `data-metadata`, `tools`,
  `reference-docs`), then split by k-means on embeddings until a folder lists at most `leaf_max`
  documents. Folders are named and summarised bottom-up by the LLM; per-document cards are cached
  by content hash, so rebuilds only re-summarise changed documents. Needs only numpy.
- **`corpus2skill`: the vendored research implementation** (`third_party/corpus2skill`, MIT),
  run through `c2s_bridge.py` with its LLM calls redirected to your client. Kept for comparison.

Both produce the same layout. To drop the third-party code, delete `third_party/corpus2skill/`
and `o2a_gen/c2s_bridge.py`; the native engine never imports them.

## Setup

```bash
pip install -r o2a_gen/requirements.txt   # scikit-learn is only needed for engine: corpus2skill
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

## 1. Build the catalog (once, and again when the inputs change)

```
catalog_src/
  schema/      REQUIRED. The O2A agent syntax document(s). Markdown/text is split on headings
               that name an agent_class (e.g. "## database_agent"); anything else counts as
               general rules and is given to every agent. YAML/JSON keyed by class also works.
  metadata/    .sql DDL, .csv (table,column,type,description), .md notes (e.g. which connection
               env var each table uses), or .yaml/.json table specs
  tools/       tool definitions (.yaml/.json with a name, or .md)
  reference/   optional: policies, investor guidelines, SOP excerpts (.md/.txt)
```

```bash
python -m o2a_gen compile-catalog --config gen_config.yaml --src catalog_src --out catalog_build
```

An `agents/` folder here is ignored with a warning. Put only non-PII material in the catalog:
its text is sent to the LLM when compiling.

## 2. Generate, then compare

```bash
python -m o2a_gen generate --config gen_config.yaml \
    --procedure procedures/pmi_ddn.md --catalog catalog_build --out generated/pmi_ddn \
    --compare-with pipelines/pmi_ddn        # optional: existing YAMLs for this procedure
```

Procedures can be `.md`, `.txt`, `.docx` (`pip install python-docx`) or `.pdf` (`pip install pypdf`).

`--compare-with` is read only after the YAMLs are written. As a guard against existing YAMLs
leaking into generation, `generate` refuses to run if:
- the catalog contains agent documents,
- the catalog was built from files inside the comparison folder, or
- the comparison folder is the output folder.

Output:

| File | What it is |
|---|---|
| `generated/pmi_ddn/*.yaml` | one agent per file, each headed by the procedure step and lines it came from |
| `generated/pmi_ddn_steps.json` | the steps as extracted from the procedure, to check the reading was right |
| `generated/pmi_ddn_report.json` | step → agent coverage, which catalog docs grounded each agent, gaps the model reported, validation findings, LLM usage, and `comparison` if `--compare-with` was given |

Exit code is 1 if validation found errors.

## Validate and compare on their own

```bash
python -m o2a_gen validate generated/pmi_ddn --inputs loan_number --catalog catalog_build
python -m o2a_gen compare --gold pipelines/pmi_ddn --generated generated/pmi_ddn
```

`validate` works on any O2A YAML directory, including hand-written ones. It checks:
- every referenced agent has a YAML, and each file is named after its agent
- every key an agent reads was written earlier on its path; keys written in only some router
  branches produce a warning
- LlmAgent `{refs}` and transform `{{ refs }}` are declared in `input_keys`
- SQL `:params` are declared, and connection URLs are `${ENV:VAR}`, never literals
- router targets match `sub_agents`, priorities are unique, and there is a default route
- with `--catalog`: every field appears in the agent syntax for that class

`compare` matches agents by name, then by output_key and class, then by similar name, and reports
coverage, class/key agreement, input-key overlap, whether each database agent reads the same
tables, and which agents were missed or extra.

## How each step type becomes an agent

| Procedure step | agent_class | Written from |
|---|---|---|
| look up / pull data | `database_agent` | metadata: tables, columns, connection env var |
| derive / flag / format | `slv_transformation_agent` | the input data's fields + transform syntax |
| review / judge / write | `LlmAgent` | guidelines, tools from the catalog |
| if … otherwise … | `decision_router_agent` + one target per branch | the upstream flag |
| supervisor approval | `agent_gate` | the gate's fields from the agent syntax |
| a phase of several steps | `SequentialAgent` | |
| the whole procedure | `resumable_orchestrator` | |

Each model answer is checked before it is accepted. It must reference only session keys that
exist at that point, bind SQL values as `:params`, name only catalog tools, and include one route
per branch. A failed answer is sent back to the model with the error, for up to 3 tries. Fields
the syntax requires beyond these (e.g. a gate's resume event) are returned in `extra`, and the
validator flags any field the syntax doesn't define. Fixed per-class defaults can also be set
under `generation.agent_templates`.

## Tests

```bash
python -m pytest tests/o2a_gen -q
```

The tests use a scripted fake model (`tests/o2a_gen/fake_tachyon.py`), so they run offline. They
cover both catalog engines, splitting the agent syntax by class, navigation, validation retries,
YAML output, and a full generate-then-compare run against a hand-built PMI DDN-style set
(`tests/o2a_gen/fixtures/gold`). They also check that nothing from that set appears in any
model call. The fixture procedure, syntax and catalog are illustrative, not real.

## Limits

- Generated YAMLs are a draft for review, not a replacement for sign-off.
- Very long procedures may exceed the extraction call's output limit; split them by phase.
- The step-type → agent_class mapping above is fixed in `plan.py`; change it there if your
  runtime uses different classes for these jobs.
