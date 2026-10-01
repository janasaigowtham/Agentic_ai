# o2a_gen: procedure document → O2A agent YAMLs

Architecture and a detailed end-to-end walkthrough: open [`docs/architecture.html`](docs/architecture.html) in a browser.

Generates the agent YAMLs for a review workflow from four inputs:

| Input | What it gives the generator |
|---|---|
| **Procedure** | the steps, their order, the decisions and the approvals |
| **Agent syntax** | the O2A runtime schema of each `agent_class`: its fields and rules |
| **Metadata** | tables, columns and connections the lookups use |
| **Tools** | tools that LlmAgents may call (plus optional reference docs: policies, guidelines) |

## Web UI (upload the inputs, get the workflow YAMLs)

```bash
cd "/path/to/Agentic_ai"
source ../.venv/bin/activate
pip install -r o2a_gen/requirements.txt
export ANTHROPIC_API_KEY="..."       # Claude Opus 5.5
export DEEPINFRA_API_KEY="..."       # Nemotron-3-Embed-8B
python -m o2a_gen.webui              # open http://127.0.0.1:8765
```

Drop in the procedure (Markdown), tools, metadata and agent syntax, press **Generate
workflow**, and watch each stage. The result is the agent tree with every YAML, the gaps
(values the inputs did not give, left as empty placeholders) and the skills each agent was
written from, plus a zip of everything. Existing YAMLs can be added under *Options*; they are
compared only after generation. Each run is kept in `runs/<id>/` (git-ignored). The UI uses
`gen_config.anthropic.yaml`; set `O2A_GEN_CONFIG` to use another config.

### Playbook mode (recommended)

Upload a **playbook** (Markdown) with the other inputs. It says how to build the workflow:
how to check the logic, how to orchestrate the agents and how to map data from tools. When
a playbook is given, generation runs `harness.py` instead of the step-by-step generator:

| Stage | Applies | Saves (next to `workflow/`) |
|---|---|---|
| Recipe | the playbook, as a build checklist | `workflow_recipe.json` |
| Logic | the playbook's logic rules to the procedure | `workflow_flow.json` / `.md` |
| Data map | its data rules to the tools and metadata | `workflow_data_map.json` / `.md` |
| Orchestrate | its roles and orchestration rules | `workflow_plan.json` / `.md` |
| Write | one writer per agent: plan entry, class syntax, sources | `workflow/*.yaml` |
| Review | code checks + a fresh model review; blocking problems go back to the writer | `workflow_review.json`, `workflow_placeholders.md` |

The playbook and the agent syntax are the system prompt of every call (cached). Between
stages, code checks what came back and sends problems back to the model: classes must be
in the syntax, cited procedure lines must exist, every mapped tool and field must appear
in the tools or metadata, every check in the flow must be covered by an agent, routes and
sub-agents must exist, and every key an agent reads must be written by some agent or be a
pipeline input. Nothing in the code is specific to a procedure, tool or agent class. The
build notes (`flow.md`, `data_map.md`, `plan.md`, `placeholders.md`) appear in the web UI
under the stats, and in the zip.

### Claude cost settings (`gen_config.anthropic.yaml`)

- **Prompt caching.** Every system prompt is marked for caching, so repeated calls with the
  same instructions read them at a fraction of the input price. In a conversation (retries,
  the navigator) the latest turn is marked too, so the history is read from cache on the
  next turn. A one-shot prompt's own text is not marked, because it is never read back.
- **Batch API** (`llm.batch`). Compile calls that run many at once go through the Message
  Batches API at half price: catalog summary cards, folder pages, and procedure section
  summaries. Groups smaller than `batch_min_requests` are sent as normal calls, with no batch
  wait. Anything a batch doesn't answer cleanly is retried as a normal call: a declined item
  (which then goes to the fallback model), an error, or an answer that doesn't parse. If the
  Batch endpoint itself fails, the whole group falls back to normal calls. Most batches finish
  within minutes, and the limit is 24 hours. `O2A_ANTHROPIC_BATCH_MAX_WAIT_MINUTES` (default
  120) cancels a batch that runs longer. Remove `llm.batch` to turn batching off while you
  iterate.
- **Effort.** Compile calls use `catalog.effort` (default `low`). Everything else uses
  `O2A_ANTHROPIC_EFFORT`, or the model's default when it is unset.
- **Usage.** `llm_usage` in the run report, and the last line of `compile-catalog`, show
  calls, batch calls, cache reads and cache writes.

Existing agent YAMLs are **not** an input. They are used only after generation, to measure
how close the generated set is (`--compare-with`, or `compare`).

```
                      agent syntax ┐
 metadata, tools, reference docs ──┴─► compile-catalog ─► catalog skill tree (+ saved vectors)
                                                               │                  │
 procedure ─► 1 compile procedure ─► procedure skill tree ◄────┼── similarity ────┘
              (sections by heading,   (summaries, related      │   (likely tools & data
               LLM summary, embed)     sections, hints)        │    per section)
                    │                        │                 │
                    └► 2 extract steps ─► 3 plan ─► 4 ground ◄─┘ ─► 5 write YAMLs ─► 6 validate ─► compare
                       (LLM, whole text   (no LLM)  (browse procedure section +                   (only if existing
                        + outline)                   catalog, class syntax, LLM)                   YAMLs are given)
```

Every LLM call goes through **one file you control** (`providers/tachyon_provider.py`), so it
runs on Tachyon with the models you choose.

## The technique (compile, then navigate)

This follows the Corpus2Skill approach. **The procedure is the corpus**: it is embedded and
summarised into a skill tree, and the question answered for each of its steps is "what agent
YAML implements this step?". The **tools, metadata and agent syntax** are the knowledge the
answer is built from, compiled into a second tree.

- **Procedure tree** (`proctree.py`, at generate time). The procedure is split into sections
  by its own headings, **in document order**. Order is the workflow, so it is not re-clustered.
  The LLM summarises each section bottom-up, and each section is embedded. The embeddings add
  what headings can't:
  - **related sections**: the most similar other parts of the procedure (definitions,
    appendices, rules stated elsewhere);
  - **likely tools & data**: the catalog's metadata, tools and reference documents most
    similar to the section, using the catalog's saved vectors.
- **Catalog tree** (`compile-catalog`): built from the agent syntax, metadata, tools and
  reference docs. It is grouped by kind, then by embedding similarity, and summarised.
- **Extraction** reads the whole procedure, so no step can be missed, plus the section outline.
- **Grounding**: for each step, the navigator browses one combined tree (`procedure/` plus the
  catalog folders). It starts from the step's own procedure section, its related sections and its
  likely tools & data, and fetches the actual documents before using them. The **agent-syntax
  section for the agent's class is always included** directly, and the validator checks every
  generated field against it.
- Similarity hints are suggestions: the model still opens the documents, and every answer is
  checked before it is accepted.

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
3. **Choose embeddings.** One model embeds both the catalog (tools, metadata, agent syntax,
   reference) and the procedure's sections, so the two can be compared.
   - `embedding.provider: local` runs sentence-transformers on this machine's GPU
     (`pip install sentence-transformers` plus a CUDA build of torch). The example config is set
     up for `nvidia/Nemotron-3-Embed-8B-BF16` (the Hugging Face ID; DeepInfra's is `nvidia/Nemotron-3-Embed-8B`): `trust_remote_code`, bfloat16 (about 16 GB of GPU
     memory), optional `dimensions` to keep the first N of its 4096 dimensions, and a
     `query_template` used only when matching procedure sections to the catalog. Documents are
     always embedded without a prefix. The model is loaded once per run.
   - `provider: llm` sends texts to the `embed` function in the provider file instead (for
     example, the same model served by vLLM).
   - Changing the model, `dimensions` or `document_template` requires rebuilding the catalog;
     vectors saved with different settings are ignored, with a message.

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
| `generated/pmi_ddn_procedure/` | the compiled procedure tree: sections, summaries, related sections, likely tools & data (`procedure_links.json`) |
| `generated/pmi_ddn_steps.json` | the steps as extracted from the procedure, to check the reading was right |
| `generated/pmi_ddn_report.json` | step → agent coverage, each step's procedure section, which catalog docs grounded each agent, gaps the model reported, validation findings, LLM usage, and `comparison` if `--compare-with` was given |

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

## How agent classes and fields are chosen

Nothing in the code knows any agent class. The classes are read from the agent syntax
document (its class table, its `agent_class:` examples, and headings that name a class), and:

- **Extraction** gives the model the whole syntax document, and it picks each step's
  `agent_class` from what the syntax says each class is for, plus the orchestrator class and the
  class that runs a phase's steps in order. Branching steps get the class the syntax gives for
  routing.
- **Writing an agent** gives the model that class's syntax section and the documents the
  navigator found in the compiled skill tree (the step's procedure section, its likely tools and
  data from the embedding match, and whatever it browses to). Every value comes from those
  documents; a value they do not give is left as `""` and listed as a gap.
- **Checks** are class-independent: fields must be named in the class's syntax, every
  session-key reference (`{{ key }}`, `{key}`, `:key`) must exist when the agent runs, and a
  router must route to each branch's agent. A failing answer is sent back with the error, up to 3
  tries; after that it is kept and the problem reported.

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
- The agent classes come only from the agent syntax document; if a class is missing there, it
  cannot be chosen.
