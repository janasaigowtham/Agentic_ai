"""End-to-end: procedure document -> validated O2A agent YAMLs + a review report."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from o2a_gen.compare import compare_dirs
from o2a_gen.config import GenConfig
from o2a_gen.emit import write_plan
from o2a_gen.ground import ground_plan
from o2a_gen.harness import has_playbook, run_playbook
from o2a_gen.llm import LLMClient
from o2a_gen.navigator import Catalog, CombinedCatalog
from o2a_gen.plan import build_plan
from o2a_gen.procedure import extract_procedure, load_procedure_text
from o2a_gen.proctree import build_procedure_tree
from o2a_gen.validate import validate_dir


def generate(procedure_path: Path, out_dir: Path, cfg: GenConfig, client: LLMClient, *,
             catalog_dir: Path | None = None, workers: int = 4, overwrite: bool = False,
             compare_with: Path | None = None) -> dict:
    """Generate from the procedure + catalog only. ``compare_with`` (existing YAMLs) is
    read after the YAMLs are written, never before, so it cannot influence generation."""
    procedure_path, out_dir = Path(procedure_path), Path(out_dir)
    catalog = Catalog(catalog_dir) if catalog_dir else None
    if catalog is not None:
        leaked = sorted({m.get("name") for m in catalog.index.values() if m.get("kind") == "agent"})
        if leaked:
            raise ValueError(f"catalog {catalog_dir} contains existing agent YAMLs {leaked[:5]}; "
                             "rebuild it with compile-catalog (they are not a generation input)")
    if compare_with is not None:
        gold = Path(compare_with).resolve()
        if gold == out_dir.resolve():
            raise ValueError("--compare-with must not be the output directory")
        if catalog is not None and any(
                gold == Path(m.get("source", "")).resolve()
                or gold in Path(m.get("source", "")).resolve().parents
                for m in catalog.index.values()):
            raise ValueError(f"catalog was built from files inside {compare_with}; "
                             "existing YAMLs must not be a generation input")
    old = list(out_dir.glob("*.yaml")) if out_dir.exists() else []
    if old and not overwrite:
        raise FileExistsError(f"{out_dir} already has {len(old)} YAML files; use --overwrite")
    if catalog is None:
        raise ValueError("a compiled catalog is required: it holds the agent syntax, tools and "
                         "metadata the workflow is built from")
    for p in old:
        p.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)

    text = load_procedure_text(procedure_path)
    playbook_mode = has_playbook(catalog)
    proc_tree, browse, outline = None, catalog, ""
    if cfg.generation.get("compile_procedure", True):
        print(f"[1/{7 if playbook_mode else 6}] Compiling procedure {procedure_path.name} "
              "into a skill tree ...")
        proc_dir = out_dir.parent / f"{out_dir.name}_procedure"
        proc_tree = build_procedure_tree(
            client, text, procedure_path.stem, proc_dir, models=cfg.models,
            embedding=cfg.embedding, catalog_root=catalog_dir,
            catalog_index=catalog.index if catalog else None, workers=workers,
            effort=cfg.catalog.get("effort", "low"))
        browse = CombinedCatalog(Catalog(proc_dir), catalog)
        outline = "\n".join(f"{'  ' * max(0, s.level - 1)}- {s.title} (lines {s.start}-{s.end}): "
                             f"{s.summary}" for s in proc_tree.root.walk() if s.level > 0)
        print(f"      {sum(1 for _ in proc_tree.root.walk()) - 1} sections, "
              f"{len(proc_tree.docs)} documents -> {proc_dir}")

    if playbook_mode:
        report = run_playbook(client, cfg, browse, text, out_dir, outline=outline,
                              workers=workers)
        report["procedure"] = procedure_path.name
        report["procedure_tree"] = ({"dir": str(proc_tree.out_dir),
                                     "sections": sum(1 for _ in proc_tree.root.walk()) - 1,
                                     "documents": len(proc_tree.docs)} if proc_tree else None)
        report["llm_usage"] = getattr(getattr(client, "usage", None), "as_dict", lambda: {})()
        if compare_with is not None:
            print(f"[+] Comparing with existing YAMLs in {compare_with} ...")
            report["comparison"] = compare_dirs(Path(compare_with), out_dir)
        (out_dir.parent / f"{out_dir.name}_report.json").write_text(json.dumps(report, indent=2))
        return report

    print("[2/6] Extracting steps (one long answer; can take several minutes) ...")
    proc = extract_procedure(client, text, model=cfg.model("extract"),
                             classes=catalog.classes(), syntax=catalog.syntax_document(),
                             source=procedure_path.name, outline=outline)
    steps = proc.all_steps()
    print(f"      {len(proc.phases)} phases, {len(steps)} steps")

    print("[3/6] Planning agents ...")
    plan = build_plan(proc, cfg)
    print(f"      {len(plan.nodes())} agents")

    print("[4/6] Writing each agent's fields from its syntax, tools and metadata ...")
    groundings = ground_plan(client, plan, cfg, browse, max_workers=workers, proc_tree=proc_tree,
                             procedure_text=text)

    print(f"[5/6] Writing YAMLs to {out_dir} ...")
    write_plan(plan, cfg, out_dir, procedure_path.name)

    print("[6/6] Validating ...")
    syntax = ({n.agent_class: catalog.syntax_for(n.agent_class) for n in plan.nodes()}
              if catalog else None)
    findings = validate_dir(out_dir, plan.inputs, syntax)

    index = browse.index if browse else {}
    report = {
        "procedure": procedure_path.name,
        "pipeline": plan.root.name,
        "pipeline_inputs": plan.inputs,
        "agents": len(plan.nodes()),
        "coverage": [
            {"step": s.id, "title": s.title, "lines": s.source_lines,
             "agent_class": s.agent_class,
             "agent": plan.by_step[s.id].name if s.id in plan.by_step else None}
            for s in steps
        ],
        "grounding": [
            {"agent": g.node, "step": g.step_id,
             "catalog_docs": [{"id": d, **{k: index.get(d, {}).get(k) for k in ("kind", "name")}}
                              for d in (g.nav.doc_ids if g.nav else [])],
             "procedure_section": g.procedure_section,
             "navigator_notes": g.nav.notes if g.nav else "",
             "navigator_turns": g.nav.turns if g.nav else 0,
             "gaps": g.gaps, "warnings": g.warnings}
            for g in groundings
        ],
        "procedure_tree": ({"dir": str(proc_tree.out_dir),
                            "sections": sum(1 for _ in proc_tree.root.walk()) - 1,
                            "documents": len(proc_tree.docs)} if proc_tree else None),
        "findings": [asdict(f) for f in findings],
        "errors": sum(f.severity == "ERROR" for f in findings),
        "warnings": sum(f.severity == "WARN" for f in findings)
                    + sum(len(g.warnings) for g in groundings),
        "llm_usage": getattr(getattr(client, "usage", None), "as_dict", lambda: {})(),
    }
    if compare_with is not None:
        print(f"[+] Comparing with existing YAMLs in {compare_with} ...")
        report["comparison"] = compare_dirs(Path(compare_with), out_dir)
    (out_dir.parent / f"{out_dir.name}_report.json").write_text(json.dumps(report, indent=2))
    (out_dir.parent / f"{out_dir.name}_steps.json").write_text(
        json.dumps(asdict(proc), indent=2))
    return report
