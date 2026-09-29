"""End-to-end: procedure document -> validated O2A agent YAMLs + a review report."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from o2a_gen.config import GenConfig
from o2a_gen.emit import write_plan
from o2a_gen.ground import ground_plan
from o2a_gen.llm import LLMClient
from o2a_gen.navigator import Catalog
from o2a_gen.plan import build_plan
from o2a_gen.procedure import extract_procedure, load_procedure_text
from o2a_gen.validate import validate_dir


def generate(procedure_path: Path, out_dir: Path, cfg: GenConfig, client: LLMClient, *,
             catalog_dir: Path | None = None, workers: int = 4, overwrite: bool = False) -> dict:
    procedure_path, out_dir = Path(procedure_path), Path(out_dir)
    old = list(out_dir.glob("*.yaml")) if out_dir.exists() else []
    if old and not overwrite:
        raise FileExistsError(f"{out_dir} already has {len(old)} YAML files; use --overwrite")
    for p in old:
        p.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] Reading procedure {procedure_path.name} ...")
    proc = extract_procedure(client, load_procedure_text(procedure_path),
                             model=cfg.model("extract"), source=procedure_path.name)
    steps = proc.all_steps()
    print(f"      {len(proc.phases)} phases, {len(steps)} steps")

    print("[2/5] Planning agents ...")
    plan = build_plan(proc, cfg)
    print(f"      {len(plan.nodes())} agents")

    catalog = Catalog(catalog_dir) if catalog_dir else None
    print(f"[3/5] Grounding each step {'in the catalog' if catalog else '(no catalog)'} ...")
    groundings = ground_plan(client, plan, cfg, catalog, max_workers=workers)

    print(f"[4/5] Writing YAMLs to {out_dir} ...")
    write_plan(plan, cfg, out_dir, procedure_path.name)

    print("[5/5] Validating ...")
    findings = validate_dir(out_dir, plan.inputs)

    index = catalog.index if catalog else {}
    report = {
        "procedure": procedure_path.name,
        "pipeline": plan.root.name,
        "pipeline_inputs": plan.inputs,
        "agents": len(plan.nodes()),
        "coverage": [
            {"step": s.id, "title": s.title, "lines": s.source_lines, "kind": s.kind,
             "agent": plan.by_step[s.id].name if s.id in plan.by_step else None}
            for s in steps
        ],
        "grounding": [
            {"agent": g.node, "step": g.step_id,
             "catalog_docs": [{"id": d, **{k: index.get(d, {}).get(k) for k in ("kind", "name")}}
                              for d in (g.nav.doc_ids if g.nav else [])],
             "navigator_notes": g.nav.notes if g.nav else "",
             "navigator_turns": g.nav.turns if g.nav else 0,
             "gaps": g.gaps, "warnings": g.warnings}
            for g in groundings
        ],
        "findings": [asdict(f) for f in findings],
        "errors": sum(f.severity == "ERROR" for f in findings),
        "warnings": sum(f.severity == "WARN" for f in findings)
                    + sum(len(g.warnings) for g in groundings),
        "llm_usage": getattr(getattr(client, "usage", None), "as_dict", lambda: {})(),
    }
    (out_dir.parent / f"{out_dir.name}_report.json").write_text(json.dumps(report, indent=2))
    (out_dir.parent / f"{out_dir.name}_steps.json").write_text(
        json.dumps(asdict(proc), indent=2))
    return report
