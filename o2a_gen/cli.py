"""o2a-gen command line.

  python -m o2a_gen compile-catalog --config gen_config.yaml --src catalog_src --out catalog_build
  python -m o2a_gen generate --config gen_config.yaml --procedure proc.md --catalog catalog_build \\
      --out generated/pmi_ddn [--compare-with pipelines/pmi_ddn]
  python -m o2a_gen validate generated/pmi_ddn --inputs loan_number
  python -m o2a_gen compare --gold pipelines/pmi_ddn --generated generated/pmi_ddn
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="o2a_gen", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("compile-catalog", help="compile agent syntax, metadata, tools and reference docs into a skill tree")
    c.add_argument("--config", type=Path, required=True)
    c.add_argument("--src", type=Path, required=True)
    c.add_argument("--out", type=Path, required=True)

    g = sub.add_parser("generate", help="procedure document -> agent YAMLs")
    g.add_argument("--config", type=Path, required=True)
    g.add_argument("--procedure", type=Path, required=True)
    g.add_argument("--catalog", type=Path, help="compiled catalog dir (omit to generate without grounding)")
    g.add_argument("--out", type=Path, required=True)
    g.add_argument("--workers", type=int, default=4)
    g.add_argument("--overwrite", action="store_true")
    g.add_argument("--compare-with", type=Path,
                   help="existing YAMLs for this procedure; compared only after generation")

    v = sub.add_parser("validate", help="static checks on a YAML directory")
    v.add_argument("dir", type=Path)
    v.add_argument("--inputs", nargs="*", default=[], help="session keys supplied by the caller")
    v.add_argument("--catalog", type=Path, help="compiled catalog, to check fields against the agent syntax")

    k = sub.add_parser("compare", help="score generated YAMLs against a hand-built set")
    k.add_argument("--gold", type=Path, required=True)
    k.add_argument("--generated", type=Path, required=True)
    k.add_argument("--json", type=Path, help="also write the full comparison here")

    args = ap.parse_args(argv)

    if args.cmd in ("compile-catalog", "generate"):
        from o2a_gen.config import load_config
        from o2a_gen.llm import build_client
        cfg = load_config(args.config)
        client = build_client(cfg.llm)

    if args.cmd == "compile-catalog":
        from o2a_gen.catalog import compile_catalog
        skills = compile_catalog(args.src, args.out, client, cfg)
        print(f"catalog skill tree: {skills}")
        return 0

    if args.cmd == "generate":
        from o2a_gen.generate import generate
        rep = generate(args.procedure, args.out, cfg, client, catalog_dir=args.catalog,
                       workers=args.workers, overwrite=args.overwrite,
                       compare_with=args.compare_with)
        for f in rep["findings"]:
            print(f"  {f['severity']:5} {f['code']:14} {f['agent']}: {f['message']}")
        for gr in rep["grounding"]:
            for w in gr["warnings"]:
                print(f"  WARN  grounding      {w}")
        if "comparison" in rep:
            print("\nComparison with existing YAMLs:")
            _print_comparison(rep["comparison"])
        print(f"\n{rep['agents']} agents, {rep['errors']} errors, {rep['warnings']} warnings. "
              f"Report: {args.out.parent / (args.out.name + '_report.json')}")
        return 1 if rep["errors"] else 0

    if args.cmd == "validate":
        from o2a_gen.validate import validate_dir
        syntax = None
        if args.catalog:
            import yaml

            from o2a_gen.navigator import Catalog
            cat = Catalog(args.catalog)
            classes = {yaml.safe_load(p.read_text()).get("agent_class") for p in args.dir.glob("*.y*ml")}
            syntax = {c: cat.syntax_for(c) for c in classes if c}
        findings = validate_dir(args.dir, args.inputs, syntax)
        for f in findings:
            print(f)
        errors = sum(f.severity == "ERROR" for f in findings)
        print(f"\n{errors} errors, {len(findings) - errors} warnings")
        return 1 if errors else 0

    if args.cmd == "compare":
        from o2a_gen.compare import compare_dirs
        res = compare_dirs(args.gold, args.generated)
        _print_comparison(res)
        if args.json:
            args.json.write_text(json.dumps(res, indent=2))
        return 0
    return 2


def _print_comparison(res: dict) -> None:
    for key in ("gold_agents", "generated_agents", "matched", "coverage", "class_agreement",
                "output_key_agreement", "mean_input_keys_jaccard", "db_table_agreement",
                "missed_gold_agents", "extra_generated_agents"):
        print(f"  {key:26} {res[key]}")


if __name__ == "__main__":
    sys.exit(main())
