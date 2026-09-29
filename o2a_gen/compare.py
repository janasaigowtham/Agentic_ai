"""Scores generated YAMLs against a hand-built ("gold") set for the same procedure."""

from __future__ import annotations

import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

from o2a_gen.validate import db_block, input_keys, load_agents

_TABLE = re.compile(r"\b(?:from|join)\s+([A-Za-z_][\w.]*)", re.I)


def tables_in(a: dict) -> set[str]:
    return {t.upper() for t in _TABLE.findall(str(db_block(a).get("query") or ""))}


def _match(gold: dict[str, dict], gen: dict[str, dict]) -> dict[str, str]:
    """gold name -> generated name. Exact name, then same output_key+class, then similar name."""
    pairs: dict[str, str] = {}
    free = set(gen)
    for g in gold:
        if g in free:
            pairs[g] = g
            free.discard(g)
    by_key = {}
    for n in free:
        a = gen[n]
        if a.get("output_key") and a.get("agent_class") != "decision_router_agent":
            by_key.setdefault((a["output_key"], a.get("agent_class")), n)
    for g, a in gold.items():
        if g in pairs:
            continue
        n = by_key.get((a.get("output_key"), a.get("agent_class")))
        if n in free:
            pairs[g] = n
            free.discard(n)
    candidates = []
    for g, a in gold.items():
        if g in pairs:
            continue
        for n in free:
            if gen[n].get("agent_class") == a.get("agent_class"):
                candidates.append((SequenceMatcher(None, g, n).ratio(), g, n))
    for score, g, n in sorted(candidates, reverse=True):
        if score < 0.6:
            break
        if g not in pairs and n in free:
            pairs[g] = n
            free.discard(n)
    return pairs


def _jaccard(a: set, b: set) -> float:
    return 1.0 if not a and not b else len(a & b) / len(a | b)


def compare_dirs(gold_dir: Path, gen_dir: Path) -> dict:
    gold, _ = load_agents(gold_dir)
    gen, _ = load_agents(gen_dir)
    pairs = _match(gold, gen)
    rows = []
    for g, n in sorted(pairs.items()):
        ga, na = gold[g], gen[n]
        row = {
            "gold": g, "generated": n,
            "class_match": ga.get("agent_class") == na.get("agent_class"),
            "output_key_match": ga.get("output_key") == na.get("output_key"),
            "input_keys_jaccard": round(_jaccard(set(input_keys(ga)), set(input_keys(na))), 2),
        }
        if ga.get("agent_class") == "database_agent":
            gt, nt = tables_in(ga), tables_in(na)
            row["tables_gold"], row["tables_generated"] = sorted(gt), sorted(nt)
            row["tables_match"] = gt == nt
        if ga.get("agent_class") == "decision_router_agent":
            row["routes_gold"] = len(ga.get("routes") or [])
            row["routes_generated"] = len(na.get("routes") or [])
        rows.append(row)
    n = len(rows) or 1
    db_rows = [r for r in rows if "tables_match" in r]
    return {
        "gold_agents": len(gold),
        "generated_agents": len(gen),
        "matched": len(rows),
        "coverage": round(len(rows) / (len(gold) or 1), 3),
        "class_agreement": round(sum(r["class_match"] for r in rows) / n, 3),
        "output_key_agreement": round(sum(r["output_key_match"] for r in rows) / n, 3),
        "mean_input_keys_jaccard": round(sum(r["input_keys_jaccard"] for r in rows) / n, 3),
        "db_table_agreement": (round(sum(r["tables_match"] for r in db_rows) / len(db_rows), 3)
                               if db_rows else None),
        "class_counts_gold": dict(Counter(a.get("agent_class") for a in gold.values())),
        "class_counts_generated": dict(Counter(a.get("agent_class") for a in gen.values())),
        "missed_gold_agents": sorted(set(gold) - set(pairs)),
        "extra_generated_agents": sorted(set(gen) - set(pairs.values())),
        "pairs": rows,
    }
