"""Prompt-quality checks: N1a-N1d, N2."""
from __future__ import annotations

import re

from ..core.graph import extract_template_keys
from ..core.models import Flaw, Severity
from ..core.registry import register

SECTION_HINTS = (
    "##", "###", "context", "instruction", "rules", "output",
    "format", "step", "input data", "evaluation", "critical",
)
OUTPUT_FORMAT_HINTS = ("json", "output format", "return", "respond with", "schema")
PARSING_CONSUMER_CLASSES = {"transformation_agent", "slv_transformation_agent", "decision_router_agent"}
_WORD_RE = re.compile(r"\w+")


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _ngrams(tokens: list[str], n: int = 8) -> set[tuple[str, ...]]:
    return {tuple(tokens[i : i + n]) for i in range(max(0, len(tokens) - n + 1))}


@register("N1a", axis="prompt_quality", stage="static")
def instruction_unknown_ref(run, graph, config):
    """Instruction references a session key nothing produces."""
    flaws: list[Flaw] = []
    for node in graph.nodes.values():
        if not node.is_llm or not node.instruction:
            continue
        refs = extract_template_keys(node.instruction)
        missing = sorted(
            k for k in (refs - set(graph.producers) - set(node.input_keys)) if "_" in k
        )
        if not missing:
            continue
        flaws.append(
            Flaw(
                type="instruction_unknown_ref",
                severity=Severity.SEV_1,
                node=node.name,
                description=f"{node.name}'s instruction references {missing} which nothing produces.",
                fix="Fix the key name, or add the producing node.",
                evidence={"missing_refs": missing, "known_producers": sorted(graph.producers)},
                agent_class=node.agent_class,
            )
        )
    return flaws


@register("N1b", axis="prompt_quality", stage="static")
def instruction_too_long(run, graph, config):
    """Instruction over token threshold."""
    threshold = config.get("prompt_token_threshold", 2000)
    flaws: list[Flaw] = []
    for node in graph.nodes.values():
        if not node.is_llm or not node.instruction:
            continue
        approx_tokens = _approx_tokens(node.instruction)
        if approx_tokens <= threshold:
            continue
        flaws.append(
            Flaw(
                type="instruction_too_long",
                severity=Severity.SEV_3,
                node=node.name,
                description=f"{node.name}'s instruction is ~{approx_tokens} tokens (threshold {threshold}).",
                fix="Trim the instruction, or move reference material to a retrieved document.",
                evidence={
                    "approx_tokens": approx_tokens,
                    "threshold": threshold,
                    "char_count": len(node.instruction),
                },
                agent_class=node.agent_class,
            )
        )
    return flaws


@register("N1c", axis="prompt_quality", stage="static")
def instruction_unstructured(run, graph, config):
    """Long instruction with no section structure."""
    flaws: list[Flaw] = []
    for node in graph.nodes.values():
        if not node.is_llm or not node.instruction:
            continue
        approx_tokens = _approx_tokens(node.instruction)
        if approx_tokens < 300:
            continue
        lowered = node.instruction.lower()
        section_hits = sum(1 for hint in SECTION_HINTS if hint in lowered)
        if section_hits >= 2:
            continue
        flaws.append(
            Flaw(
                type="instruction_unstructured",
                severity=Severity.SEV_3,
                node=node.name,
                description=f"{node.name}'s instruction is long (~{approx_tokens} tokens) with little structure.",
                fix="Add section headers (context, rules, output format) to make the prompt scannable.",
                evidence={"section_hits": section_hits, "approx_tokens": approx_tokens},
                agent_class=node.agent_class,
            )
        )
    return flaws


@register("N1d", axis="prompt_quality", stage="static")
def instruction_missing_output_format(run, graph, config):
    """Output parsed downstream but no output format stated."""
    flaws: list[Flaw] = []
    for node in graph.nodes.values():
        if not node.is_llm or not node.output_key:
            continue
        parsing_consumers = [
            c
            for c in graph.consumers.get(node.output_key, [])
            if c != node.name and graph.nodes.get(c) and graph.nodes[c].agent_class in PARSING_CONSUMER_CLASSES
        ]
        if not parsing_consumers:
            continue
        lowered = (node.instruction or "").lower()
        if any(hint in lowered for hint in OUTPUT_FORMAT_HINTS):
            continue
        flaws.append(
            Flaw(
                type="instruction_missing_output_format",
                severity=Severity.SEV_2,
                node=node.name,
                description=(
                    f"{node.name}'s output is parsed downstream but its instruction "
                    "never states an output format."
                ),
                fix="State the expected output format (e.g. strict JSON schema) in the instruction.",
                evidence={"output_key": node.output_key, "parsing_consumers": parsing_consumers},
                agent_class=node.agent_class,
            )
        )
    return flaws


@register("N2", axis="prompt_quality", stage="static")
def instruction_repeated_block(run, graph, config):
    """Repeated instruction block across LlmAgent prompts."""
    threshold = config.get("instruction_overlap_threshold", 0.25)
    llm_nodes = [n for n in graph.nodes.values() if n.is_llm and n.instruction]

    tokenized = {}
    for node in llm_nodes:
        tokens = _WORD_RE.findall(node.instruction.lower())
        tokenized[node.name] = tokens

    flaws: list[Flaw] = []
    seen_pairs = set()
    for i, a in enumerate(llm_nodes):
        for b in llm_nodes[i + 1 :]:
            pair = tuple(sorted((a.name, b.name)))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            tokens_a, tokens_b = tokenized[a.name], tokenized[b.name]
            if len(tokens_a) < 50 or len(tokens_b) < 50:
                continue
            ngrams_a, ngrams_b = _ngrams(tokens_a), _ngrams(tokens_b)
            if not ngrams_a or not ngrams_b:
                continue
            shared = ngrams_a & ngrams_b
            overlap_ratio = len(shared) / min(len(ngrams_a), len(ngrams_b))
            if overlap_ratio <= threshold:
                continue
            flaws.append(
                Flaw(
                    type="instruction_repeated_block",
                    severity=Severity.SEV_3,
                    node=a.name,
                    description=(
                        f"{a.name} and {b.name} share a near-identical instruction block "
                        f"(overlap {overlap_ratio:.0%})."
                    ),
                    fix="Factor the shared block into a system-level prompt fragment.",
                    evidence={
                        "node_a": a.name,
                        "node_b": b.name,
                        "overlap_ratio": round(overlap_ratio, 4),
                        "shared_ngrams": len(shared),
                    },
                    agent_class=a.agent_class,
                )
            )
    return flaws
