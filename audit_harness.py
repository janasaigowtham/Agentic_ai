#!/usr/bin/env python3
"""
audit_harness.py — scans a repo for LiteLLM/OTEL/Splunk/Arize wiring and
generates a grounded audit prompt (real file paths + line numbers) instead
of a generic "go find X" template.

Usage:
    python3 audit_harness.py /path/to/repo [--output prompt.md] [--context 3]

Output:
    Prints the generated prompt to stdout and writes it to --output
    (default: audit_prompt.generated.md in the current directory).
"""

import argparse
import re
import sys
from pathlib import Path

SKIP_DIRS = {
    ".git", "node_modules", "venv", ".venv", "env", "dist", "build",
    "__pycache__", ".mypy_cache", ".pytest_cache", ".tox", "target",
    ".idea", ".vscode", "vendor",
}
MAX_FILE_BYTES = 2_000_000
TEXT_EXTENSIONS = {
    ".py", ".yaml", ".yml", ".json", ".toml", ".env", ".cfg", ".ini",
    ".md", ".txt", ".sh",
}

# category -> compiled regex patterns to search for (case-insensitive)
CATEGORIES = {
    "litellm_config": [
        r"litellm_settings", r"success_callback", r"failure_callback",
        r"\blitellm\.callbacks\b", r"router_settings",
    ],
    "otel_setup": [
        r"OTEL_EXPORTER_OTLP_\w*", r"OTLPSpanExporter", r"TracerProvider",
        r"BatchSpanProcessor", r"opentelemetry\.sdk",
    ],
    "arize": [
        r"\barize\b", r"ARIZE_SPACE\w*", r"ARIZE_API_KEY", r"openinference",
        r"arize\.otel",
    ],
    "splunk": [
        r"\bsplunk\b", r"\bHEC\b", r"hec_token", r"splunk_hec",
    ],
    "custom_logger": [
        r"CustomLogger", r"log_success_event", r"async_log_success_event",
        r"log_failure_event",
    ],
    "collector_config": [
        r"^receivers:", r"^exporters:", r"^processors:", r"otlp:\s*$",
    ],
    "existing_masking": [
        r"turn_off_message_logging", r"presidio", r"\bredact\w*",
        r"\bmask\w*\b", r"pii", r"\bscrub\w*",
    ],
}

CATEGORY_LABELS = {
    "litellm_config": "LiteLLM config / callback wiring",
    "otel_setup": "OpenTelemetry exporter/tracer setup",
    "arize": "Arize integration",
    "splunk": "Splunk integration",
    "custom_logger": "Custom LiteLLM logger classes",
    "collector_config": "OTel Collector config",
    "existing_masking": "Existing PII masking/redaction logic",
}


def iter_text_files(root: Path):
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.suffix.lower() not in TEXT_EXTENSIONS and path.name not in {
            "Dockerfile", ".env", ".env.example",
        }:
            continue
        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                continue
        except OSError:
            continue
        yield path


def scan_repo(root: Path, context: int):
    compiled = {
        cat: [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in pats]
        for cat, pats in CATEGORIES.items()
    }
    results = {cat: [] for cat in CATEGORIES}

    for path in iter_text_files(root):
        try:
            text = path.read_text(errors="ignore")
        except (OSError, UnicodeDecodeError):
            continue
        lines = text.splitlines()

        for cat, patterns in compiled.items():
            for pat in patterns:
                for m in pat.finditer(text):
                    line_no = text.count("\n", 0, m.start()) + 1
                    start = max(0, line_no - 1 - context)
                    end = min(len(lines), line_no + context)
                    snippet = "\n".join(
                        f"{i+1}: {lines[i]}" for i in range(start, end)
                    )
                    rel = path.relative_to(root)
                    results[cat].append((str(rel), line_no, snippet))
                    break  # one hit per pattern per file is enough signal
    return results


def dedupe(hits):
    seen = set()
    out = []
    for rel, line_no, snippet in hits:
        key = (rel, line_no)
        if key in seen:
            continue
        seen.add(key)
        out.append((rel, line_no, snippet))
    return out


def render_findings(results) -> str:
    blocks = []
    any_hits = False
    for cat, hits in results.items():
        hits = dedupe(hits)
        label = CATEGORY_LABELS[cat]
        if not hits:
            blocks.append(f"### {label}\n_No matches found in this repo._\n")
            continue
        any_hits = True
        lines = [f"### {label}"]
        for rel, line_no, snippet in hits[:8]:  # cap noise per category
            lines.append(f"\n**{rel}:{line_no}**\n```\n{snippet}\n```")
        if len(hits) > 8:
            lines.append(f"\n_...and {len(hits) - 8} more match(es) in this category._")
        blocks.append("\n".join(lines))
    if not any_hits:
        blocks.insert(
            0,
            "**No LiteLLM/OTEL/Arize/Splunk signal was found anywhere in this "
            "repo.** Either the wiring lives outside this tree (e.g. a proxy "
            "config repo, infra-as-code repo, or env vars set at deploy time) "
            "or the integration hasn't been added yet — say so explicitly "
            "rather than guessing.\n",
        )
    return "\n\n".join(blocks)


PROMPT_TEMPLATE = """\
Audit this repository for a PII/compliance issue in our LiteLLM observability pipeline.

## Background
LiteLLM is configured to export traces via OpenTelemetry. Both Splunk and Arize
currently receive data from the same OTLP exporter/endpoint, which means any PII
(SSNs, emails, etc.) present in LLM prompts/completions is being logged to Splunk
unmasked — a compliance problem. Arize needs full-fidelity, unmasked traces for
evaluation, so any fix must NOT degrade what Arize receives.

## Repo scan results (grounding — verify these, don't re-derive from scratch)
The following files/lines were found by a static scan of `{repo}`. Use these as
your starting points; confirm each is still accurate (the scan is regex-based
and can have false positives/negatives) before proposing a fix.

{findings}

## Step 1 — Confirm the actual wiring
- Using the matches above, confirm: are Splunk and Arize sharing one
  OTLPSpanExporter/endpoint, or going through a Collector that fans out?
- If "OTel Collector config" had no matches, check whether a collector config
  lives in a separate repo/infra directory — say so if you can't find it here.
- If "LiteLLM config / callback wiring" had no matches, this repo may not be
  where LiteLLM is invoked from — report that rather than fabricating findings.

## Step 2 — Identify what's actually being logged
- From the matched files, trace what ends up in span attributes: messages,
  prompts, completions, or request/response bodies set as span attributes
  (e.g. `gen_ai.prompt`, `gen_ai.completion`, `llm.input_messages.*`,
  `llm.output_messages.*`, or repo-specific attribute names).
- From "Existing PII masking/redaction logic" matches, determine whether any
  masking already exists, and whether it's global (would affect Arize too)
  or scoped correctly.
- Flag any other loggers/callbacks (Datadog, Langfuse, S3, custom webhooks)
  with the same shared-PII-leak risk — don't assume Splunk/Arize are the only two.

## Step 3 — Propose a fix scoped to this repo's actual architecture
Requirements:
- Splunk-bound trace data must have PII (SSNs, emails, and other sensitive
  patterns) masked before export.
- Arize-bound trace data must remain full-fidelity/unmasked.
- Prefer the smallest change consistent with what Step 1 found:
  - If Splunk and Arize already use separate LiteLLM callbacks, implement
    masking as a SpanProcessor or CustomLogger wrapper on only the
    Splunk-bound tracer — don't touch the Arize callback's code path.
  - If both share one exporter/endpoint with no Collector in between,
    recommend introducing a minimal OTel Collector (or splitting into two
    in-process tracers) rather than a broad global redaction flag.
  - Do not use `litellm.turn_off_message_logging` or a pre-call PII guardrail
    as the fix — both mutate/strip data for every destination, including Arize.
- Match this repo's existing code style/conventions (config format, naming,
  whether it uses YAML proxy config vs. SDK callbacks).

## Step 4 — Output
For each issue found, report:
1. File + line reference (from real files in this repo, not the scan snippets alone)
2. What's currently happening (short quote)
3. Why it's a compliance risk
4. A concrete proposed diff that fixes it without affecting the Arize pipeline
5. Any residual risk a human should verify

Do not modify files yet — report findings and proposed diffs first for review.
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("repo", type=Path, help="Path to the repo to scan")
    ap.add_argument(
        "--output", type=Path, default=Path("audit_prompt.generated.md"),
        help="Where to write the generated prompt (default: ./audit_prompt.generated.md)",
    )
    ap.add_argument(
        "--context", type=int, default=2,
        help="Lines of context around each match (default: 2)",
    )
    args = ap.parse_args()

    root = args.repo.resolve()
    if not root.is_dir():
        print(f"error: {root} is not a directory", file=sys.stderr)
        sys.exit(1)

    results = scan_repo(root, args.context)
    findings = render_findings(results)
    prompt = PROMPT_TEMPLATE.format(repo=root, findings=findings)

    args.output.write_text(prompt)
    print(prompt)
    print(f"\n\n[written to {args.output.resolve()}]", file=sys.stderr)


if __name__ == "__main__":
    main()
