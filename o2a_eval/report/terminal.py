"""Terminal scorecard renderer. Uses rich when importable, plain text otherwise."""
from __future__ import annotations

from ..core.models import Scorecard, Severity

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    HAS_RICH = True
except ImportError:  # pragma: no cover
    HAS_RICH = False

STATUS_COLORS = {"PASS": "green", "FLAGGED": "yellow", "FAIL": "red"}
SEVERITY_STYLES = {
    Severity.SEV_1: "bold red",
    Severity.SEV_2: "yellow",
    Severity.SEV_3: "cyan",
    Severity.INFO: "dim",
}


def _evidence_preview(evidence: dict) -> str:
    items = list((evidence or {}).items())[:3]
    preview = ", ".join(f"{k}={v}" for k, v in items)
    return preview[:160]


def render_scorecard(card: Scorecard, console=None) -> None:
    if HAS_RICH:
        _render_rich(card, console)
    else:
        _render_plain(card)


def _render_rich(card: Scorecard, console=None) -> None:
    console = console or Console()
    color = STATUS_COLORS.get(card.status, "white")

    console.rule(f"o2a-eval · {card.pipeline} · {card.trace_id[:12]}")

    lines = [
        f"[bold {color}]{card.status}[/bold {color}]  overall={card.overall:.2f}",
        f"duration={card.duration_s:.2f}s  spans={card.span_count}  hash={card.pipeline_hash}",
    ]
    if card.judge_calls_budget:
        lines.append(f"judge calls: {card.judge_calls_used}/{card.judge_calls_budget}")
    console.print(Panel("\n".join(lines), title="scorecard"))

    axis_table = Table(title="axes")
    axis_table.add_column("axis")
    axis_table.add_column("score")
    axis_table.add_column("")
    for axis, score in card.axes.items():
        bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
        axis_table.add_row(axis, f"{score:.2f}", bar)
    console.print(axis_table)

    if not card.flaws:
        console.print("[green]no flaws[/green]")
        return

    for flaw in card.flaws:
        style = SEVERITY_STYLES.get(flaw.severity, "white")
        console.print(
            f"[{style}]{flaw.severity.label}[/{style}] {flaw.check_id} {flaw.type} "
            f"node={flaw.node}"
        )
        console.print(f"  {flaw.description}")
        if flaw.fix:
            console.print(f"  fix: {flaw.fix}")
        preview = _evidence_preview(flaw.evidence)
        if preview:
            console.print(f"  evidence: {preview}")


def _render_plain(card: Scorecard) -> None:
    print(f"=== o2a-eval · {card.pipeline} · {card.trace_id[:12]} ===")
    print(f"status={card.status} overall={card.overall:.2f}")
    print(f"duration={card.duration_s:.2f}s spans={card.span_count} hash={card.pipeline_hash}")
    if card.judge_calls_budget:
        print(f"judge calls: {card.judge_calls_used}/{card.judge_calls_budget}")
    print("-- axes --")
    for axis, score in card.axes.items():
        bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
        print(f"{axis:16s} {score:.2f} {bar}")
    if not card.flaws:
        print("no flaws")
        return
    print("-- flaws --")
    for flaw in card.flaws:
        print(f"[{flaw.severity.label}] {flaw.check_id} {flaw.type} node={flaw.node}")
        print(f"  {flaw.description}")
        if flaw.fix:
            print(f"  fix: {flaw.fix}")
        preview = _evidence_preview(flaw.evidence)
        if preview:
            print(f"  evidence: {preview}")
