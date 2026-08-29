"""o2a-eval: an evaluator for O2A agent pipelines."""
from . import checks as _checks  # noqa: F401 - registers checks on import
from .adapters.base import FrameworkAdapter
from .capture.attach import attach
from .core.graph import load_pipeline
from .core.models import Flaw, PipelineGraph, Run, Scorecard, Severity
from .core.rollup import build_scorecard
from .core.trace import build_run, load_jsonl

__all__ = [
    "attach",
    "FrameworkAdapter",
    "load_pipeline",
    "build_run",
    "load_jsonl",
    "build_scorecard",
    "Scorecard",
    "Flaw",
    "Severity",
    "Run",
    "PipelineGraph",
]
