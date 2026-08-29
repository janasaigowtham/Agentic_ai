"""attach(): observer-only integration. Never drives the pipeline, only watches.

opentelemetry is an optional dependency (the `otel` extra). Everything in this
module that touches it does so lazily, inside function bodies, so importing
o2a_eval (and running `lint`/`graph`/`checks`) never requires it.
"""
from __future__ import annotations

import atexit
import functools
import importlib
from typing import Any

from ..adapters.base import FrameworkAdapter, set_post_attributes, set_pre_attributes
from .ephemeral import EphemeralTraceStore

_STATE: dict = {"attached": False}


def _import_class(path: str):
    module_path, _, cls_name = path.rpartition(".")
    module = importlib.import_module(module_path)
    cls = getattr(module, cls_name)
    return module, cls_name, cls


def _serialize(s) -> dict:
    ctx = s.get_span_context()
    return {
        "trace_id": format(ctx.trace_id, "032x"),
        "span_id": format(ctx.span_id, "016x"),
        "parent_id": format(s.parent.span_id, "016x") if s.parent else None,
        "name": s.name,
        "start_ns": s.start_time,
        "end_ns": s.end_time,
        "status": s.status.status_code.name if s.status else "UNSET",
        "attributes": dict(s.attributes or {}),
        "events": [
            {"name": e.name, "attributes": dict(e.attributes or {}), "ts": e.timestamp}
            for e in (s.events or [])
        ],
    }


def _make_exporter_class():
    from opentelemetry.sdk.trace.export import SpanExportResult, SpanExporter

    class EphemeralExporter(SpanExporter):
        def __init__(self, store: EphemeralTraceStore):
            self.store = store

        def export(self, spans):
            for s in spans:
                self.store.add(_serialize(s))
            return SpanExportResult.SUCCESS

        def shutdown(self):
            pass

        def force_flush(self, timeout_millis: int = 30000) -> bool:
            return True

    return EphemeralExporter


def install_enrichment(adapter: FrameworkAdapter) -> None:
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode

    tracer = trace.get_tracer("o2a_eval")

    module, cls_name, base_cls = _import_class(adapter.base_class_path)
    original = getattr(base_cls, adapter.run_method)
    if getattr(original, "_o2a_wrapped", False):
        return

    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        session: dict = {}
        try:
            session = adapter.get_session((self,) + args, kwargs) or {}
        except Exception:
            pass
        keys_before = set(session.keys())
        name = adapter.get_agent_name(self)

        with tracer.start_as_current_span(name) as span:
            try:
                set_pre_attributes(span, self, session, adapter)
            except Exception:
                pass
            try:
                result = original(self, *args, **kwargs)
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise
            try:
                set_post_attributes(span, self, session, keys_before, result, adapter)
            except Exception:
                pass
            return result

    wrapped._o2a_wrapped = True
    setattr(base_cls, adapter.run_method, wrapped)


def attach(adapter: FrameworkAdapter, config: dict | None = None, evaluate_on_exit: bool = True) -> EphemeralTraceStore:
    if _STATE["attached"]:
        return _STATE["store"]

    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    config = dict(config or {})
    store = EphemeralTraceStore(mode=config.get("storage_mode", "memory"))

    provider = trace.get_tracer_provider()
    if not isinstance(provider, TracerProvider):
        provider = TracerProvider()
        trace.set_tracer_provider(provider)

    exporter_cls = _make_exporter_class()
    provider.add_span_processor(BatchSpanProcessor(exporter_cls(store)))

    install_enrichment(adapter)
    _STATE.update(attached=True, store=store, config=config)
    if evaluate_on_exit:
        atexit.register(_on_exit)
    return store


def _on_exit() -> None:
    try:
        from opentelemetry import trace

        trace.get_tracer_provider().force_flush()

        store: EphemeralTraceStore = _STATE["store"]
        if not len(store):
            return
        config = _STATE["config"]

        from ..core.graph import load_pipeline
        from ..core.rollup import build_scorecard
        from ..core.trace import build_run
        from ..report.scrubber import write_scorecard
        from ..report.terminal import render_scorecard

        run = build_run(store.read_all())
        graph = load_pipeline(config["agent_dir"], config.get("pipeline") or run.pipeline_name)
        card = build_scorecard(run, graph, config)
        render_scorecard(card)
        if config.get("output_dir"):
            write_scorecard(card, run, config["output_dir"])
    except Exception as e:  # noqa: BLE001 - evaluation must never break process exit
        print(f"[o2a-eval] evaluation skipped: {type(e).__name__}: {e}")
    finally:
        try:
            _STATE["store"].cleanup()
        except Exception:
            pass
