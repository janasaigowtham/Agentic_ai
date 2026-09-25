from trh.core.harness_models import CaseSummary, Step, Trajectory
from trh.pipeline.specialists.base import Specialist


class DecisionRoutingSpecialist(Specialist):
    name = "decision_routing"
    agent_classes = frozenset({"decision_router_agent"})
    focus = (
        "whether the evaluated routes cover the condition space completely "
        "-- e.g. eq-only routing with no declared default -- and whether the "
        "chosen target matches the declared routing table"
    )

    def build_user_prompt(self, step: Step, trajectory: Trajectory, case_summary: CaseSummary) -> str:
        base = super().build_user_prompt(step, trajectory, case_summary)
        span = step.raw_span
        routes = span.evaluated_routes if span else []
        chosen = span.chosen_target if span else None
        return base + f"\nEvaluated routes: {routes!r}\nChosen target: {chosen!r}\n"
