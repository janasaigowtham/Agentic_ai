from trh.core.harness_models import CaseSummary, Step, Trajectory
from trh.pipeline.specialists.base import Specialist


class OrchestrationPlanFidelitySpecialist(Specialist):
    name = "orchestration_plan_fidelity"
    agent_classes = frozenset({"resumable_orchestrator", "SequentialAgent", "FailFastLoopAgent"})
    focus = (
        "whether execution matched the declared order, whether a declared "
        "step was skipped entirely, and whether the step's total duration is "
        "reasonable against what a pipeline of this shape should take"
    )

    def build_user_prompt(self, step: Step, trajectory: Trajectory, case_summary: CaseSummary) -> str:
        base = super().build_user_prompt(step, trajectory, case_summary)
        duration_s = step.completed_at - step.started_at
        children = [s for s in trajectory.steps if s.parent_step_id == step.step_id]
        children_summary = [
            {"agent_name": c.agent_name, "started_at": c.started_at, "completed_at": c.completed_at}
            for c in children
        ]
        return (
            base
            + f"\nTotal duration: {duration_s:.3f}s\n"
            + f"Direct children, in actual execution order: {children_summary!r}\n"
        )
