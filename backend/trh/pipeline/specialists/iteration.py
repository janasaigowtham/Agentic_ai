from trh.pipeline.specialists.base import Specialist


class IterationSpecialist(Specialist):
    name = "iteration"
    agent_classes = frozenset({"iteration_agent"})
    focus = (
        "whether the loop's exit condition is sound (no risk of running "
        "forever or exiting before its declared work is done) and whether "
        "each iteration's result was actually used by the next one"
    )
