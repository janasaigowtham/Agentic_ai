from trh.pipeline.specialists.base import Specialist


class LamDomainSpecialist(Specialist):
    name = "lam_domain"
    agent_classes = frozenset({"lam_verdict_synthesizer"})
    focus = (
        "whether the synthesized verdict is actually supported by the "
        "structured fields it was given, and whether it stayed within its "
        "declared domain rather than fabricating a classification"
    )
