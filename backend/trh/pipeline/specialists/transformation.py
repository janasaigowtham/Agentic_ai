from trh.pipeline.specialists.base import Specialist


class TransformationSpecialist(Specialist):
    name = "transformation"
    agent_classes = frozenset({"transformation_agent", "slv_transformation_agent"})
    focus = (
        "whether the transform's declared input_keys actually cover what it "
        "needed, and whether its output was ever consumed downstream "
        "(evidence.unused_columns)"
    )
