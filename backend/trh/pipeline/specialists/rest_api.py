from trh.pipeline.specialists.base import Specialist


class RestApiSpecialist(Specialist):
    name = "rest_api"
    agent_classes = frozenset({"rest_api_agent"})
    focus = (
        "call necessity, correct use of the response downstream, and whether "
        "this call duplicates an earlier one with identical input "
        "(evidence.duplicate_of_step_ids)"
    )
