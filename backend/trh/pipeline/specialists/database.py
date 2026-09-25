from trh.pipeline.specialists.base import Specialist


class DatabaseSpecialist(Specialist):
    name = "database"
    agent_classes = frozenset({"database_agent"})
    focus = (
        "query necessity and correctness (evidence.resolved_query / "
        "evidence.query_params), whether this call duplicates an earlier one "
        "with identical input (evidence.duplicate_of_step_ids), and whether "
        "retrieved columns are ever used downstream (evidence.unused_columns)"
    )
