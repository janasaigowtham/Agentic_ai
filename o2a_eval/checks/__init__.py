"""Import every check module so their @register decorators fire."""
from . import database, hygiene, judge, lifecycle, prompts, router, structural  # noqa: F401
