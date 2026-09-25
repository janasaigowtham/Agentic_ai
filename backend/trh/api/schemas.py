from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel


class GateDecisionRequest(BaseModel):
    decision: Literal["approved", "declined", "hold"]
    reviewer: str
    note: Optional[str] = None
