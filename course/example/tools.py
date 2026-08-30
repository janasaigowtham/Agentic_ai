"""Example tools: deterministic, side-effect-free stand-ins for real
integrations (an order database, a payments API). See
course/03-tools-and-function-calling.md for why they're kept this narrow."""
from __future__ import annotations

_ORDERS = {
    "A1001": {"order_id": "A1001", "total": 42.50, "item": "wireless mouse"},
    "A1002": {"order_id": "A1002", "total": 310.00, "item": "monitor"},
    "A1003": {"order_id": "A1003", "total": 1500.00, "item": "laptop"},
}


def lookup_order(order_id: str) -> dict:
    """Look up an order by id.

    Returns {order_id, total, item}; an unknown id returns a zero-total
    placeholder rather than raising, so a bad id becomes a normal branch in
    the pipeline instead of a crash the model has to improvise around.
    """
    return _ORDERS.get(order_id, {"order_id": order_id, "total": 0.0, "item": "unknown"})


def calculate_refund(order: dict, reason: str) -> float:
    """Compute the refund amount for an order. Full refund for now - a real
    implementation would apply policy (partial refunds, restocking fees)."""
    if not order:
        return 0.0
    return round(order.get("total", 0.0), 2)
