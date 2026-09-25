"""Synthetic seed data for the mock source systems.

Multiple loan_number scenarios exercise different branches of the declared
pipeline (ICMP required vs. skipped, investor record present vs. absent,
a careful gate review vs. a rubber-stamped one) so a review isn't limited
to the single hand-authored fixture trajectory in fixtures/trace.jsonl.
"""
from __future__ import annotations

from dataclasses import dataclass

from trh.execution.mock_db import MockDatabase


@dataclass(frozen=True)
class LoanScenario:
    loan_number: str
    label: str
    inv_class_code: str
    letter_effective_date: str | None
    borrower_name: str
    has_investor_record: bool
    investor_id: str | None
    investor_name: str | None
    gate_duration_s: float


LOAN_SCENARIOS: dict[str, LoanScenario] = {
    "1234567890": LoanScenario(
        loan_number="1234567890",
        label="ICMP required, investor present, gate reviewed carefully",
        inv_class_code="COMM",
        letter_effective_date="2024-01-15",
        borrower_name="Jane Doe",
        has_investor_record=True,
        investor_id="INV001",
        investor_name="Acme Capital",
        gate_duration_s=42.0,
    ),
    "2222222222": LoanScenario(
        loan_number="2222222222",
        label="ICMP not required, investor present",
        inv_class_code="RES",
        letter_effective_date=None,
        borrower_name="Marcus Lee",
        has_investor_record=True,
        investor_id="INV002",
        investor_name="Riverside Partners",
        gate_duration_s=38.0,
    ),
    "3333333333": LoanScenario(
        loan_number="3333333333",
        label="ICMP required, no investor record, gate rubber-stamped",
        inv_class_code="COMM",
        letter_effective_date="2023-11-02",
        borrower_name="Priya Nair",
        has_investor_record=False,
        investor_id=None,
        investor_name=None,
        gate_duration_s=0.08,
    ),
    "4444444444": LoanScenario(
        loan_number="4444444444",
        label="ICMP not required, no investor record",
        inv_class_code="GOVT",
        letter_effective_date=None,
        borrower_name="Oscar Ibarra",
        has_investor_record=False,
        investor_id=None,
        investor_name=None,
        gate_duration_s=51.0,
    ),
}


def build_mock_database() -> MockDatabase:
    db = MockDatabase()
    loan_rows = [
        {
            "ln_no": s.loan_number,
            "inv_class_code": s.inv_class_code,
            "letter_effective_date": s.letter_effective_date,
            "borrower_name": s.borrower_name,
        }
        for s in LOAN_SCENARIOS.values()
    ]
    investor_rows = [
        {"loan_number": s.loan_number, "investor_id": s.investor_id, "investor_name": s.investor_name}
        for s in LOAN_SCENARIOS.values()
        if s.has_investor_record
    ]
    db.seed_table("MSP_LOAN_MASTER7_CS", loan_rows, key_column="ln_no")
    db.seed_table("investor_master", investor_rows, key_column="loan_number")
    return db
