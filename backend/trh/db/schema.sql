-- Trajectory Review Harness demo schema. stdlib sqlite3 only, per the harness's
-- full-isolation principle: this is the harness's own storage, nothing shared
-- with a target system's execution tables.

CREATE TABLE IF NOT EXISTS trajectories (
    trajectory_id     TEXT PRIMARY KEY,
    source_pipeline   TEXT NOT NULL,
    source_spec_ref   TEXT,
    started_at        REAL NOT NULL,
    completed_at      REAL,
    outcome           TEXT
);

CREATE TABLE IF NOT EXISTS steps (
    step_id                TEXT PRIMARY KEY,
    trajectory_id          TEXT NOT NULL REFERENCES trajectories(trajectory_id),
    seq                    INTEGER NOT NULL,
    step_type              TEXT NOT NULL,
    agent_name             TEXT NOT NULL,
    agent_class            TEXT NOT NULL,
    input                  TEXT,
    output                 TEXT,
    started_at             REAL NOT NULL,
    completed_at           REAL NOT NULL,
    parent_step_id         TEXT,
    declared_contract_ref  TEXT,
    tap                    TEXT NOT NULL,
    evidence               TEXT
);
CREATE INDEX IF NOT EXISTS idx_steps_trajectory ON steps(trajectory_id);

CREATE TABLE IF NOT EXISTS case_summaries (
    trajectory_id   TEXT PRIMARY KEY REFERENCES trajectories(trajectory_id),
    goal            TEXT NOT NULL,
    plan_declared   TEXT,
    plan_actual     TEXT NOT NULL,
    outcome         TEXT,
    step_type_map   TEXT NOT NULL,
    orient_notes    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS verdicts (
    verdict_id      TEXT PRIMARY KEY,
    trajectory_id   TEXT NOT NULL REFERENCES trajectories(trajectory_id),
    specialist      TEXT NOT NULL,
    step_ids        TEXT NOT NULL,
    trigger_event   TEXT NOT NULL,
    finding         TEXT NOT NULL,
    severity        TEXT NOT NULL,
    evidence        TEXT
);
CREATE INDEX IF NOT EXISTS idx_verdicts_trajectory ON verdicts(trajectory_id);

CREATE TABLE IF NOT EXISTS recommendations (
    recommendation_id   TEXT PRIMARY KEY,
    trajectory_id       TEXT NOT NULL REFERENCES trajectories(trajectory_id),
    tied_to_root_cause  INTEGER NOT NULL,
    root_cause_chain    TEXT NOT NULL,
    description         TEXT NOT NULL,
    proposed_fix        TEXT NOT NULL,
    status              TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_recommendations_trajectory ON recommendations(trajectory_id);

CREATE TABLE IF NOT EXISTS gate_decisions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    recommendation_id   TEXT NOT NULL REFERENCES recommendations(recommendation_id),
    decision            TEXT NOT NULL,
    reviewer            TEXT NOT NULL,
    note                TEXT,
    timestamp           TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_gate_decisions_recommendation ON gate_decisions(recommendation_id);

CREATE TABLE IF NOT EXISTS calibration_records (
    case_id         TEXT PRIMARY KEY,
    trajectory_id   TEXT NOT NULL REFERENCES trajectories(trajectory_id),
    harness_verdict TEXT,
    human_label     TEXT,
    agreement       INTEGER NOT NULL,
    reviewed_by     TEXT NOT NULL,
    timestamp       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS fact_checks (
    trajectory_id   TEXT PRIMARY KEY REFERENCES trajectories(trajectory_id),
    confidence      TEXT NOT NULL,
    notes           TEXT NOT NULL,
    routed_back     INTEGER NOT NULL
);

-- Not part of the spec's data model: status-tracking for the API's async
-- trigger/poll pattern (POST .../review kicks this off, GET .../reviews/{id} polls it).
CREATE TABLE IF NOT EXISTS reviews (
    review_id           TEXT PRIMARY KEY,
    trajectory_id       TEXT,
    status              TEXT NOT NULL,
    error               TEXT,
    stage_lineage_used  TEXT,
    created_at          TEXT NOT NULL,
    updated_at          TEXT NOT NULL
);
