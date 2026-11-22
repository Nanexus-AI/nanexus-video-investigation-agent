"""Investigation Persistence schema. Distinct from LangGraph checkpoint tables."""

SCHEMA_VERSION = 3

# Application tables use the nx_ prefix so they never collide with native
# Checkpointer tables such as ``checkpoints`` and ``writes``.
DDL = f"""
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS nx_schema_meta (
    key TEXT PRIMARY KEY NOT NULL,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS nx_cases (
    case_id TEXT PRIMARY KEY NOT NULL,
    status TEXT NOT NULL,
    question TEXT NOT NULL,
    policy_ref TEXT NOT NULL,
    created_at TEXT NOT NULL,
    vms_instance_id TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    payload_fingerprint TEXT NOT NULL,
    schema_version INTEGER NOT NULL DEFAULT {SCHEMA_VERSION}
);

CREATE TABLE IF NOT EXISTS nx_operations (
    operation_id TEXT PRIMARY KEY NOT NULL,
    case_id TEXT NOT NULL REFERENCES nx_cases(case_id),
    plan_step_id TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    content_fingerprint TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (case_id, plan_step_id, content_fingerprint)
);

CREATE TABLE IF NOT EXISTS nx_artifacts (
    artifact_kind TEXT NOT NULL,
    artifact_id TEXT NOT NULL,
    operation_id TEXT NOT NULL REFERENCES nx_operations(operation_id),
    case_id TEXT NOT NULL REFERENCES nx_cases(case_id),
    payload_json TEXT NOT NULL,
    payload_fingerprint TEXT NOT NULL,
    native_record_id TEXT,
    vms_instance_id TEXT,
    created_at TEXT NOT NULL,
    PRIMARY KEY (artifact_kind, artifact_id)
);

CREATE INDEX IF NOT EXISTS ix_nx_artifacts_case
    ON nx_artifacts(case_id, artifact_kind);

CREATE INDEX IF NOT EXISTS ix_nx_artifacts_operation
    ON nx_artifacts(operation_id, artifact_kind);

CREATE TABLE IF NOT EXISTS nx_registration_groups (
    operation_id TEXT PRIMARY KEY NOT NULL REFERENCES nx_operations(operation_id),
    case_id TEXT NOT NULL REFERENCES nx_cases(case_id),
    group_fingerprint TEXT NOT NULL,
    member_kinds TEXT NOT NULL,
    receipt_id TEXT NOT NULL,
    observation_id TEXT,
    evidence_id TEXT,
    status TEXT NOT NULL CHECK (status = 'committed'),
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_nx_registration_groups_case
    ON nx_registration_groups(case_id);

CREATE TABLE IF NOT EXISTS nx_budget_policies (
    case_id TEXT PRIMARY KEY NOT NULL REFERENCES nx_cases(case_id),
    max_tool_attempts INTEGER NOT NULL CHECK (max_tool_attempts >= 0),
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS nx_budget_reservations (
    reservation_id TEXT PRIMARY KEY NOT NULL,
    case_id TEXT NOT NULL REFERENCES nx_cases(case_id),
    operation_id TEXT NOT NULL,
    attempt_id TEXT NOT NULL,
    units INTEGER NOT NULL CHECK (units > 0),
    status TEXT NOT NULL CHECK (
        status IN ('reserved', 'confirmed', 'failed', 'unresolved')
    ),
    payload_fingerprint TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    execution_id TEXT,
    UNIQUE (case_id, attempt_id)
);

CREATE INDEX IF NOT EXISTS ix_nx_budget_case_status
    ON nx_budget_reservations(case_id, status);
"""

# Additive statements for upgrading schema_version 1 → 2.
DDL_V2_UPGRADE = """
CREATE TABLE IF NOT EXISTS nx_registration_groups (
    operation_id TEXT PRIMARY KEY NOT NULL REFERENCES nx_operations(operation_id),
    case_id TEXT NOT NULL REFERENCES nx_cases(case_id),
    group_fingerprint TEXT NOT NULL,
    member_kinds TEXT NOT NULL,
    receipt_id TEXT NOT NULL,
    observation_id TEXT,
    evidence_id TEXT,
    status TEXT NOT NULL CHECK (status = 'committed'),
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_nx_registration_groups_case
    ON nx_registration_groups(case_id);
"""

# Schema v3: allow multiple observation/evidence artifacts per logical operation.
# Prior UNIQUE(operation_id, artifact_kind) silently blocked multi-event tool results.
DDL_V3_UPGRADE = """
PRAGMA foreign_keys = OFF;

CREATE TABLE IF NOT EXISTS nx_artifacts_v3 (
    artifact_kind TEXT NOT NULL,
    artifact_id TEXT NOT NULL,
    operation_id TEXT NOT NULL,
    case_id TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    payload_fingerprint TEXT NOT NULL,
    native_record_id TEXT,
    vms_instance_id TEXT,
    created_at TEXT NOT NULL,
    PRIMARY KEY (artifact_kind, artifact_id)
);

INSERT OR IGNORE INTO nx_artifacts_v3(
    artifact_kind, artifact_id, operation_id, case_id,
    payload_json, payload_fingerprint, native_record_id,
    vms_instance_id, created_at
)
SELECT
    artifact_kind, artifact_id, operation_id, case_id,
    payload_json, payload_fingerprint, native_record_id,
    vms_instance_id, created_at
FROM nx_artifacts;

DROP TABLE nx_artifacts;
ALTER TABLE nx_artifacts_v3 RENAME TO nx_artifacts;

CREATE INDEX IF NOT EXISTS ix_nx_artifacts_case
    ON nx_artifacts(case_id, artifact_kind);

CREATE INDEX IF NOT EXISTS ix_nx_artifacts_operation
    ON nx_artifacts(operation_id, artifact_kind);

PRAGMA foreign_keys = ON;
"""

APP_TABLE_PREFIX = "nx_"
CHECKPOINT_TABLES = frozenset({"checkpoints", "writes", "checkpoint_migrations"})
