"""SQLite connection helpers for Investigation Persistence.

This database is application-owned and must not share tables with the native
LangGraph Checkpointer database file.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from nanexus_investigation.persistence.schema import (
    APP_TABLE_PREFIX,
    DDL,
    DDL_V3_UPGRADE,
    SCHEMA_VERSION,
)

_REQUIRED_APP_TABLES = frozenset(
    {
        "nx_schema_meta",
        "nx_cases",
        "nx_operations",
        "nx_artifacts",
        "nx_registration_groups",
        "nx_budget_policies",
        "nx_budget_reservations",
    }
)


@dataclass
class InvestigationDatabase:
    """Thin SQLite handle for the Investigation Persistence store."""

    path: Path
    conn: sqlite3.Connection

    @classmethod
    def open(cls, path: str | Path) -> "InvestigationDatabase":
        db_path = Path(path)
        if db_path.parent and str(db_path.parent) not in {".", ""}:
            db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(
            str(db_path),
            isolation_level=None,  # explicit transactions
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        db = cls(path=db_path, conn=conn)
        db.migrate()
        return db

    def migrate(self) -> None:
        # executescript() issues an implicit commit; do not wrap it in BEGIN/COMMIT.
        # CREATE IF NOT EXISTS applies additive tables (e.g. registration groups) to
        # older databases without destroying existing rows.
        # Version is recorded only after required tables are verified present.
        self.conn.executescript(DDL)

        row = self.conn.execute(
            "SELECT value FROM nx_schema_meta WHERE key = 'schema_version'"
        ).fetchone()
        if row is None:
            with self.transaction():
                self._assert_schema_installed()
                self.conn.execute(
                    "INSERT INTO nx_schema_meta(key, value) VALUES (?, ?)",
                    ("schema_version", str(SCHEMA_VERSION)),
                )
            return

        current = int(row["value"])
        if current < 1:
            raise RuntimeError(
                f"unsupported schema_version={current}; expected 1..{SCHEMA_VERSION}"
            )
        if current > SCHEMA_VERSION:
            raise RuntimeError(
                f"unsupported schema_version={current}; expected {SCHEMA_VERSION}"
            )
        if current == SCHEMA_VERSION:
            self._assert_schema_installed()
            return

        # current < SCHEMA_VERSION: apply structural upgrades then bump version.
        if current < 3:
            # Rebuild nx_artifacts without UNIQUE(operation_id, artifact_kind).
            self.conn.executescript(DDL_V3_UPGRADE)
        with self.transaction():
            self._assert_schema_installed()
            self.conn.execute(
                "UPDATE nx_schema_meta SET value = ? WHERE key = 'schema_version'",
                (str(SCHEMA_VERSION),),
            )

    def _assert_schema_installed(self) -> None:
        tables = self.table_names()
        missing = sorted(
            name
            for name in _REQUIRED_APP_TABLES
            if name.startswith(APP_TABLE_PREFIX) and name not in tables
        )
        if missing:
            raise RuntimeError(
                "investigation schema incomplete after migrate; "
                f"missing tables: {', '.join(missing)}"
            )

    def transaction(self):
        return _Transaction(self.conn)

    def close(self) -> None:
        self.conn.close()

    def table_names(self) -> set[str]:
        rows = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
        return {row["name"] for row in rows}


class _Transaction:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def __enter__(self) -> sqlite3.Connection:
        self._conn.execute("BEGIN IMMEDIATE")
        return self._conn

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self._conn.execute("COMMIT")
        else:
            self._conn.execute("ROLLBACK")
