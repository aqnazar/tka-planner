"""Cases, plans and committed geometry, in SQLite.

Records go in the database, geometry goes in the mesh directory beside it, and the
database refers to geometry by content hash. Keeping the two apart is what lets a
hundred-megabyte femur be stored once and referenced by every plan that did not change
it, and it keeps the database small enough to copy.

Every row carries an ``owner``. It is the empty string for a single-user install and is
never read there, but its presence is the difference between multi-user being a filter
and multi-user being a migration. The spec asks for the schema not to assume one user;
this is what that costs.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from .meshes import MeshStore

__all__ = ["Store", "CaseRecord"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS cases (
    case_id     TEXT NOT NULL,
    owner       TEXT NOT NULL DEFAULT '',
    side        TEXT NOT NULL,
    femur_path  TEXT NOT NULL,
    tibia_path  TEXT NOT NULL,
    library     TEXT,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    PRIMARY KEY (case_id, owner)
);

CREATE TABLE IF NOT EXISTS plans (
    plan_key   TEXT NOT NULL,
    owner      TEXT NOT NULL DEFAULT '',
    case_id    TEXT NOT NULL,
    controls   TEXT NOT NULL,
    scalars    TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (plan_key, owner)
);

CREATE TABLE IF NOT EXISTS commits (
    plan_key TEXT NOT NULL,
    owner    TEXT NOT NULL DEFAULT '',
    node     TEXT NOT NULL,
    mesh_id  TEXT NOT NULL,
    record   TEXT NOT NULL,
    PRIMARY KEY (plan_key, owner, node)
);

CREATE TABLE IF NOT EXISTS meshes (
    mesh_id    TEXT PRIMARY KEY,
    n_vertices INTEGER NOT NULL,
    n_faces    INTEGER NOT NULL,
    bytes      INTEGER NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS plans_by_case ON plans (owner, case_id);
"""


class CaseRecord(dict):
    """A case row. A dict, because it goes straight out as JSON."""


class Store:
    """The case database and the mesh directory that belongs to it."""

    def __init__(self, root, *, owner: str = ""):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.owner = owner
        self.meshes = MeshStore(self.root / "meshes")
        self._db_path = self.root / "cases.db"
        with closing(self._connect()) as connection:
            connection.executescript(SCHEMA)
            connection.commit()

    def _connect(self) -> sqlite3.Connection:
        # A connection per operation rather than one held open: the server is threaded,
        # and SQLite connections are not safe to share across threads. Opening a local
        # file is measured in microseconds, so there is nothing to save by keeping one.
        connection = sqlite3.connect(self._db_path, timeout=10.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    # ------------------------------------------------------------------
    # Cases
    # ------------------------------------------------------------------

    def save_case(self, case_id, *, side, femur_path, tibia_path, library=None) -> None:
        now = _now()
        with closing(self._connect()) as connection:
            connection.execute(
                """
                INSERT INTO cases
                    (case_id, owner, side, femur_path, tibia_path, library,
                     created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (case_id, owner) DO UPDATE SET
                    side = excluded.side,
                    femur_path = excluded.femur_path,
                    tibia_path = excluded.tibia_path,
                    library = excluded.library,
                    updated_at = excluded.updated_at
                """,
                (case_id, self.owner, str(side), str(femur_path), str(tibia_path),
                 str(library) if library else None, now, now),
            )
            connection.commit()

    def cases(self) -> list:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM cases WHERE owner = ? ORDER BY updated_at DESC",
                (self.owner,),
            ).fetchall()
        return [CaseRecord(dict(row)) for row in rows]

    def case(self, case_id: str):
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM cases WHERE owner = ? AND case_id = ?",
                (self.owner, case_id),
            ).fetchone()
        return CaseRecord(dict(row)) if row else None

    # ------------------------------------------------------------------
    # Plans and commits
    # ------------------------------------------------------------------

    def save_plan(self, plan_key, *, case_id, controls: dict, scalars: dict) -> None:
        with closing(self._connect()) as connection:
            connection.execute(
                """
                INSERT INTO plans
                    (plan_key, owner, case_id, controls, scalars, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT (plan_key, owner) DO UPDATE SET
                    controls = excluded.controls,
                    scalars = excluded.scalars
                """,
                (plan_key, self.owner, case_id, json.dumps(controls),
                 json.dumps(scalars), _now()),
            )
            connection.commit()

    def plans_for(self, case_id: str) -> list:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM plans WHERE owner = ? AND case_id = ? "
                "ORDER BY created_at DESC",
                (self.owner, case_id),
            ).fetchall()
        return [
            {**dict(row),
             "controls": json.loads(row["controls"]),
             "scalars": json.loads(row["scalars"])}
            for row in rows
        ]

    def cached_commit(self, plan_key: str):
        """The committed geometry for a plan, or ``None`` if it was never computed.

        A cache entry whose mesh file has gone missing is reported as a miss rather than
        as a broken hit, so deleting the mesh directory degrades to recomputation.
        """
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT node, mesh_id, record FROM commits "
                "WHERE owner = ? AND plan_key = ?",
                (self.owner, plan_key),
            ).fetchall()
        if not rows:
            return None
        if not all(self.meshes.has(row["mesh_id"]) for row in rows):
            return None
        return {
            row["node"]: {
                "mesh_id": row["mesh_id"],
                "record": json.loads(row["record"]),
            }
            for row in rows
        }

    def save_commit(self, plan_key: str, entries: dict) -> None:
        """Record a commit. ``entries`` maps a node name to its mesh id and record."""
        with closing(self._connect()) as connection:
            connection.executemany(
                """
                INSERT INTO commits (plan_key, owner, node, mesh_id, record)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT (plan_key, owner, node) DO UPDATE SET
                    mesh_id = excluded.mesh_id,
                    record = excluded.record
                """,
                [
                    (plan_key, self.owner, node, entry["mesh_id"],
                     json.dumps(entry.get("record") or {}))
                    for node, entry in entries.items()
                ],
            )
            connection.commit()

    # ------------------------------------------------------------------
    # Meshes
    # ------------------------------------------------------------------

    def put_mesh(self, mesh) -> str:
        """Store a mesh and index it, returning its id."""
        digest = self.meshes.put(mesh)
        with closing(self._connect()) as connection:
            connection.execute(
                """
                INSERT INTO meshes (mesh_id, n_vertices, n_faces, bytes, created_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT (mesh_id) DO NOTHING
                """,
                (digest, int(mesh.vertices.shape[0]), int(mesh.faces.shape[0]),
                 self.meshes.size_bytes(digest), _now()),
            )
            connection.commit()
        return digest

    def get_mesh(self, digest: str):
        return self.meshes.get(digest)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
