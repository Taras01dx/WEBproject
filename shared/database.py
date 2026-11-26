"""SQLite (WAL mode) persistence layer for users, tasks, workers, and cluster events."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from shared.config import DB_PATH
from shared.security import hash_password

_db_lock = threading.RLock()


class Database:
    """Thread-safe SQLite repository with WAL mode for concurrent cluster access."""

    def __init__(self, db_path: Path | str = DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_schema()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path), timeout=15.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.execute("PRAGMA busy_timeout=10000;")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init_schema(self) -> None:
        with _db_lock, self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'user',
                    created_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    username TEXT NOT NULL,
                    matrix_size INTEGER NOT NULL,
                    method TEXT NOT NULL,
                    input_mode TEXT NOT NULL DEFAULT 'random',
                    seed INTEGER,
                    custom_payload TEXT,
                    timeout_sec INTEGER NOT NULL DEFAULT 45,
                    status TEXT NOT NULL DEFAULT 'queued',
                    progress REAL NOT NULL DEFAULT 0.0,
                    stage TEXT NOT NULL DEFAULT 'У черзі',
                    worker_id TEXT,
                    queue_position INTEGER DEFAULT 0,
                    estimated_wait_sec REAL DEFAULT 0.0,
                    created_at REAL NOT NULL,
                    started_at REAL,
                    finished_at REAL,
                    execution_time_ms REAL,
                    residual_norm REAL,
                    iterations INTEGER,
                    solution_preview TEXT,
                    error_message TEXT,
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_tasks_user_created ON tasks(user_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status, created_at ASC);

                CREATE TABLE IF NOT EXISTS system_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    source TEXT NOT NULL,
                    message TEXT NOT NULL,
                    details TEXT,
                    created_at REAL NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_events_created ON system_events(created_at DESC);
                """
            )

        self._seed_default_users()

    def _seed_default_users(self) -> None:
        """Create default demo accounts (admin & student) if missing."""
        now = time.time()
        defaults = [
            ("admin", "admin123", "admin"),
            ("student", "student123", "user"),
        ]
        with _db_lock, self.connect() as conn:
            for username, password, role in defaults:
                existing = conn.execute(
                    "SELECT id FROM users WHERE username = ?", (username,)
                ).fetchone()
                if not existing:
                    conn.execute(
                        "INSERT INTO users (username, password_hash, role, created_at) VALUES (?, ?, ?, ?)",
                        (username, hash_password(password), role, now),
                    )

    # ---------------- Users ----------------

    def create_user(self, username: str, password: str, role: str = "user") -> dict[str, Any]:
        now = time.time()
        with _db_lock, self.connect() as conn:
            cur = conn.execute(
                "INSERT INTO users (username, password_hash, role, created_at) VALUES (?, ?, ?, ?)",
                (username, hash_password(password), role, now),
            )
            return {
                "id": cur.lastrowid,
                "username": username,
                "role": role,
                "created_at": now,
            }

    def get_user_by_username(self, username: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
            return dict(row) if row else None

    def get_user_by_id(self, user_id: int) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT id, username, role, created_at FROM users WHERE id = ?", (user_id,)
            ).fetchone()
            return dict(row) if row else None

    # ---------------- Tasks ----------------

    def create_task(self, task_data: dict[str, Any]) -> dict[str, Any]:
        with _db_lock, self.connect() as conn:
            conn.execute(
                """
                INSERT INTO tasks (
                    id, user_id, username, matrix_size, method, input_mode, seed,
                    custom_payload, timeout_sec, status, progress, stage,
                    queue_position, estimated_wait_sec, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task_data["id"],
                    task_data["user_id"],
                    task_data["username"],
                    task_data["matrix_size"],
                    task_data["method"],
                    task_data.get("input_mode", "random"),
                    task_data.get("seed"),
                    json.dumps(task_data["custom_payload"]) if task_data.get("custom_payload") else None,
                    task_data["timeout_sec"],
                    task_data.get("status", "queued"),
                    task_data.get("progress", 0.0),
                    task_data.get("stage", "У черзі"),
                    task_data.get("queue_position", 0),
                    task_data.get("estimated_wait_sec", 0.0),
                    task_data["created_at"],
                ),
            )
        return self.get_task(task_data["id"])  # type: ignore[return-value]

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
            return self._serialize_task(row) if row else None

    def update_task(self, task_id: str, **fields: Any) -> dict[str, Any] | None:
        if not fields:
            return self.get_task(task_id)
        if "solution_preview" in fields and isinstance(fields["solution_preview"], (list, dict)):
            fields["solution_preview"] = json.dumps(fields["solution_preview"])
        cols = ", ".join(f"{k} = ?" for k in fields.keys())
        values = list(fields.values()) + [task_id]
        with _db_lock, self.connect() as conn:
            conn.execute(f"UPDATE tasks SET {cols} WHERE id = ?", values)
        return self.get_task(task_id)

    def delete_task(self, task_id: str, user_id: int | None = None) -> bool:
        with _db_lock, self.connect() as conn:
            if user_id is not None:
                cur = conn.execute(
                    "DELETE FROM tasks WHERE id = ? AND user_id = ? AND status IN ('completed', 'failed', 'cancelled')",
                    (task_id, user_id),
                )
            else:
                cur = conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
            return cur.rowcount > 0

    def list_user_tasks(self, user_id: int, limit: int = 50) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM tasks WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
            return [self._serialize_task(r) for r in rows]

    def list_all_tasks(self, limit: int = 100) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM tasks ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [self._serialize_task(r) for r in rows]

    def list_active_tasks(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM tasks WHERE status IN ('queued', 'running') ORDER BY created_at ASC"
            ).fetchall()
            return [self._serialize_task(r) for r in rows]

    def list_queued_tasks(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM tasks WHERE status = 'queued' ORDER BY created_at ASC"
            ).fetchall()
            return [self._serialize_task(r) for r in rows]

    def count_user_tasks(self, user_id: int) -> tuple[int, int]:
        """Return (active_count, total_count) for a user."""
        with self.connect() as conn:
            active = conn.execute(
                "SELECT COUNT(*) FROM tasks WHERE user_id = ? AND status IN ('queued', 'running')",
                (user_id,),
            ).fetchone()[0]
            total = conn.execute(
                "SELECT COUNT(*) FROM tasks WHERE user_id = ?",
                (user_id,),
            ).fetchone()[0]
            return int(active), int(total)

    def get_average_ms_per_cubic_n(self, method: str) -> float | None:
        """Empirical calibration constant k where time_ms ~= k * n^3 (or n^2 for jacobi)."""
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT matrix_size, execution_time_ms
                FROM tasks
                WHERE status = 'completed' AND method = ? AND execution_time_ms IS NOT NULL AND matrix_size >= 50
                ORDER BY finished_at DESC LIMIT 20
                """,
                (method,),
            ).fetchall()
            if not rows:
                return None
            ratios = []
            power = 3 if method == "gauss" else 2
            for r in rows:
                denom = float(r["matrix_size"] ** power)
                if denom > 0:
                    ratios.append(float(r["execution_time_ms"]) / denom)
            return sum(ratios) / len(ratios) if ratios else None

    def get_task_stats(self) -> dict[str, int]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) as cnt FROM tasks GROUP BY status"
            ).fetchall()
            counts = {r["status"]: int(r["cnt"]) for r in rows}
            return {
                "queued": counts.get("queued", 0),
                "running": counts.get("running", 0),
                "completed": counts.get("completed", 0),
                "failed": counts.get("failed", 0),
                "cancelled": counts.get("cancelled", 0),
                "total": sum(counts.values()),
            }

    def mark_orphaned_tasks_failed(self) -> None:
        """On fresh startup, mark any previously running/queued tasks as failed."""
        now = time.time()
        with _db_lock, self.connect() as conn:
            conn.execute(
                """
                UPDATE tasks
                SET status = 'failed',
                    stage = 'Перервано перезапуском сервера',
                    error_message = 'Сервер було перезапущено під час виконання задачі',
                    finished_at = ?
                WHERE status IN ('running', 'queued')
                """,
                (now,),
            )

    # ---------------- System Events ----------------

    def log_event(
        self,
        event_type: str,
        source: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        now = time.time()
        with _db_lock, self.connect() as conn:
            conn.execute(
                "INSERT INTO system_events (event_type, source, message, details, created_at) VALUES (?, ?, ?, ?, ?)",
                (event_type, source, message, json.dumps(details) if details else None, now),
            )

    def list_events(self, limit: int = 50) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM system_events ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
            events = []
            for r in rows:
                item = dict(r)
                if item.get("details"):
                    try:
                        item["details"] = json.loads(item["details"])
                    except Exception:
                        pass
                events.append(item)
            return events

    @staticmethod
    def _serialize_task(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        item["cancel_requested"] = bool(item.get("cancel_requested", 0))
        if item.get("solution_preview"):
            try:
                item["solution_preview"] = json.loads(item["solution_preview"])
            except Exception:
                pass
        if item.get("custom_payload"):
            try:
                item["custom_payload"] = json.loads(item["custom_payload"])
            except Exception:
                pass
        return item
