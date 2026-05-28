from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_name TEXT NOT NULL,
    model_name TEXT NOT NULL,
    strategy_name TEXT NOT NULL,
    benchmark_name TEXT NOT NULL,
    config_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    status TEXT NOT NULL DEFAULT 'running',
    UNIQUE(experiment_name, model_name, strategy_name, benchmark_name)
);

CREATE TABLE IF NOT EXISTS results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(id),
    qa_uid TEXT NOT NULL,
    question_type TEXT,
    video_path TEXT,
    raw_response TEXT,
    extracted_answer TEXT,
    ground_truth TEXT,
    is_correct INTEGER,
    latency_ms REAL,
    num_frames INTEGER,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(run_id, qa_uid)
);

CREATE INDEX IF NOT EXISTS idx_results_run_id ON results(run_id);
"""


class Database:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path))
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    def get_or_create_run(
        self,
        experiment_name: str,
        model_name: str,
        strategy_name: str,
        benchmark_name: str,
        config_json: str,
    ) -> int:
        """Get existing run ID or create a new one."""
        cur = self.conn.execute(
            "SELECT id FROM runs WHERE experiment_name=? AND model_name=? "
            "AND strategy_name=? AND benchmark_name=?",
            (experiment_name, model_name, strategy_name, benchmark_name),
        )
        row = cur.fetchone()
        if row:
            return row[0]

        cur = self.conn.execute(
            "INSERT INTO runs (experiment_name, model_name, strategy_name, "
            "benchmark_name, config_json) VALUES (?, ?, ?, ?, ?)",
            (experiment_name, model_name, strategy_name, benchmark_name, config_json),
        )
        self.conn.commit()
        return cur.lastrowid

    def get_completed_uids(self, run_id: int) -> set[str]:
        """Return set of QA UIDs already completed for this run."""
        cur = self.conn.execute(
            "SELECT qa_uid FROM results WHERE run_id=?", (run_id,)
        )
        return {row[0] for row in cur.fetchall()}

    def save_result(
        self,
        run_id: int,
        qa_uid: str,
        question_type: str | None,
        video_path: str,
        raw_response: str,
        extracted_answer: str,
        ground_truth: str | None,
        is_correct: bool | None,
        latency_ms: float,
        num_frames: int,
    ) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO results "
            "(run_id, qa_uid, question_type, video_path, raw_response, "
            "extracted_answer, ground_truth, is_correct, latency_ms, num_frames) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id, qa_uid, question_type, video_path, raw_response,
                extracted_answer, ground_truth,
                1 if is_correct else (0 if is_correct is not None else None),
                latency_ms, num_frames,
            ),
        )
        self.conn.commit()

    def mark_run_running(self, run_id: int) -> None:
        self.conn.execute(
            "UPDATE runs SET status='running' WHERE id=?", (run_id,)
        )
        self.conn.commit()

    def mark_run_completed(self, run_id: int) -> None:
        self.conn.execute(
            "UPDATE runs SET status='completed' WHERE id=?", (run_id,)
        )
        self.conn.commit()

    def mark_run_failed(self, run_id: int) -> None:
        self.conn.execute(
            "UPDATE runs SET status='failed' WHERE id=?", (run_id,)
        )
        self.conn.commit()

    def get_all_runs(self) -> list[dict]:
        cur = self.conn.execute(
            "SELECT id, experiment_name, model_name, strategy_name, "
            "benchmark_name, status, created_at FROM runs ORDER BY id"
        )
        return [
            {
                "id": r[0], "experiment_name": r[1], "model_name": r[2],
                "strategy_name": r[3], "benchmark_name": r[4],
                "status": r[5], "created_at": r[6],
            }
            for r in cur.fetchall()
        ]

    def get_run_results(self, run_id: int) -> list[dict]:
        cur = self.conn.execute(
            "SELECT qa_uid, question_type, extracted_answer, ground_truth, "
            "is_correct, latency_ms, num_frames FROM results WHERE run_id=?",
            (run_id,),
        )
        return [
            {
                "qa_uid": r[0], "question_type": r[1],
                "extracted_answer": r[2], "ground_truth": r[3],
                "is_correct": r[4], "latency_ms": r[5], "num_frames": r[6],
            }
            for r in cur.fetchall()
        ]

    def get_accuracy_summary(self, run_id: int) -> dict:
        """Compute accuracy summary for a run (dev split only)."""
        results = self.get_run_results(run_id)
        scored = [r for r in results if r["is_correct"] is not None]
        if not scored:
            return {"total": 0, "correct": 0, "accuracy": 0.0, "by_type": {}}

        total = len(scored)
        correct = sum(1 for r in scored if r["is_correct"])

        by_type: dict[str, dict] = {}
        for r in scored:
            qt = r["question_type"] or "unknown"
            if qt not in by_type:
                by_type[qt] = {"total": 0, "correct": 0}
            by_type[qt]["total"] += 1
            if r["is_correct"]:
                by_type[qt]["correct"] += 1

        for v in by_type.values():
            v["accuracy"] = round(v["correct"] / v["total"], 4) if v["total"] else 0.0

        return {
            "total": total,
            "correct": correct,
            "accuracy": round(correct / total, 4),
            "by_type": by_type,
        }

    def close(self) -> None:
        self.conn.close()
