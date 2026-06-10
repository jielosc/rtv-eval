from __future__ import annotations

import csv
import json
from statistics import mean
from pathlib import Path

from rich.console import Console
from rich.table import Table

from rtv_eval.storage.db import Database


def export_table(db: Database, runs: list[dict], console: Console) -> None:
    """Print a comparison table of all runs."""
    # Collect all strategies and models
    models: list[str] = []
    strategies: list[str] = []
    summaries: dict[tuple[str, str], dict] = {}

    for r in runs:
        if r["status"] != "completed":
            continue
        summary = db.get_accuracy_summary(r["id"])
        key = (r["model_name"], r["strategy_name"])
        summaries[key] = summary
        if r["model_name"] not in models:
            models.append(r["model_name"])
        if r["strategy_name"] not in strategies:
            strategies.append(r["strategy_name"])

    if not summaries:
        console.print("[yellow]No completed runs to report.[/]")
        return

    # Main accuracy table
    table = Table(title="Accuracy (%) - Model x Strategy")
    table.add_column("Model", style="bold")
    for s in strategies:
        table.add_column(s, justify="right")

    for m in models:
        row = [m]
        for s in strategies:
            key = (m, s)
            if key in summaries:
                acc = summaries[key]["accuracy"] * 100
                row.append(f"{acc:.1f}")
            else:
                row.append("-")
        table.add_row(*row)

    console.print(table)

    # Per question-type breakdown for each model
    all_types: set[str] = set()
    for s in summaries.values():
        all_types.update(s.get("by_type", {}).keys())
    types_sorted = sorted(all_types)

    if types_sorted:
        for m in models:
            sub_table = Table(title=f"Breakdown by Question Type - {m}")
            sub_table.add_column("Question Type", style="bold")
            for s in strategies:
                sub_table.add_column(s, justify="right")

            for qt in types_sorted:
                row = [qt]
                for s in strategies:
                    key = (m, s)
                    if key in summaries and qt in summaries[key].get("by_type", {}):
                        acc = summaries[key]["by_type"][qt]["accuracy"] * 100
                        row.append(f"{acc:.1f}")
                    else:
                        row.append("-")
                sub_table.add_row(*row)

            console.print(sub_table)


def export_json(db: Database, runs: list[dict]) -> list[dict]:
    """Export all run summaries as JSON-serializable dicts."""
    result = []
    for r in runs:
        if r["status"] != "completed":
            continue
        summary = db.get_accuracy_summary(r["id"])
        result.append({
            "run_id": r["id"],
            "experiment": r["experiment_name"],
            "model": r["model_name"],
            "strategy": r["strategy_name"],
            "benchmark": r["benchmark_name"],
            "accuracy": summary["accuracy"],
            "correct": summary["correct"],
            "total": summary["total"],
            "by_type": summary["by_type"],
            "created_at": r["created_at"],
        })
    return result


def export_csv(db: Database, runs: list[dict], output_path: Path) -> None:
    """Export results to CSV."""
    rows = []
    for r in runs:
        if r["status"] != "completed":
            continue
        summary = db.get_accuracy_summary(r["id"])
        row = {
            "run_id": r["id"],
            "experiment": r["experiment_name"],
            "model": r["model_name"],
            "strategy": r["strategy_name"],
            "benchmark": r["benchmark_name"],
            "accuracy": summary["accuracy"],
            "correct": summary["correct"],
            "total": summary["total"],
        }
        # Add per-type accuracy as columns
        for qt, stats in summary.get("by_type", {}).items():
            row[f"acc_{qt}"] = stats["accuracy"]
        rows.append(row)

    if not rows:
        return

    # Collect all field names across every row to handle differing question_type columns
    all_keys: list[str] = list(dict.fromkeys(k for row in rows for k in row))
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, restval="", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_summary_rows(db: Database, runs: list[dict]) -> list[dict]:
    rows = []
    for r in runs:
        result_rows = db.get_run_result_rows(r["id"])
        summary = db.get_accuracy_summary(r["id"])
        scored_rows = [row for row in result_rows if row["is_correct"] is not None]
        latency_rows = [row["latency_ms"] for row in result_rows if row["latency_ms"] is not None]
        frame_rows = [row["num_frames"] for row in result_rows if row["num_frames"] is not None]
        rows.append({
            "run_id": r["id"],
            "experiment": r["experiment_name"],
            "model": r["model_name"],
            "strategy": r["strategy_name"],
            "benchmark": r["benchmark_name"],
            "status": r["status"],
            "created_at": r["created_at"],
            "result_count": len(result_rows),
            "scored_count": len(scored_rows),
            "correct": summary["correct"],
            "total": summary["total"],
            "accuracy": summary["accuracy"],
            "avg_latency_ms": round(mean(latency_rows), 4) if latency_rows else None,
            "avg_num_frames": round(mean(frame_rows), 4) if frame_rows else None,
        })
    return rows


def build_result_rows(db: Database, runs: list[dict]) -> list[dict]:
    rows = []
    for r in runs:
        for result in db.get_run_result_rows(r["id"]):
            rows.append({
                "run_id": r["id"],
                "experiment": r["experiment_name"],
                "model": r["model_name"],
                "strategy": r["strategy_name"],
                "benchmark": r["benchmark_name"],
                "run_status": r["status"],
                "run_created_at": r["created_at"],
                "qa_uid": result["qa_uid"],
                "question_type": result["question_type"],
                "video_path": result["video_path"],
                "raw_response": result["raw_response"],
                "extracted_answer": result["extracted_answer"],
                "ground_truth": result["ground_truth"],
                "is_correct": result["is_correct"],
                "latency_ms": result["latency_ms"],
                "num_frames": result["num_frames"],
                "result_created_at": result["created_at"],
            })
    return rows


def write_rows_json(rows: list[dict], output_path: Path) -> None:
    output_path.write_text(
        json.dumps(rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def write_rows_csv(rows: list[dict], output_path: Path) -> None:
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return

    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
