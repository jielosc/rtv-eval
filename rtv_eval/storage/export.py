from __future__ import annotations

import csv
import json
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

    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
