from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from rtv_eval.config import load_config
from rtv_eval.storage.db import Database
from rtv_eval.storage.export import (
    build_result_rows,
    build_summary_rows,
    export_csv,
    export_json,
    export_table,
    write_rows_csv,
    write_rows_json,
)

console = Console()


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging")
def cli(verbose: bool) -> None:
    """RTV-Eval: Real-Time Video Understanding Evaluation Platform."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--dry-run", is_flag=True, help="Print experiment plan without running")
def run(config_path: str, dry_run: bool) -> None:
    """Run an evaluation experiment from a YAML config."""
    from rtv_eval.runner import Runner

    config = load_config(config_path)

    if dry_run:
        console.print(f"[bold]Experiment:[/] {config.name}")
        console.print(f"[bold]Benchmark:[/] {config.benchmark.name} (split={config.benchmark.split})")
        console.print(f"[bold]Models:[/] {[m.name for m in config.models]}")
        console.print(f"[bold]Strategies:[/] {[s.name for s in config.strategies]}")
        console.print(f"[bold]Concurrency:[/] {config.concurrency}")
        combos = len(config.models) * len(config.strategies)
        console.print(f"[bold]Total combos:[/] {combos}")
        return

    runner = Runner(config)
    asyncio.run(runner.run())

    # Print summary
    console.print("\n[bold green]Done![/]")
    for run_info in runner.db.get_all_runs():
        if run_info["experiment_name"] == config.name:
            summary = runner.db.get_accuracy_summary(run_info["id"])
            console.print(
                f"  {run_info['model_name']} x {run_info['strategy_name']}: "
                f"{summary['accuracy']*100:.1f}% ({summary['correct']}/{summary['total']})"
            )


@cli.command()
@click.argument("db_path", type=click.Path(exists=True))
@click.option("--format", "fmt", type=click.Choice(["table", "json", "csv"]), default="table")
@click.option("--output", "-o", type=click.Path(), default=None, help="Output file path")
def report(db_path: str, fmt: str, output: str | None) -> None:
    """Generate comparison report from a results database."""
    db = Database(Path(db_path))
    runs = db.get_all_runs()

    if not runs:
        console.print("[yellow]No runs found in database.[/]")
        db.close()
        return

    if fmt == "table":
        export_table(db, runs, console)
    elif fmt == "json":
        data = export_json(db, runs)
        text = json.dumps(data, indent=2, ensure_ascii=False)
        if output:
            Path(output).write_text(text)
            console.print(f"Written to {output}")
        else:
            console.print(text)
    elif fmt == "csv":
        if not output:
            console.print("[red]--output is required for CSV format.[/]")
            db.close()
            return
        export_csv(db, runs, Path(output))
        console.print(f"Written to {output}")

    db.close()


@cli.command()
@click.argument("db_path", type=click.Path(exists=True))
@click.option(
    "--view",
    type=click.Choice(["summary", "results"]),
    default="summary",
    show_default=True,
    help="Choose aggregate run data or per-question long-form data.",
)
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["json", "csv"]),
    default="csv",
    show_default=True,
)
@click.option("--output", "-o", type=click.Path(), default=None, help="Output file path")
def export(db_path: str, view: str, fmt: str, output: str | None) -> None:
    """Export visualization-friendly data from a results database."""
    db = Database(Path(db_path))
    runs = db.get_all_runs()

    if not runs:
        console.print("[yellow]No runs found in database.[/]")
        db.close()
        return

    rows = build_summary_rows(db, runs) if view == "summary" else build_result_rows(db, runs)
    output_path = Path(output) if output else Path(db_path).with_suffix(f".{view}.{fmt}")

    if fmt == "json":
        write_rows_json(rows, output_path)
    else:
        write_rows_csv(rows, output_path)

    console.print(f"Written to {output_path}")
    db.close()


@cli.command()
@click.argument("db_path", type=click.Path(exists=True))
def status(db_path: str) -> None:
    """Show status of all runs in a results database."""
    db = Database(Path(db_path))
    runs = db.get_all_runs()

    if not runs:
        console.print("[yellow]No runs found.[/]")
        db.close()
        return

    table = Table(title="Run Status")
    table.add_column("ID", style="dim")
    table.add_column("Experiment")
    table.add_column("Model")
    table.add_column("Strategy")
    table.add_column("Status")
    table.add_column("Progress")
    table.add_column("Created")

    for r in runs:
        results = db.get_run_results(r["id"])
        done = len(results)
        status_style = {
            "completed": "[green]completed[/]",
            "running": "[yellow]running[/]",
            "failed": "[red]failed[/]",
        }.get(r["status"], r["status"])

        table.add_row(
            str(r["id"]),
            r["experiment_name"],
            r["model_name"],
            r["strategy_name"],
            status_style,
            str(done),
            r["created_at"],
        )

    console.print(table)
    db.close()


if __name__ == "__main__":
    cli()
