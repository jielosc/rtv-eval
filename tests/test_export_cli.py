from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from click.testing import CliRunner

from rtv_eval.cli import cli
from rtv_eval.storage.db import Database


class ExportCliTest(unittest.TestCase):
    def test_export_summary_and_results_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            db_path = tmp_path / "results.db"
            db = Database(db_path)

            run_id = db.get_or_create_run(
                experiment_name="exp1",
                model_name="model-a",
                strategy_name="8frames_480p",
                benchmark_name="motionbench",
                config_json="{}",
            )
            db.save_result(
                run_id=run_id,
                qa_uid="q1",
                question_type="motion",
                video_path="video1.mp4",
                raw_response="The person is walking.",
                extracted_answer="walking",
                ground_truth="walking",
                is_correct=True,
                latency_ms=123.4,
                num_frames=8,
            )
            db.save_result(
                run_id=run_id,
                qa_uid="q2",
                question_type="counting",
                video_path="video2.mp4",
                raw_response="Two objects.",
                extracted_answer="two",
                ground_truth="three",
                is_correct=False,
                latency_ms=234.5,
                num_frames=8,
            )
            db.mark_run_completed(run_id)
            db.close()

            runner = CliRunner()
            summary_path = tmp_path / "summary.csv"
            results_path = tmp_path / "results.csv"

            summary_result = runner.invoke(
                cli,
                ["export", str(db_path), "--view", "summary", "--output", str(summary_path)],
            )
            self.assertEqual(summary_result.exit_code, 0, summary_result.output)

            results_result = runner.invoke(
                cli,
                ["export", str(db_path), "--view", "results", "--output", str(results_path)],
            )
            self.assertEqual(results_result.exit_code, 0, results_result.output)

            with summary_path.open(newline="", encoding="utf-8") as f:
                summary_rows = list(csv.DictReader(f))
            self.assertEqual(len(summary_rows), 1)
            self.assertEqual(summary_rows[0]["model"], "model-a")
            self.assertEqual(summary_rows[0]["accuracy"], "0.5")
            self.assertEqual(summary_rows[0]["avg_num_frames"], "8")

            with results_path.open(newline="", encoding="utf-8") as f:
                result_rows = list(csv.DictReader(f))
            self.assertEqual(len(result_rows), 2)
            self.assertEqual(result_rows[0]["qa_uid"], "q1")
            self.assertEqual(result_rows[0]["question_type"], "motion")
            self.assertEqual(result_rows[1]["is_correct"], "0")


if __name__ == "__main__":
    unittest.main()
