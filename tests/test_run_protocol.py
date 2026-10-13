from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from rtv_eval.storage.db import Database


class RunProtocolTest(unittest.TestCase):
    def test_resume_requires_same_configuration_and_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Database(Path(tmp) / 'results.db')
            try:
                args = dict(experiment_name='exp', model_name='model', strategy_name='1fps', benchmark_name='motionbench')
                old = db.get_or_create_run(**args, config_json='{"model": {}}')
                self.assertEqual(old, db.get_or_create_run(**args, config_json='{ "model" : {} }'))
                with self.assertRaisesRegex(ValueError, 'new experiment name'):
                    db.get_or_create_run(**args, config_json='{"model": {}, "evaluation_protocol": "mcq-letter-v2"}')
                fresh = db.get_or_create_run(**{**args, 'experiment_name': 'exp_v2'}, config_json='{"evaluation_protocol": "mcq-letter-v2"}')
                self.assertNotEqual(old, fresh)
            finally:
                db.close()

    def test_schema_upgrade_does_not_validate_or_rewrite_legacy_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'old.db'
            with sqlite3.connect(path) as old:
                old.execute('''CREATE TABLE results (
                    id INTEGER PRIMARY KEY, run_id INTEGER, qa_uid TEXT,
                    question_type TEXT, video_path TEXT, raw_response TEXT,
                    extracted_answer TEXT, ground_truth TEXT, is_correct INTEGER,
                    latency_ms REAL, num_frames INTEGER, created_at TEXT)''')
                old.execute("INSERT INTO results (id, run_id, qa_uid, raw_response, extracted_answer) VALUES (1, 1, 'q1', 'a person', 'A')")
            db = Database(path)
            try:
                row = db.get_run_result_rows(1)[0]
                self.assertEqual(row['raw_response'], 'a person')
                self.assertEqual(row['extracted_answer'], 'A')
                self.assertIsNone(row['answer_status'])
            finally:
                db.close()


if __name__ == '__main__':
    unittest.main()
