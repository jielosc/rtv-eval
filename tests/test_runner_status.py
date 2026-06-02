from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import httpx
import openai

from rtv_eval.benchmarks.base import QAEntry
from rtv_eval.config import ExperimentConfig
from rtv_eval.runner import Runner


class _FakeBenchmark:
    def load(self, config) -> list[QAEntry]:
        return [
            QAEntry(
                uid="q1",
                video_path="video.mp4",
                question="Question?",
                answer="A",
                metadata={"question_type": "mcq"},
            )
        ]

    @staticmethod
    def build_video_index(video_root: Path) -> dict[str, Path]:
        return {"video.mp4": Path("/tmp/video.mp4")}


class _FailingCaller:
    async def call(self, question: str, frame_b64_list: list[str]) -> str:
        request = httpx.Request("POST", "http://127.0.0.1:8000/v1/chat/completions")
        raise openai.APIConnectionError(request=request)


class _FakePreprocessor:
    def __init__(self, config) -> None:
        self.config = config

    def process(self, video_path: Path) -> list[str]:
        return ["frame"]


def _build_config(tmp_path: Path) -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "name": "test_experiment",
            "output_dir": str(tmp_path),
            "concurrency": 1,
            "benchmark": {
                "name": "motionbench",
                "type": "motionbench",
                "data_dir": str(tmp_path),
                "video_root": str(tmp_path),
            },
            "models": [
                {
                    "name": "internvl3-8b",
                    "base_url": "http://127.0.0.1:8000/v1",
                    "api_key_env": "DUMMY",
                    "model_name": "internvl3-8b",
                }
            ],
            "strategies": [
                {
                    "name": "1fps_360p",
                    "frame_sampling": {"mode": "fps", "fps": 1.0},
                    "resolution": {"mode": "preset", "preset": "360p"},
                }
            ],
        }
    )


class RunnerStatusTest(unittest.TestCase):
    def test_connection_error_marks_run_failed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _build_config(Path(tmpdir))
            with (
                mock.patch("rtv_eval.runner._load_benchmark", return_value=_FakeBenchmark()),
                mock.patch("rtv_eval.runner._load_caller", return_value=_FailingCaller()),
                mock.patch("rtv_eval.runner.VideoPreprocessor", _FakePreprocessor),
            ):
                runner = Runner(config)
                try:
                    asyncio.run(runner.run())

                    runs = runner.db.get_all_runs()
                    self.assertEqual(len(runs), 1)
                    self.assertEqual(runs[0]["status"], "failed")
                    self.assertEqual(runner.db.get_run_results(runs[0]["id"]), [])
                finally:
                    runner.db.close()


if __name__ == "__main__":
    unittest.main()
