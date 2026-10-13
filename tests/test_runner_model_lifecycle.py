from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from contextlib import asynccontextmanager
from pathlib import Path
from unittest import mock

from rtv_eval.config import ModelConfig
from rtv_eval.runner import Runner
from tests.test_runner_status import _FakeBenchmark, _FakePreprocessor, _build_config


class RunnerModelLifecycleTest(unittest.TestCase):
    def test_one_process_per_model_across_strategies_and_no_start_on_resume(self):
        events = []
        @asynccontextmanager
        async def session(config, output_dir):
            events.append(('start', config.name))
            try:
                yield config.model_copy(update={'base_url':'http://127.0.0.1:12345/v1'})
            finally:
                events.append(('stop', config.name))
        def caller(config):
            self.assertEqual(config.base_url, 'http://127.0.0.1:12345/v1')
            c = mock.AsyncMock()
            c.call.return_value = 'A'
            return c
        with tempfile.TemporaryDirectory() as tmp:
            config = _build_config(Path(tmp))
            config.models = [ModelConfig(name=name,backend='local',model_name='Qwen/Qwen3-VL-8B-Instruct') for name in ('first','second')]
            config.strategies.append(config.strategies[0].model_copy(update={'name':'second-strategy'}))
            with (mock.patch('rtv_eval.runner._load_benchmark',return_value=_FakeBenchmark()),
                  mock.patch('rtv_eval.runner.VideoPreprocessor',_FakePreprocessor),
                  mock.patch('rtv_eval.runner._load_caller',side_effect=caller) as create_caller,
                  mock.patch('rtv_eval.runner.model_session',session)):
                runner = Runner(config)
                try:
                    asyncio.run(runner.run())
                    self.assertEqual(events,[('start','first'),('stop','first'),('start','second'),('stop','second')])
                    self.assertEqual(create_caller.call_count,4)
                    snapshots = runner.db.conn.execute('SELECT config_json FROM runs').fetchall()
                    self.assertTrue(all(json.loads(row[0])['model']['base_url'] is None for row in snapshots))
                    asyncio.run(runner.run())
                    self.assertEqual(len(events),4)
                    self.assertEqual(create_caller.call_count,4)
                finally:
                    runner.db.close()

    def test_initialization_error_still_exits_model_session(self):
        events=[]
        @asynccontextmanager
        async def session(config, output_dir):
            events.append('start')
            try:
                yield config
            finally:
                events.append('stop')
        with tempfile.TemporaryDirectory() as tmp:
            config=_build_config(Path(tmp))
            with (mock.patch('rtv_eval.runner._load_benchmark',return_value=_FakeBenchmark()),
                  mock.patch('rtv_eval.runner.VideoPreprocessor',_FakePreprocessor),
                  mock.patch('rtv_eval.runner._load_caller',side_effect=RuntimeError('caller failed')),
                  mock.patch('rtv_eval.runner.model_session',session)):
                runner=Runner(config)
                try:
                    with self.assertRaisesRegex(RuntimeError,'caller failed'):
                        asyncio.run(runner.run())
                    self.assertEqual(events,['start','stop'])
                finally:
                    runner.db.close()

    def test_empty_benchmark_does_not_start_a_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            config=_build_config(Path(tmp))
            benchmark=_FakeBenchmark()
            benchmark.load=lambda config: []
            with (mock.patch('rtv_eval.runner._load_benchmark',return_value=benchmark),
                  mock.patch('rtv_eval.runner.model_session') as session):
                runner=Runner(config)
                try:
                    asyncio.run(runner.run())
                    session.assert_not_called()
                finally:
                    runner.db.close()
