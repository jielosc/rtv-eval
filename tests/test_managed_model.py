from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from rtv_eval.config import ModelConfig
from rtv_eval.models import managed
from rtv_eval.models.openai_compat import OpenAICompatCaller


FAKE_SERVER = '''
import argparse,json,os,signal
from http.server import BaseHTTPRequestHandler,HTTPServer
args=argparse.ArgumentParser();args.add_argument('--port',type=int);port=args.parse_args().port
class Handler(BaseHTTPRequestHandler):
 def do_GET(self):
  payload={'pid':os.getpid()+int(os.getenv('WRONG_PID','0')),'loaded':not os.getenv('NEVER_READY'),'model':json.loads(os.environ['RTV_LOCAL_MODEL_CONFIG'])['model']}
  self.send_response(200);self.end_headers();self.wfile.write(json.dumps(payload).encode())
 def do_POST(self):
  payload=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
  assert payload['model']=='fake-model'
  response={'id':'test','object':'chat.completion','created':0,'model':'fake-model','choices':[{'index':0,'message':{'role':'assistant','content':'B'},'finish_reason':'stop'}]}
  self.send_response(200);self.send_header('Content-Type','application/json');self.end_headers();self.wfile.write(json.dumps(response).encode())
 def log_message(self,*args): pass
if os.getenv('IGNORE_TERM'): signal.signal(signal.SIGTERM,signal.SIG_IGN)
HTTPServer(('127.0.0.1',port),Handler).serve_forever()
'''


class ManagedModelTest(unittest.IsolatedAsyncioTestCase):
    async def _exercise(self, *, action=None, script=FAKE_SERVER, **env):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            server = path / 'fake_server.py'; server.write_text(script)
            config = ModelConfig(name='local-test', backend='local', model_name='fake-model',
                                 local={'startup_timeout': 2, 'shutdown_timeout': 0.2})
            real_popen = subprocess.Popen
            processes = []
            def launch(command, **kwargs):
                processes.append(real_popen([sys.executable, str(server), *command[-2:]], **kwargs))
                return processes[-1]
            try:
                with mock.patch.dict('os.environ', env), mock.patch.object(managed.subprocess, 'Popen', side_effect=launch):
                    async with managed.model_session(config, path) as active:
                        self.assertIsNone(config.base_url)
                        self.assertTrue(active.base_url.startswith('http://127.0.0.1:'))
                        if action:
                            await action(active)
                return processes[-1].returncode
            finally:
                self.assertTrue(processes)
                self.assertIsNotNone(processes[-1].poll(), 'Model process leaked')

    async def test_starts_real_process_and_calls_api_without_api_key(self):
        async def evaluate(active):
            caller = OpenAICompatCaller(active)
            try:
                self.assertEqual(await caller.call('Choose one option', []), 'B')
            finally:
                await caller.client.close()
        await self._exercise(action=evaluate)

    async def test_stops_after_evaluation_exception(self):
        async def fail(active):
            raise ValueError('evaluation failed')
        with self.assertRaisesRegex(ValueError, 'evaluation failed'):
            await self._exercise(action=fail)

    async def test_stops_after_cancellation(self):
        async def cancel(active):
            asyncio.current_task().cancel()
            await asyncio.sleep(0)
        with self.assertRaises(asyncio.CancelledError):
            await self._exercise(action=cancel)

    async def test_startup_timeout_cleans_up(self):
        with self.assertRaises(TimeoutError):
            await self._exercise(NEVER_READY='1')

    async def test_does_not_accept_another_process_as_ready(self):
        with self.assertRaises(TimeoutError):
            await self._exercise(WRONG_PID='1')

    async def test_startup_failure_reports_log(self):
        with self.assertRaisesRegex(RuntimeError, 'model.log'):
            await self._exercise(script='raise SystemExit(2)')

    @unittest.skipUnless(sys.platform != 'win32', 'POSIX signal test')
    async def test_kills_process_that_ignores_shutdown(self):
        code = await self._exercise(IGNORE_TERM='1')
        self.assertEqual(code, -9)

    async def test_remote_backend_does_not_spawn(self):
        config = ModelConfig(name='remote', model_name='model', base_url='http://example.invalid/v1')
        with mock.patch.object(managed.subprocess, 'Popen') as spawn:
            async with managed.model_session(config, Path('/tmp/unused')) as active:
                self.assertIs(active, config)
            spawn.assert_not_called()

    def test_local_configuration_rejects_external_endpoint(self):
        with self.assertRaises(ValueError):
            ModelConfig(name='local', backend='local', model_name='model', base_url='http://external/v1')
