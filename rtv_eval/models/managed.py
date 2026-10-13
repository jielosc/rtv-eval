"""Own the lifetime of a model process for one model's evaluation strategies."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from contextlib import asynccontextmanager
from pathlib import Path

from rtv_eval.config import ModelConfig

logger = logging.getLogger(__name__)


def _available_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def _health(url):
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(url, timeout=1) as response:
            return json.load(response)
    except (OSError, ValueError, urllib.error.URLError):
        return {}


def _stop(process, timeout):
    if process.poll() is None:
        if os.name == 'posix':
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        else:
            process.terminate()
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            if os.name == 'posix':
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                process.kill()
            process.wait()


@asynccontextmanager
async def model_session(config: ModelConfig, output_dir: Path):
    if config.backend != 'local':
        yield config
        return

    local = config.local
    port = _available_port()
    url = f'http://127.0.0.1:{port}'
    output_dir.mkdir(parents=True, exist_ok=True)
    name = re.sub(r'[^a-zA-Z0-9_.-]', '_', config.name)
    log_path = output_dir / f'{name}.model.log'
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join(filter(None, [str(Path(__file__).resolve().parents[2]), env.get('PYTHONPATH')]))
    env['RTV_LOCAL_MODEL_CONFIG'] = json.dumps({
        'model': config.model_name, 'device': local.device, 'dtype': local.dtype,
        'attn_implementation': local.attn_implementation,
        'cache_dir': str(Path(local.cache_dir).expanduser().resolve()) if local.cache_dir else str(Path.home() / '.cache/huggingface/hub'),
    })
    with log_path.open('w', encoding='utf-8') as log:
        process = subprocess.Popen(
            [local.python_executable or sys.executable, '-m', 'rtv_eval.serving.api_server', '--port', str(port)],
            env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=(os.name == 'posix'),
        )
        try:
            deadline = time.monotonic() + local.startup_timeout
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f'Local model failed to start; see {log_path}')
                health = await asyncio.to_thread(_health, f'{url}/healthz')
                if health.get('pid') == process.pid and health.get('loaded') and health.get('model') == config.model_name:
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError(f'Local model startup timed out; see {log_path}')
                await asyncio.sleep(0.2)
            logger.info('Local model ready: %s (PID %s)', config.name, process.pid)
            # Keep the temporary port out of the saved experiment configuration.
            yield config.model_copy(update={'base_url': f'{url}/v1'})
        finally:
            await asyncio.to_thread(_stop, process, local.shutdown_timeout)
            logger.info('Local model stopped: %s', config.name)
