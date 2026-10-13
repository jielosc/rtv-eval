from __future__ import annotations

import base64
import importlib.util
import io
import unittest
from unittest import mock

from PIL import Image


@unittest.skipUnless(importlib.util.find_spec('fastapi'), 'Optional local dependencies not installed')
class LocalInferenceApiTest(unittest.TestCase):
    def test_real_sdk_payload_and_lifespan_without_gpu(self):
        from fastapi.testclient import TestClient
        from rtv_eval.serving import api_server
        from rtv_eval.serving.qwen3vl_service import ServiceConfig

        service = mock.Mock()
        service.infer_messages.return_value = 'B'
        buffer = io.BytesIO()
        Image.new('RGB', (2, 2)).save(buffer, format='JPEG')
        image_url = 'data:image/jpeg;base64,' + base64.b64encode(buffer.getvalue()).decode()
        payload = {'model':'test', 'messages':[
            {'role':'system','content':'Choose a letter'},
            {'role':'user','content':[{'type':'image_url','image_url':{'url':image_url}},
                                      {'type':'text','text':'Question?'}]}],
                   'max_tokens':8, 'temperature':0}
        with mock.patch.object(api_server, 'Qwen3VLService', return_value=service):
            with TestClient(api_server.build_app(ServiceConfig(model='test'))) as client:
                service.load.assert_called_once()
                self.assertTrue(client.get('/healthz').json()['loaded'])
                response = client.post('/v1/chat/completions', json=payload)
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.json()['choices'][0]['message']['content'], 'B')
                kwargs = service.infer_messages.call_args.kwargs
                self.assertEqual(kwargs['max_new_tokens'], 8)
                self.assertEqual(kwargs['temperature'], 0)
                self.assertEqual(len(kwargs['images']), 1)
                self.assertEqual(kwargs['messages'][0]['role'], 'system')
                for url in ('/', '/realtime', '/ws/realtime'):
                    self.assertEqual(client.get(url).status_code, 404)
                bad = {**payload, 'messages':[{'role':'user','content':[{'type':'image_url','image_url':{'url':'data:image/jpeg;base64,!bad'}}]}]}
                self.assertEqual(client.post('/v1/chat/completions', json=bad).status_code, 400)
                self.assertEqual(client.post('/v1/chat/completions', json={**payload,'model':'other'}).status_code, 404)
            service.close.assert_called_once()

    def test_failed_load_also_closes_service(self):
        from fastapi.testclient import TestClient
        from rtv_eval.serving import api_server
        from rtv_eval.serving.qwen3vl_service import ServiceConfig
        service = mock.Mock()
        service.load.side_effect = RuntimeError('cannot load weights')
        with mock.patch.object(api_server,'Qwen3VLService',return_value=service):
            with self.assertRaisesRegex(RuntimeError, 'cannot load weights'):
                with TestClient(api_server.build_app(ServiceConfig())):
                    pass
        service.close.assert_called_once()
