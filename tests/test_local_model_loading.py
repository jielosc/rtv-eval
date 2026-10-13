from __future__ import annotations

import tempfile
import types
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest import mock

from rtv_eval.serving.qwen3vl_service import Qwen3VLService, ServiceConfig, resolve_model_source


class LocalModelLoadingTest(unittest.TestCase):
    def test_cpu_inference_uses_requested_generation_parameters(self):
        model = mock.Mock(device='cpu')
        model.to.return_value = model
        model.generate.return_value = [[1, 2, 3]]
        processor = mock.MagicMock()
        processor.return_value = {'input_ids': [[1, 2]]}
        processor.batch_decode.return_value = ['B']
        transformers = types.SimpleNamespace(
            AutoConfig=mock.Mock(), AutoProcessor=mock.Mock(), Qwen3VLForConditionalGeneration=mock.Mock())
        transformers.AutoConfig.from_pretrained.return_value.model_type = 'qwen3_vl'
        transformers.AutoProcessor.from_pretrained.return_value = processor
        transformers.Qwen3VLForConditionalGeneration.from_pretrained.return_value = model
        torch = types.SimpleNamespace(bfloat16='bf16',float16='fp16',float32='fp32',
                                      is_tensor=lambda _:False,inference_mode=nullcontext,
                                      cuda=types.SimpleNamespace(is_available=lambda:False))
        with mock.patch.dict('sys.modules', {'torch':torch,'transformers':transformers}):
            service = Qwen3VLService(ServiceConfig(device='cpu'))
            try:
                service.load()
                self.assertEqual(service.infer_messages([{'role':'user','content':[]}],max_new_tokens=16,temperature=0),'B')
                self.assertFalse(model.generate.call_args.kwargs['do_sample'])
                self.assertEqual(model.generate.call_args.kwargs['max_new_tokens'],16)
                service.infer_messages([],max_new_tokens=8,temperature=0.5)
                self.assertTrue(model.generate.call_args.kwargs['do_sample'])
                self.assertEqual(model.generate.call_args.kwargs['temperature'],0.5)
                transformers.Qwen3VLForConditionalGeneration.from_pretrained.assert_called_once()
                transformers.AutoProcessor.from_pretrained.assert_called_once()
            finally:
                service.close()
            self.assertIsNone(service.model)
            self.assertIsNone(service.processor)

    def test_incomplete_cache_falls_back_to_model_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp)/'models--Qwen--Qwen3-VL-8B-Instruct').mkdir()
            self.assertEqual(resolve_model_source('Qwen/Qwen3-VL-8B-Instruct',tmp),'Qwen/Qwen3-VL-8B-Instruct')
