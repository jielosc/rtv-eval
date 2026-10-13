from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rtv_eval.benchmarks.motionbench import MotionBench
from rtv_eval.config import BenchmarkConfig


class MotionBenchPromptTest(unittest.TestCase):
    def _load(self, question, answer='B', split='dev'):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / 'video_info.meta.jsonl').write_text(json.dumps({
                'qa': [{'uid': 'q1', 'question': question, 'answer': answer}],
                'video_path': 'video.mp4', 'video_info': {'duration': 1},
            }) + '\n')
            return MotionBench().load(BenchmarkConfig(
                name='motionbench', type='motionbench', data_dir=tmp,
                video_root=tmp, split=split,
            ))[0]

    def test_description_question_has_explicit_single_choice_contract(self):
        original = ('Please describe the detailed breakdown of the action in the video.\n'
                    'A. Turn, then reach.\nB. Reach, then turn.\nC. Wave.')
        entry = self._load(original)
        self.assertIn(original, entry.question)
        self.assertIn('single-choice', entry.question)
        self.assertIn('Output exactly one uppercase letter (A, B, C) and nothing else.', entry.question)
        self.assertEqual(entry.metadata['valid_options'], ['A', 'B', 'C'])
        self.assertNotIn('correct answer is B', entry.question)

    def test_test_split_has_same_prompt_without_ground_truth(self):
        question = 'What happens?\nA. Wave.\nB. Turn.'
        dev = self._load(question)
        test = self._load(question, answer='NA', split='test')
        self.assertEqual(dev.question, test.question)
        self.assertIsNone(test.answer)

    def test_rejects_malformed_options_or_ground_truth(self):
        for question, answer in [('Describe the video.', 'B'),
                                 ('Question?\nA. One.\nA. Two.', 'A'),
                                 ('Question?\nA. One.\nE. Two.', 'A'),
                                 ('Question?\nA. One.\nB. Two.', 'D')]:
            with self.subTest(question=question), self.assertRaises(ValueError):
                self._load(question, answer=answer)


if __name__ == '__main__':
    unittest.main()
