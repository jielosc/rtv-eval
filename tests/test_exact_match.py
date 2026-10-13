from __future__ import annotations

import unittest

from rtv_eval.scoring.exact_match import ExactMatchScorer


class ExactMatchTest(unittest.TestCase):
    def test_accepts_only_a_single_uppercase_option(self):
        scorer = ExactMatchScorer()
        self.assertEqual(scorer.score(' \nC\t', 'C'), ('C', True))
        self.assertEqual(scorer.score('B', 'C'), ('B', False))
        for response in (
            '', 'a', 'c', 'E', 'AB', 'A or C', 'A. Tennis', '**B**',
            'The person is holding a tool. The correct answer is C.',
            'In the video, a person is working.',
            'A. Red\nB. Blue\nThe answer is B.',
            'Because the person is running.',
        ):
            with self.subTest(response=response):
                self.assertEqual(scorer.score(response, 'A'), ('', False))

    def test_rejects_option_absent_from_question(self):
        scorer = ExactMatchScorer()
        self.assertEqual(scorer.score('D', 'B', valid_options=['A', 'B', 'C']), ('', False))
        self.assertEqual(scorer.score('B', 'B', valid_options=['A', 'B', 'C']), ('B', True))


if __name__ == '__main__':
    unittest.main()
