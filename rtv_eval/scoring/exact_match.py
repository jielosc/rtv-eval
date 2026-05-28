from __future__ import annotations

import re

from rtv_eval.scoring.base import Scorer

_LETTER_RE = re.compile(r"\b([A-Da-d])\b")


class ExactMatchScorer(Scorer):
    def score(self, raw_response: str, ground_truth: str) -> tuple[str, bool]:
        extracted = self._extract_letter(raw_response)
        correct = extracted == ground_truth.upper()
        return extracted, correct

    @staticmethod
    def _extract_letter(response: str) -> str:
        """Extract a single letter A-D from a model's response."""
        response = response.strip()
        if not response:
            return ""

        # Try regex for standalone letter
        match = _LETTER_RE.search(response)
        if match:
            return match.group(1).upper()

        # Fallback: first character if it's a valid option
        if response[0].upper() in "ABCD":
            return response[0].upper()

        return ""
