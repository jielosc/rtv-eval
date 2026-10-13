from __future__ import annotations

from collections.abc import Collection

from rtv_eval.scoring.base import Scorer

class ExactMatchScorer(Scorer):
    def score(
        self, raw_response: str, ground_truth: str,
        *, valid_options: Collection[str] = ("A", "B", "C", "D"),
    ) -> tuple[str, bool]:
        extracted = self._extract_letter(raw_response, valid_options=valid_options)
        correct = bool(extracted) and extracted == ground_truth.upper()
        return extracted, correct

    @staticmethod
    def _extract_letter(
        response: str, *, valid_options: Collection[str] = ("A", "B", "C", "D"),
    ) -> str:
        """Accept exactly one uppercase option; malformed output stays invalid."""
        response = response.strip()
        if len(response) == 1 and response in "ABCD" and response in valid_options:
            return response
        return ""
