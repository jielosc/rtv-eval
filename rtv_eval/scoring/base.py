from __future__ import annotations

from abc import ABC, abstractmethod


class Scorer(ABC):
    @abstractmethod
    def score(self, raw_response: str, ground_truth: str) -> tuple[str, bool]:
        """Extract answer from raw response and compare to ground truth.

        Returns (extracted_answer, is_correct).
        """
        ...
