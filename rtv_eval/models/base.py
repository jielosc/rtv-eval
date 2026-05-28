from __future__ import annotations

from abc import ABC, abstractmethod


class ModelCaller(ABC):
    @abstractmethod
    async def call(self, question: str, frame_b64_list: list[str]) -> str:
        """Send a question with video frames to the model, return raw response text."""
        ...
