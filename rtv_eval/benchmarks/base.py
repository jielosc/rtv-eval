from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class QAEntry:
    uid: str
    video_path: str
    question: str
    answer: str | None              # ground truth letter, or None for test set
    metadata: dict = field(default_factory=dict)


class Benchmark(ABC):
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def load(self, config) -> list[QAEntry]: ...

    @abstractmethod
    def resolve_video_path(self, video_path: str) -> Path: ...
