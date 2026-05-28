from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class Strategy(ABC):
    @abstractmethod
    def process(self, video_path: Path) -> list[str]:
        """Process a video file and return base64-encoded JPEG images."""
        ...
