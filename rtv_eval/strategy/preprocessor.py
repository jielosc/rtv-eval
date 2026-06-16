from __future__ import annotations

import base64
import io
import logging
from pathlib import Path

import numpy as np
from PIL import Image

from rtv_eval.config import StrategyConfig
from rtv_eval.strategy.adaptive_sampler import sample_adaptive
from rtv_eval.strategy.base import Strategy
from rtv_eval.strategy.frame_sampler import sample_frames
from rtv_eval.strategy.resizer import resize_frame

logger = logging.getLogger(__name__)


class VideoPreprocessor(Strategy):
    def __init__(self, config: StrategyConfig):
        self.config = config

    def process(self, video_path: Path) -> list[str]:
        """Process a video: sample frames, resize, encode as base64 JPEG.

        Returns a list of base64-encoded JPEG strings ready for API consumption.
        """
        if self.config.adaptive is not None:
            return self._process_adaptive(video_path)
        return self._process_static(video_path)

    def _process_static(self, video_path: Path) -> list[str]:
        frames = sample_frames(video_path, self.config.frame_sampling)
        if not frames:
            logger.warning("No frames extracted from %s", video_path)
            return []

        result: list[str] = []
        for frame in frames:
            resized = resize_frame(frame, self.config.resolution)
            b64 = _encode_jpeg(resized)
            result.append(b64)

        return result

    def _process_adaptive(self, video_path: Path) -> list[str]:
        """Content-adaptive path: each frame carries its own resolution + quality."""
        frames = sample_adaptive(video_path, self.config.adaptive)
        if not frames:
            logger.warning("No frames extracted from %s", video_path)
            return []

        result: list[str] = []
        for af in frames:
            resized = resize_frame(af.frame, af.resolution)
            b64 = _encode_jpeg(resized, quality=af.quality)
            result.append(b64)

        return result


def _encode_jpeg(frame: np.ndarray, quality: int = 85) -> str:
    """Encode an RGB numpy array as a base64 JPEG string."""
    pil_img = Image.fromarray(frame)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")
