from __future__ import annotations

import cv2
import numpy as np

from rtv_eval.config import ResolutionConfig

PRESETS: dict[str, int] = {
    "360p": 360,
    "480p": 480,
    "720p": 720,
    "1080p": 1080,
}


def resize_frame(frame: np.ndarray, config: ResolutionConfig) -> np.ndarray:
    """Resize a frame according to the resolution config, preserving aspect ratio."""
    h, w = frame.shape[:2]

    if config.mode == "preset":
        target = PRESETS[config.preset]
        if h > w:  # portrait: constrain by width
            if w <= target:
                return frame
            scale = target / w
            new_w, new_h = target, int(h * scale)
        else:  # landscape: constrain by height
            if h <= target:
                return frame
            scale = target / h
            new_w, new_h = int(w * scale), target
    else:
        max_w, max_h = config.width, config.height
        scale = min(max_w / w, max_h / h)
        if scale >= 1.0:
            return frame
        new_w, new_h = int(w * scale), int(h * scale)

    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
