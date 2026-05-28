from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from rtv_eval.config import FrameSamplingConfig

logger = logging.getLogger(__name__)


def sample_frames(video_path: Path, config: FrameSamplingConfig) -> list[np.ndarray]:
    """Extract frames from a video according to the sampling config.

    Returns a list of RGB numpy arrays (H, W, 3) uint8.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    if total_frames <= 0:
        # Some videos report 0 frame count; fall back to reading until EOF
        cap.release()
        return _read_all_frames(video_path)

    if config.mode == "fps":
        interval = max(1, int(source_fps / config.fps))
        indices = list(range(0, total_frames, interval))
    else:  # count
        n = min(config.count, total_frames)
        if n <= 0:
            cap.release()
            return []
        indices = np.linspace(0, total_frames - 1, n, dtype=int).tolist()

    frames = _extract_at_indices(cap, indices)
    cap.release()
    return frames


def _extract_at_indices(cap: cv2.VideoCapture, indices: list[int]) -> list[np.ndarray]:
    """Extract frames at specific indices using sequential scan."""
    frames: list[np.ndarray] = []
    idx_set = set(indices)
    frame_no = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_no in idx_set:
            # cv2 reads BGR, convert to RGB
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_no += 1
        if len(frames) == len(indices):
            break

    return frames


def _read_all_frames(video_path: Path) -> list[np.ndarray]:
    """Fallback: read all frames when frame count is unavailable."""
    cap = cv2.VideoCapture(str(video_path))
    frames: list[np.ndarray] = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames
