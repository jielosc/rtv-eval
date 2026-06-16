from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from rtv_eval.config import AdaptiveConfig, AdaptiveTier, ResolutionConfig
from rtv_eval.strategy.change_score import change_score_norm

logger = logging.getLogger(__name__)


class AdaptiveFrame:
    """An upload frame plus the per-frame encoding decided by its window's tier."""

    __slots__ = ("frame", "resolution", "quality", "tier_name")

    def __init__(
        self, frame: np.ndarray, resolution: ResolutionConfig, quality: int, tier_name: str
    ) -> None:
        self.frame = frame                # RGB uint8 (H, W, 3)
        self.resolution = resolution
        self.quality = quality
        self.tier_name = tier_name


def _select_tier(score: float, tiers: list[AdaptiveTier]) -> AdaptiveTier:
    """Pick the highest-bound tier whose min_score <= score.

    Tiers are read in config order; we keep the last one whose lower bound is
    cleared. Falls back to the first tier if none match (score below all bounds).
    """
    chosen = tiers[0]
    for tier in tiers:
        if score >= tier.min_score:
            chosen = tier
    return chosen


def _analysis_scores(
    cap: cv2.VideoCapture,
    total_frames: int,
    source_fps: float,
    config: AdaptiveConfig,
) -> dict[int, float]:
    """Decode the low-res analysis stream and score change at each analysis step.

    Returns a map from source frame index -> change score C computed against the
    previous analysis frame. The decision is causal: C at index i reflects the
    transition from the previous analysed frame into frame i.
    """
    step = max(1, int(round(source_fps / config.analysis_fps)))
    analysis_indices = list(range(0, total_frames, step))
    idx_set = set(analysis_indices)

    scores: dict[int, float] = {}
    prev_gray: np.ndarray | None = None
    ema: float | None = None
    frame_no = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_no in idx_set:
            small = cv2.resize(
                frame,
                (config.analysis_width, config.analysis_height),
                interpolation=cv2.INTER_AREA,
            )
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                c = change_score_norm(
                    prev_gray,
                    gray,
                    config.w_frame_diff,
                    config.w_ssim,
                    config.d_norm_lo,
                    config.d_norm_hi,
                    config.s_norm_lo,
                    config.s_norm_hi,
                )
                if config.smoothing > 0.0:
                    ema = c if ema is None else config.smoothing * ema + (1 - config.smoothing) * c
                    c = ema
                scores[frame_no] = c
            prev_gray = gray
        frame_no += 1

    return scores


def _window_tiers(
    scores: dict[int, float],
    total_frames: int,
    source_fps: float,
    config: AdaptiveConfig,
) -> list[tuple[int, int, AdaptiveTier]]:
    """Group analysis scores into fixed-length windows and assign a tier to each.

    Returns a list of (window_start_idx, window_end_idx_exclusive, tier).
    """
    window_frames = max(1, int(round(config.window_seconds * source_fps)))
    windows: list[tuple[int, int, AdaptiveTier]] = []

    for start in range(0, total_frames, window_frames):
        end = min(start + window_frames, total_frames)
        window_scores = [c for idx, c in scores.items() if start <= idx < end]
        # Empty window (no analysis transition fell inside): treat as no change.
        mean_c = float(np.mean(window_scores)) if window_scores else 0.0
        tier = _select_tier(mean_c, config.tiers)
        windows.append((start, end, tier))

    return windows


def _plan_upload_indices(
    windows: list[tuple[int, int, AdaptiveTier]],
    source_fps: float,
) -> dict[int, AdaptiveTier]:
    """Decide which source frame indices to upload and at which tier.

    Within each window, upload at the tier's fps, evenly spaced across the
    window's frame span (at least one frame per non-empty window).
    """
    plan: dict[int, AdaptiveTier] = {}
    for start, end, tier in windows:
        span = end - start
        if span <= 0:
            continue
        duration = span / source_fps
        n = max(1, int(round(tier.fps * duration)))
        n = min(n, span)
        offsets = np.linspace(0, span - 1, n, dtype=int)
        for off in offsets:
            plan[start + int(off)] = tier
    return plan


def sample_adaptive(video_path: Path, config: AdaptiveConfig) -> list[AdaptiveFrame]:
    """Content-adaptive sampling: score change on a low-res stream, then upload
    each window at its tier's fps / resolution / quality.

    Returns AdaptiveFrame objects (RGB frame + per-frame encoding) in time order.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    if total_frames <= 0:
        cap.release()
        logger.warning("Adaptive sampling needs a known frame count; %s reports 0.", video_path)
        return []

    # Pass 1: low-res analysis stream -> per-window tier decisions.
    scores = _analysis_scores(cap, total_frames, source_fps, config)
    windows = _window_tiers(scores, total_frames, source_fps, config)
    plan = _plan_upload_indices(windows, source_fps)
    cap.release()

    if not plan:
        return []

    # Pass 2: extract the planned upload frames at full source resolution.
    cap = cv2.VideoCapture(str(video_path))
    upload: list[AdaptiveFrame] = []
    wanted = set(plan)
    frame_no = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_no in wanted:
            tier = plan[frame_no]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            upload.append(AdaptiveFrame(rgb, tier.resolution, tier.quality, tier.name))
        frame_no += 1
        if len(upload) == len(wanted):
            break
    cap.release()

    logger.debug(
        "Adaptive %s: %d windows, %d upload frames (tiers: %s)",
        video_path.name,
        len(windows),
        len(upload),
        {t: sum(1 for f in upload if f.tier_name == t) for t in {f.tier_name for f in upload}},
    )
    return upload
