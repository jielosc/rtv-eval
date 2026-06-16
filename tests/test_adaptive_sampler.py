from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from rtv_eval.config import AdaptiveConfig, AdaptiveTier, ResolutionConfig, StrategyConfig
from rtv_eval.strategy.adaptive_sampler import _select_tier, sample_adaptive
from rtv_eval.strategy.preprocessor import VideoPreprocessor


def _write_video(path: Path, fps: int = 10) -> None:
    """Write a 4s test video: 2s static, then 2s of high per-frame change."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w, h = 320, 240
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    # 2 seconds static (solid gray)
    static = np.full((h, w, 3), 100, dtype=np.uint8)
    for _ in range(2 * fps):
        writer.write(static)
    # 2 seconds high-motion (full-frame random noise each frame)
    rng = np.random.default_rng(0)
    for _ in range(2 * fps):
        writer.write(rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8))
    writer.release()


def _tiers() -> list[AdaptiveTier]:
    return [
        AdaptiveTier(name="low", min_score=0.0, fps=1.0,
                     resolution=ResolutionConfig(mode="preset", preset="720p"), quality=90),
        AdaptiveTier(name="mid", min_score=0.15, fps=2.0,
                     resolution=ResolutionConfig(mode="preset", preset="480p"), quality=75),
        AdaptiveTier(name="high", min_score=0.4, fps=4.0,
                     resolution=ResolutionConfig(mode="preset", preset="360p"), quality=60),
    ]


class SelectTierTest(unittest.TestCase):
    def test_select_tier_picks_highest_cleared_bound(self) -> None:
        tiers = _tiers()
        self.assertEqual(_select_tier(0.0, tiers).name, "low")
        self.assertEqual(_select_tier(0.2, tiers).name, "mid")
        self.assertEqual(_select_tier(0.9, tiers).name, "high")


class AdaptiveSamplerTest(unittest.TestCase):
    def test_high_motion_uploads_more_frames(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            vp = Path(tmp) / "clip.mp4"
            _write_video(vp)
            cfg = AdaptiveConfig(
                analysis_fps=5.0,
                window_seconds=1.0,
                tiers=_tiers(),  # defaults: pure D (w_ssim=0)
            )
            frames = sample_adaptive(vp, cfg)
            self.assertGreater(len(frames), 0)

            low_count = sum(1 for f in frames if f.tier_name == "low")
            high_count = sum(1 for f in frames if f.tier_name == "high")
            # Static half should land in "low", noisy half in "high".
            self.assertGreater(low_count, 0)
            self.assertGreater(high_count, 0)
            # High tier uploads at 4x the fps, so it should contribute more frames.
            self.assertGreater(high_count, low_count)

    def test_preprocessor_adaptive_path_returns_base64(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            vp = Path(tmp) / "clip.mp4"
            _write_video(vp)
            strategy = StrategyConfig(
                name="adaptive_test",
                adaptive=AdaptiveConfig(analysis_fps=5.0, tiers=_tiers()),
            )
            result = VideoPreprocessor(strategy).process(vp)
            self.assertGreater(len(result), 0)
            self.assertTrue(all(isinstance(s, str) and s for s in result))


if __name__ == "__main__":
    unittest.main()
